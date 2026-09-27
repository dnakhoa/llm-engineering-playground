"""The Checks CLI: grade an agent on every Spine module up to N.

    python -m checks --modules 3 --agent path/to/my_agent.py:run
    python -m checks --modules 1-6 --mode live --spend-cap 0.50
    python -m checks --mode live --model qwen3:8b --base-url http://localhost:11434/v1

Running module N runs the suites of modules 1 to N, because hardening an agent
must not silently break what it already did. Each Case runs once, however many
suites use it. A Check that holds on every Case (``checks.on_every_case``),
such as each part of the verdict, also grades every later suite's Cases, once
per Case, so a suite that adds a Case need not list those Checks again. The last line is the one to share: "Passed through module N of M."

Offline (the default) replays the reviewed recordings: no key, no network, and
live-only Checks such as judge rubrics are reported as skipped. Live mode calls
the model on the Reader's own key, under a Spend Cap that is printed before
anything runs. A run that reaches the cap stops, and reports what finished.
Nothing here ever asks a question on the terminal, so it runs the same in CI.

Without ``--model``, a live run loads the root ``.env`` and asks
``llm.default_model()``, the same policy ``ask()`` uses: ``LLM_MODEL``, a local
server (``LLM_PROVIDER=ollama``, ``OPENAI_BASE_URL``), or the cheapest model of
the provider whose key is set. An offline run replays recordings made on one
model, so it stays on that model and needs no key.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from company.runner import (
    DEFAULT_MODEL,
    LIVE,
    OFFLINE,
    Agent,
    Outcome,
    SpendCap,
    SpendCapReached,
    load_agent,
    load_case,
    run_case,
)
from company import tracing
from llm import load_registry, resolve_model
from llm.registry import LOCAL_PROVIDER, ModelSpec, local_model
from llm.replay import ReplayMismatchError
from llm.transport import TransportError

from . import is_live_only, is_needs_trace, is_on_every_case
from .judge import Judge, judging_with
from .suites import Check, Suite, discover_suites

#: The reference Flagship Agent at the latest Spine module's end state.
DEFAULT_AGENT = "flagship.observed:run"
DEFAULT_SPEND_CAP_USD = 1.00

#: The Reader's settings: the root .env that `cp .env.example .env` creates.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

PASS = "pass"
FAIL = "fail"
SKIP = "skip"
STOPPED = "stopped"
NOT_RUN = "not run"
ERROR = "error"

_LABELS = {
    PASS: "PASS",
    FAIL: "FAIL",
    SKIP: "SKIP",
    STOPPED: "STOP",
    NOT_RUN: "STOP",
    ERROR: "ERROR",
}


@dataclass(frozen=True)
class CheckRun:
    """One Check on one Case, and what became of it."""

    module: int
    case_id: str
    name: str
    status: str
    detail: str


@dataclass
class Report:
    """A whole run: every Check's result, the spend, and how far it got."""

    through: int
    results: List[CheckRun] = field(default_factory=list)
    spend_cap_usd: Optional[float] = None
    #: What the run billed to the Reader's key. Always 0 Offline.
    spent_usd: float = 0.0
    #: Offline only: the recorded usage at registry prices, what the same run
    #: would have cost live. The Spend Cap stops an Offline run at this cost,
    #: where the live run would stop, but none of it is spent.
    would_have_cost_usd: float = 0.0
    stopped_by_spend_cap: bool = False
    stopped_reason: Optional[str] = None
    passed_through: int = 0

    @property
    def passed(self) -> bool:
        return self.passed_through == self.through and self.stopped_reason is None


# ── Arguments ─────────────────────────────────────────────────────────────────


def parse_modules(text: str) -> int:
    """``"3"``, ``"1-3"`` or ``"1..3"``: the last module of a run that starts at 1."""
    match = re.fullmatch(r"\s*(?:(\d+)\s*(?:-|\.\.)\s*)?(\d+)\s*", text)
    if not match:
        raise ValueError("A module range looks like 3, 1-3 or 1..3, not {!r}.".format(text))
    start = int(match.group(1) or 1)
    end = int(match.group(2))
    if start != 1:
        raise ValueError(
            "A run starts at module 1, because module {} is graded together with "
            "every earlier module's Checks. Use 1-{}.".format(end, end)
        )
    if end < 1:
        raise ValueError("Modules are numbered from 1.")
    return end


def _usd(value: float) -> str:
    if value >= 0.01 and round(value, 2) == value:
        return "${:.2f}".format(value)
    return "${:.6f}".format(value).rstrip("0").rstrip(".")


def _label(check: Check) -> str:
    return check.__name__.replace("_", " ")


# ── The run ───────────────────────────────────────────────────────────────────


def _price_line(spec: ModelSpec) -> str:
    cached = spec.cache_read_price_per_mtok
    return "{} in, {} cached, {} out per million tokens".format(
        _usd(spec.input_price_per_mtok),
        _usd(spec.input_price_per_mtok if cached is None else cached),
        _usd(spec.output_price_per_mtok),
    )


def run_checks(
    *,
    through: int,
    agent: Union[str, Agent] = DEFAULT_AGENT,
    mode: str = OFFLINE,
    spend_cap_usd: float = DEFAULT_SPEND_CAP_USD,
    model: str = DEFAULT_MODEL,
    base_url: Optional[str] = None,
    suites: Optional[Sequence[Suite]] = None,
    transport=None,
    out: Optional[Callable[[str], None]] = None,
) -> Report:
    """Run the suites of modules 1 to ``through`` against ``agent``.

    ``base_url`` points at a local OpenAI-compatible server serving ``model``;
    it needs no registry entry, is priced at zero, and runs live. ``suites``
    and ``transport`` replace the real ones, for tests.
    """
    write = out or (lambda line: print(line, flush=True))
    if mode not in (OFFLINE, LIVE):
        raise ValueError("mode is 'offline' or 'live', not {!r}.".format(mode))
    if base_url and mode == OFFLINE:
        raise ValueError(
            "A local server has no recordings to replay, so it runs live: "
            "add --mode live."
        )

    registry = load_registry()
    if base_url:
        registry = registry.with_model(local_model(model, base_url))
    spec = registry.get(model)

    selected = [s for s in (suites if suites is not None else discover_suites()) if s.module <= through]
    if not selected:
        raise ValueError("There are no Check suites for modules 1 to {}.".format(through))
    _require_tracing_for(selected)
    run_agent = load_agent(agent)
    agent_name = agent if isinstance(agent, str) else getattr(agent, "__name__", repr(agent))

    report = Report(through=through)
    write("Checks for modules 1 to {} · agent {} · {} · {}".format(
        through, agent_name, mode, model))
    cap: Optional[SpendCap] = None
    if spec.provider == LOCAL_PROVIDER:
        write("Spend Cap does not apply: {} at {} is a local server, priced at $0.".format(
            model, base_url))
    else:
        cap = SpendCap(spend_cap_usd)
        report.spend_cap_usd = cap.limit_usd
        write("Spend Cap: {} for this run, at registry prices for {} ({}).".format(
            _usd(cap.limit_usd), model, _price_line(spec)))
        if mode == OFFLINE:
            write("Offline: nothing is spent. Recorded usage is priced at these rates to "
                  "show what the run would cost live, and stops where the cap would.")

    # Live, a judge-rubric Check grades on the same model and key as the agent,
    # and its calls count against the same Spend Cap.
    judge: Optional[Judge] = None
    if mode == LIVE:
        judge = Judge(model=model, transport=transport, registry=registry, spend_cap=cap)

    outcomes: Dict[str, Outcome] = {}
    # Earlier suites' Checks that hold on every Case, such as each part of the
    # verdict: each later suite grades its Cases on them too, and on a Case
    # already graded on one (graded) it does not run again.
    carried: List[Check] = []
    graded: Set[Tuple[str, Check]] = set()
    for suite in selected:
        write("")
        write("Module {} · {}".format(suite.module, suite.title))
        for case_ref in suite.cases:
            case = load_case(case_ref)
            checks = _checks_for(case.id, carried, suite.checks, graded)
            if not checks:
                continue

            def emit(check: Check, status: str, detail: str, title: Optional[str] = None) -> None:
                report.results.append(
                    CheckRun(suite.module, case.id, check.__name__, status, detail))
                lines = detail.splitlines() or [""]
                write("  {:<5} {}: {}. {}".format(
                    _LABELS[status], case.id, title or _label(check), lines[0]).rstrip())
                for more in lines[1:]:
                    write("          " + more)

            if report.stopped_by_spend_cap or report.stopped_reason:
                for check in checks:
                    emit(check, NOT_RUN, "Not run: the run stopped before this Case.")
                continue
            if mode == OFFLINE and not case.recording:
                for check in checks:
                    emit(check, SKIP, "Case {} has no recording; run it live.".format(case.id))
                continue

            outcome = outcomes.get(case.id)
            if outcome is None:
                try:
                    outcome = run_case(
                        case, run_agent, model=model, mode=mode, transport=transport,
                        registry=registry, spend_cap=cap,
                    )
                except TransportError as error:
                    report.stopped_reason = str(error)
                    for check in checks:
                        emit(check, ERROR, str(error))
                    continue
                except ReplayMismatchError as error:
                    for check in checks:
                        emit(check, ERROR, str(error))
                    continue
                except Exception as error:  # the Reader's agent raised: grade it as such
                    for check in checks:
                        emit(check, ERROR, "The agent raised {}: {}".format(
                            type(error).__name__, error))
                    continue
                outcomes[case.id] = outcome

            if outcome.spend_cap_reached:
                report.stopped_by_spend_cap = True
                for check in checks:
                    emit(check, STOPPED, "Stopped mid-Case by the Spend Cap.")
                continue
            for check in checks:
                if mode == OFFLINE and is_live_only(check):
                    emit(check, SKIP, "Live only: it needs a model call no recording holds.")
                    continue
                try:
                    with judging_with(judge):
                        result = check(outcome)
                except SpendCapReached:  # a live-only Check's own model call
                    report.stopped_by_spend_cap = True
                    emit(check, STOPPED, "Stopped by the Spend Cap before its model call.")
                    continue
                except TransportError as error:
                    report.stopped_reason = str(error)
                    emit(check, ERROR, str(error))
                    continue
                emit(check, PASS if result.passed else FAIL, result.detail, result.name)
        carried.extend(
            check for check in suite.checks if is_on_every_case(check) and check not in carried)

    priced = cap.spent_usd if cap is not None else 0.0
    if mode == OFFLINE:
        report.would_have_cost_usd = priced
    else:
        report.spent_usd = priced
    report.passed_through = _passed_through(selected, report.results)

    write("")
    if report.stopped_by_spend_cap:
        unfinished = sum(1 for r in report.results if r.status in (STOPPED, NOT_RUN))
        if mode == OFFLINE:
            cost = "this run would have cost {} live, and nothing was spent".format(_usd(priced))
        else:
            cost = "spent {}".format(_usd(priced))
        write("Stopped by the Spend Cap of {}: {}. {} Checks did not finish.".format(
            _usd(cap.limit_usd if cap else 0.0), cost, unfinished))
    elif report.stopped_reason:
        write("Stopped: {}".format(report.stopped_reason))
    elif cap is not None and mode == OFFLINE:
        write("Would have cost {} live. Offline, nothing was spent.".format(_usd(priced)))
    elif cap is not None:
        write("Spent {} of the {} Spend Cap.".format(_usd(priced), _usd(cap.limit_usd)))
    write("Passed through module {} of {}.".format(report.passed_through, through))
    return report


def _require_tracing_for(suites: Sequence[Suite]) -> None:
    """Refuse, before anything runs, a run whose Checks read traces when the
    OpenTelemetry SDK that records them is not installed."""
    if tracing.available():
        return
    traced = next((suite for suite in suites
                   if any(is_needs_trace(check) for check in suite.checks)), None)
    if traced is not None:
        earlier = traced.module - 1
        tracing.require(
            "Module {} ({}) grades each Case's OpenTelemetry trace".format(
                traced.module, traced.title),
            "Modules 1 to {} run without it: --modules {}.".format(earlier, earlier)
            if earlier else "")


def _checks_for(
    case_id: str,
    carried: Sequence[Check],
    own: Sequence[Check],
    graded: Set[Tuple[str, Check]],
) -> List[Check]:
    """The Checks one suite grades a Case on: those earlier suites carry to
    every Case that the Case has not been graded on yet, then the suite's own."""
    checks = [
        check for check in carried if (case_id, check) not in graded and check not in own
    ] + list(own)
    graded.update((case_id, check) for check in checks)
    return checks


def _passed_through(suites: Sequence[Suite], results: Sequence[CheckRun]) -> int:
    """The last module N such that modules 1 to N all passed. A skip is not a failure."""
    passed = 0
    for suite in suites:
        mine = [r for r in results if r.module == suite.module]
        if any(r.status not in (PASS, SKIP) for r in mine):
            break
        passed = suite.module
    return passed


# ── Command line ──────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m checks",
        description="Run the Checks for Spine modules 1 to N against an agent.",
    )
    parser.add_argument(
        "--modules", default=None,
        help="the last module to grade: 3, 1-3 or 1..3 (default: every module)")
    parser.add_argument(
        "--agent", default=DEFAULT_AGENT,
        help="package.module:function or path/to/agent.py:function "
             "(default: {})".format(DEFAULT_AGENT))
    parser.add_argument(
        "--mode", choices=(OFFLINE, LIVE), default=OFFLINE,
        help="offline replays recordings for free; live calls the model on your key")
    parser.add_argument(
        "--spend-cap", type=float, default=DEFAULT_SPEND_CAP_USD, metavar="USD",
        help="stop the run once it has spent this much (default: {})".format(
            _usd(DEFAULT_SPEND_CAP_USD)))
    parser.add_argument(
        "--model", default=None,
        help="a registry model ID, or a local server's model name with --base-url "
             "(default: offline, {}, the model the recordings were made on; live, "
             "the model your .env selects, as llm.default_model() picks it)".format(
                 DEFAULT_MODEL))
    parser.add_argument(
        "--base-url", default=None,
        help="a local OpenAI-compatible server, e.g. http://localhost:11434/v1")
    return parser


def _model_and_base_url(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    """The model to run, and the local server it is on, if any.

    ``--model`` wins. Offline, the default is the model the recordings were made
    on. Live, it is ``llm.default_model()`` on the environment ``main()`` loaded.
    """
    if args.model:
        return args.model, args.base_url
    if args.mode == OFFLINE:
        return DEFAULT_MODEL, args.base_url
    environ: Dict[str, str] = dict(os.environ)
    if args.base_url:
        environ.update(LLM_PROVIDER="local", OPENAI_BASE_URL=args.base_url)
    spec = resolve_model(load_registry(), environ)
    if spec.provider == LOCAL_PROVIDER:
        return spec.model_id, spec.base_url
    return spec.model_id, args.base_url


#: Packages the Checks import, by import name, and the pip name that installs each.
_PIP_NAMES = {"dotenv": "python-dotenv", "opentelemetry": tracing.INSTALL_HINT.split()[-1]}


def start_error(error: BaseException) -> str:
    """The one line a run that could not start prints: what went wrong, and
    for a package the course needs that is missing, the pip line that fixes it."""
    missing = getattr(error, "name", None) if isinstance(error, ImportError) else None
    root = (missing or "").split(".")[0]
    if root in _PIP_NAMES and not isinstance(error, tracing.TracingUnavailable):
        return "{} is not installed: pip install {} (it is in requirements.txt).".format(
            _PIP_NAMES[root], _PIP_NAMES[root])
    return str(error.args[0] if error.args else error)


def load_env() -> None:
    """The Reader's keys and model settings, from the root .env. A variable
    already set wins."""
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)


#: What stops a run before it starts: exit code 2, never 1, which blames a Check.
START_ERRORS = (ValueError, KeyError, ImportError, AttributeError, TypeError,
                FileNotFoundError, RuntimeError)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """0: passed through the last module. 1: a Check failed or the cap stopped
    the run. 2: the run could not start or could not reach the model, such as
    when a package it needs is not installed."""
    args = _parser().parse_args(argv)
    try:
        load_env()
        model, base_url = _model_and_base_url(args)
        suites = discover_suites()
        through = parse_modules(args.modules) if args.modules else suites[-1].module
        report = run_checks(
            through=through,
            agent=args.agent,
            mode=args.mode,
            spend_cap_usd=args.spend_cap,
            model=model,
            base_url=base_url,
            suites=suites,
        )
    except START_ERRORS as error:
        print("checks: {}".format(start_error(error)), file=sys.stderr)
        return 2
    if report.stopped_reason:
        return 2
    return 0 if report.passed else 1
