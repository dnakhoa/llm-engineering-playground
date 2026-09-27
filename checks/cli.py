"""The Checks CLI: grade an agent on every Spine module up to N.

    python -m checks --modules 3 --agent path/to/my_agent.py:run
    python -m checks --modules 1-6 --mode live --spend-cap 0.50
    python -m checks --mode live --model qwen3:8b --base-url http://localhost:11434/v1

Running module N runs the suites of modules 1 to N, because hardening an agent
must not silently break what it already did. Each Case runs once, however many
suites use it. The last line is the one to share: "Passed through module N of M."

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
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from company.runner import (
    DEFAULT_MODEL,
    LIVE,
    OFFLINE,
    Agent,
    Outcome,
    SpendCap,
    load_agent,
    load_case,
    run_case,
)
from llm import load_registry, resolve_model
from llm.registry import LOCAL_PROVIDER, ModelSpec, local_model
from llm.replay import ReplayMismatchError
from llm.transport import TransportError

from . import is_live_only
from .suites import Check, Suite, discover_suites

#: The reference Flagship Agent at the latest Spine module's end state.
DEFAULT_AGENT = "flagship.knowledge:run"
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
    spent_usd: float = 0.0
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
            write("Offline: recorded usage is priced against the cap, and nothing is billed.")

    outcomes: Dict[str, Outcome] = {}
    for suite in selected:
        write("")
        write("Module {} · {}".format(suite.module, suite.title))
        for case_ref in suite.cases:
            case = load_case(case_ref)

            def emit(check: Check, status: str, detail: str, title: Optional[str] = None) -> None:
                report.results.append(
                    CheckRun(suite.module, case.id, check.__name__, status, detail))
                lines = detail.splitlines() or [""]
                write("  {:<5} {}: {}. {}".format(
                    _LABELS[status], case.id, title or _label(check), lines[0]).rstrip())
                for more in lines[1:]:
                    write("          " + more)

            if report.stopped_by_spend_cap or report.stopped_reason:
                for check in suite.checks:
                    emit(check, NOT_RUN, "Not run: the run stopped before this Case.")
                continue
            if mode == OFFLINE and not case.recording:
                for check in suite.checks:
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
                    for check in suite.checks:
                        emit(check, ERROR, str(error))
                    continue
                except ReplayMismatchError as error:
                    for check in suite.checks:
                        emit(check, ERROR, str(error))
                    continue
                except Exception as error:  # the Reader's agent raised: grade it as such
                    for check in suite.checks:
                        emit(check, ERROR, "The agent raised {}: {}".format(
                            type(error).__name__, error))
                    continue
                outcomes[case.id] = outcome

            if outcome.spend_cap_reached:
                report.stopped_by_spend_cap = True
                for check in suite.checks:
                    emit(check, STOPPED, "Stopped mid-Case by the Spend Cap.")
                continue
            for check in suite.checks:
                if mode == OFFLINE and is_live_only(check):
                    emit(check, SKIP, "Live only: it needs a model call no recording holds.")
                    continue
                result = check(outcome)
                emit(check, PASS if result.passed else FAIL, result.detail, result.name)

    report.spent_usd = cap.spent_usd if cap is not None else 0.0
    report.passed_through = _passed_through(selected, report.results)

    write("")
    if report.stopped_by_spend_cap:
        unfinished = sum(1 for r in report.results if r.status in (STOPPED, NOT_RUN))
        write("Stopped by the Spend Cap: spent {} of {}. {} Checks did not finish.".format(
            _usd(report.spent_usd), _usd(cap.limit_usd if cap else 0.0), unfinished))
    elif report.stopped_reason:
        write("Stopped: {}".format(report.stopped_reason))
    elif cap is not None:
        write("Spent {} of the {} Spend Cap.".format(_usd(report.spent_usd), _usd(cap.limit_usd)))
    write("Passed through module {} of {}.".format(report.passed_through, through))
    return report


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """0: passed through the last module. 1: a Check failed or the cap stopped
    the run. 2: the run could not start or could not reach the model."""
    args = _parser().parse_args(argv)
    from dotenv import load_dotenv

    # The Reader's keys and model settings. A variable already set wins.
    load_dotenv(ENV_FILE)
    try:
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
    except (ValueError, KeyError, ImportError, AttributeError, TypeError,
            FileNotFoundError, RuntimeError) as error:
        print("checks: {}".format(error.args[0] if error.args else error), file=sys.stderr)
        return 2
    if report.stopped_reason:
        return 2
    return 0 if report.passed else 1
