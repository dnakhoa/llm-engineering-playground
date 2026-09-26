"""The Case runner: run one Case against an agent, get an Outcome.

This is seam 1. Everything a Check needs comes back on the Outcome: the final
Backend state and its diff from the seed, every Action the agent attempted and
every one that ran, the transcript, usage and cost, whether the step limit
stopped the run, and whether the Case was resolved.

    from company.runner import load_case, run_case

    outcome = run_case(load_case("upgrade-to-pro"), "flagship.loop:run")
    outcome.resolved       # True
    outcome.diff           # {"accounts.acct_1001.plan": {"before": "free", "after": "pro"}}

The agent is anything callable as ``agent(customer_turn, env)`` that returns
its final reply. Pass it, or pass an importable reference such as
``"my_agent.loop:run"``, so the Checks grade the Reader's code, not ours.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from llm import load_registry
from llm.call import complete
from llm.registry import Registry
from llm.replay import ReplayTransport
from llm.types import (
    ROLE_TOOL,
    CallOptions,
    Message,
    Response,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

from .backend import ActionRecord, Actions, Backend

COMPANY_DIR = Path(__file__).resolve().parent
CASES_DIR = COMPANY_DIR / "cases"
RECORDINGS_DIR = COMPANY_DIR / "recordings"

#: The model the reviewed recordings were made against. Offline mode replays
#: them, so a different model in Offline mode is a replay mismatch, on purpose.
DEFAULT_MODEL = "claude-sonnet-5"

#: Model calls one Case may make before the runner stops the agent.
DEFAULT_STEP_LIMIT = 8

OFFLINE = "offline"
LIVE = "live"

Agent = Callable[[str, "Environment"], Optional[str]]


# ── Cases ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    """One customer request, as a data file in ``company/cases/``."""

    id: str
    customer_account_id: str
    opening_message: str
    expected_state_change: Mapping[str, Any]
    forbidden_actions: Tuple[str, ...] = ()
    follow_ups: Tuple[str, ...] = ()
    recording: Optional[str] = None
    tags: Mapping[str, Any] = field(default_factory=dict)


def load_case(case_id_or_path: Union[str, Path]) -> Case:
    """Load a Case by id (``"upgrade-to-pro"``) or by path to its JSON file."""
    path = Path(case_id_or_path)
    if not path.suffix:
        path = CASES_DIR / "{}.json".format(case_id_or_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Case(
        id=data["id"],
        customer_account_id=data["customer"]["account_id"],
        opening_message=data["opening_message"],
        expected_state_change=dict(data.get("expected_state_change") or {}),
        forbidden_actions=tuple(data.get("forbidden_actions") or ()),
        follow_ups=tuple(data.get("follow_ups") or ()),
        recording=data.get("recording"),
        tags=dict(data.get("tags") or {}),
    )


# ── What the agent gets ───────────────────────────────────────────────────────


class StepLimitReached(RuntimeError):
    """The agent used every model call the Case allows."""

    def __init__(self, step_limit: int) -> None:
        super().__init__(
            "Stopped after {} model calls: the step limit for one Case.".format(step_limit)
        )
        self.step_limit = step_limit


class SpendCapReached(RuntimeError):
    """The run has spent its Spend Cap, so no further model call goes out."""

    def __init__(self, cap: "SpendCap") -> None:
        super().__init__(
            "Stopped: this run has spent ${:.6f} of its ${:.2f} Spend Cap.".format(
                cap.spent_usd, cap.limit_usd
            )
        )
        self.cap = cap


class SpendCap:
    """The hard limit on model spend for one run of Checks, across every Case.

    Each call's cost is its usage at registry prices, cached tokens at the cache
    price. A call is refused once the run has spent the limit. A call's cost is
    only known when it returns, so the run can end past the limit by at most
    the one call that crossed it.
    """

    def __init__(self, limit_usd: float) -> None:
        if limit_usd < 0:
            raise ValueError("A Spend Cap cannot be negative, not {}.".format(limit_usd))
        self.limit_usd = float(limit_usd)
        self.spent_usd = 0.0

    @property
    def reached(self) -> bool:
        return self.spent_usd >= self.limit_usd

    def check(self) -> None:
        """Raise ``SpendCapReached`` if no more calls may go out."""
        if self.reached:
            raise SpendCapReached(self)

    def charge(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd


class Environment:
    """What an agent works with: the Actions as tools, and the model.

    ``complete`` is the provider layer's ``llm.complete`` with the model and
    transport already chosen, so the same agent code runs Offline or live.
    Every call counts toward the step limit and toward usage and cost.
    """

    def __init__(
        self,
        *,
        actions: Actions,
        model: str,
        transport,
        registry: Registry,
        step_limit: int,
        options: Optional[CallOptions] = None,
        spend_cap: Optional[SpendCap] = None,
    ) -> None:
        self._actions = actions
        self._spend_cap = spend_cap
        self._transport = transport
        self._registry = registry
        self._options = options
        self.model = model
        self.step_limit = step_limit
        self.steps = 0
        self.responses: List[Response] = []
        self.transcript: List[Message] = []

    @property
    def tools(self) -> Tuple[ToolSpec, ...]:
        return self._actions.tools

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        system: Optional[str] = None,
        options: Optional[CallOptions] = None,
    ) -> Response:
        if self.steps >= self.step_limit:
            raise StepLimitReached(self.step_limit)
        if self._spend_cap is not None:
            self._spend_cap.check()
        self.steps += 1
        response = complete(
            model=self.model,
            messages=messages,
            tools=tools,
            system=system,
            options=options if options is not None else self._options,
            transport=self._transport,
            registry=self._registry,
        )
        self.responses.append(response)
        if self._spend_cap is not None:
            self._spend_cap.charge(response.cost_usd)
        self.transcript.append(Message.assistant(response.text or None, response.tool_calls))
        return response

    def act(self, call: ToolCall) -> ToolResult:
        """Run one Action. Refusals come back as an error result for the model."""
        result = self._actions.run(call)
        last = self.transcript[-1] if self.transcript else None
        if last is not None and last.role == ROLE_TOOL:
            self.transcript[-1] = Message.tool(last.tool_results + (result,))
        else:
            self.transcript.append(Message.tool([result]))
        return result


# ── What comes back ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Outcome:
    """Everything that happened on one Case. Checks assert on this."""

    case: Case
    model: str
    reply: Optional[str]
    final_state: Mapping[str, Any]
    diff: Mapping[str, Mapping[str, Any]]
    actions_attempted: Tuple[ActionRecord, ...]
    actions_executed: Tuple[ActionRecord, ...]
    transcript: Tuple[Message, ...]
    usage: Usage
    cost_usd: float
    steps: int
    step_limit: int
    step_limit_reached: bool
    resolved: bool
    #: The run's Spend Cap stopped this Case before the agent finished.
    spend_cap_reached: bool = False


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key in value:
            out.update(_flatten(value[key], "{}.{}".format(prefix, key) if prefix else str(key)))
        return out
    return {prefix: value}


def state_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every leaf that changed, by dotted path: ``{path: {"before", "after"}}``."""
    old, new = _flatten(before), _flatten(after)
    return {
        path: {"before": old.get(path), "after": new.get(path)}
        for path in sorted(set(old) | set(new))
        if old.get(path) != new.get(path)
    }


def _resolved(case: Case, diff, executed, stopped: bool) -> bool:
    if stopped:
        return False
    if any(record.name in case.forbidden_actions for record in executed):
        return False
    return all(
        path in diff and diff[path]["after"] == value
        for path, value in case.expected_state_change.items()
    )


def _import_file(path: Path):
    """Import a Reader's agent from a ``.py`` file anywhere on disk.

    Its folder goes on ``sys.path`` so the file can import its own neighbours.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("No agent file at {}.".format(path))
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    name = "reader_agent_{}".format(path.stem)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load an agent from {}.".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_agent(agent: Union[str, Agent]) -> Agent:
    """Accept a callable, or a reference to one.

    A reference is ``"package.module:function"`` for anything importable, or
    ``"path/to/my_agent.py:function"`` for a file outside this repo.
    """
    if callable(agent):
        return agent
    module_name, sep, attribute = agent.rpartition(":")
    if not sep or not module_name or not attribute:
        raise ValueError(
            "An agent reference looks like 'package.module:function' or "
            "'path/to/agent.py:function', not {!r}.".format(agent)
        )
    if module_name.endswith(".py"):
        target = _import_file(Path(module_name))
    else:
        target = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError("{} is not callable.".format(agent))
    return target  # type: ignore[return-value]


def _transport_for(case: Case, mode: str, model: str, registry: Registry):
    if mode == OFFLINE:
        if not case.recording:
            raise ValueError(
                "Case {} has no recording, so it cannot run Offline.".format(case.id)
            )
        return ReplayTransport(RECORDINGS_DIR / case.recording)
    if mode == LIVE:
        from llm.transport import HttpTransport

        return HttpTransport(provider=registry.get(model).provider)
    raise ValueError("mode is 'offline' or 'live', not {!r}.".format(mode))


def run_case(
    case: Case,
    agent: Union[str, Agent],
    *,
    model: str = DEFAULT_MODEL,
    mode: str = OFFLINE,
    transport=None,
    step_limit: int = DEFAULT_STEP_LIMIT,
    options: Optional[CallOptions] = None,
    registry: Optional[Registry] = None,
    spend_cap: Optional[SpendCap] = None,
) -> Outcome:
    """Run ``case`` against ``agent`` on a fresh Backend and report the Outcome.

    Offline mode (the default) replays the Case's reviewed recording and needs
    no key. Pass ``transport`` to supply your own, for example a stub in tests.
    A request the recording has not seen raises ``ReplayMismatchError``.

    Pass a ``SpendCap`` shared across a run's Cases to stop the run when it
    has spent the limit; a Case it stops is reported, unresolved, with
    ``spend_cap_reached`` set.
    """
    registry = registry or load_registry()
    run_agent = load_agent(agent)
    backend = Backend.seeded()
    seed = backend.export_state()
    actions = backend.actions_for(case.customer_account_id)
    env = Environment(
        actions=actions,
        model=model,
        transport=transport if transport is not None else _transport_for(case, mode, model, registry),
        registry=registry,
        step_limit=step_limit,
        options=options,
        spend_cap=spend_cap,
    )
    env.transcript.append(Message.user(case.opening_message))

    reply: Optional[str] = None
    step_limit_reached = False
    spend_cap_reached = False
    try:
        reply = run_agent(case.opening_message, env)
    except StepLimitReached:
        step_limit_reached = True
    except SpendCapReached:
        spend_cap_reached = True

    final_state = backend.export_state()
    diff = state_diff(seed, final_state)
    attempted = tuple(actions.log)
    executed = tuple(record for record in attempted if record.executed)
    return Outcome(
        case=case,
        model=model,
        reply=reply,
        final_state=final_state,
        diff=diff,
        actions_attempted=attempted,
        actions_executed=executed,
        transcript=tuple(env.transcript),
        usage=Usage(
            input_tokens=sum(r.usage.input_tokens for r in env.responses),
            output_tokens=sum(r.usage.output_tokens for r in env.responses),
            cached_input_tokens=sum(r.usage.cached_input_tokens for r in env.responses),
            cache_write_input_tokens=sum(
                r.usage.cache_write_input_tokens for r in env.responses
            ),
        ),
        cost_usd=sum(r.cost_usd for r in env.responses),
        steps=env.steps,
        step_limit=step_limit,
        step_limit_reached=step_limit_reached,
        resolved=_resolved(
            case, diff, executed, step_limit_reached or spend_cap_reached
        ),
        spend_cap_reached=spend_cap_reached,
    )
