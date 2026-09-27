"""The Case runner: run one Case against an agent, get an Outcome.

This is seam 1. Everything a Check needs comes back on the Outcome: the final
Backend state and its diff from the seed, every Action the agent attempted and
every one that ran, the transcript, the reply to each customer turn, usage and
cost, whether the step limit stopped the run, and the verdict: whether the Case
was resolved, and if not, why.

    from company.runner import load_case, run_case

    outcome = run_case(load_case("upgrade-to-pro"), "flagship.loop:run")
    outcome.resolved       # True
    outcome.diff           # {"accounts.acct_1001.plan": {"before": "free", "after": "pro"}}

The agent is anything callable as ``agent(customer_turn, env)`` that returns
its reply to that turn. A Case with follow-up turns calls it once per turn,
with the same ``env``, so an agent that needs an earlier turn must remember it
(``env.memory``). Pass the agent, or an importable reference such as
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

from .backend import ACTION_NAMES, READ_ONLY_ACTIONS, ActionRecord, Actions, Backend
from .knowledge import SEARCH_TOOL, LexicalRetriever, Retriever, load_articles, run_search

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
    """One customer request, as a data file in ``company/cases/``.

    ``actions`` is the Actions the Case lets the agent use, in the order it is
    offered them (ADR 0005). Anything else the agent calls is refused, and
    logged as attempted but not executed. ``knowledge_base`` says whether the
    agent may search the Knowledge Base too.
    """

    id: str
    customer_account_id: str
    opening_message: str
    expected_state_change: Mapping[str, Any]
    actions: Tuple[str, ...]
    forbidden_actions: Tuple[str, ...] = ()
    follow_ups: Tuple[str, ...] = ()
    knowledge_base: bool = False
    recording: Optional[str] = None
    tags: Mapping[str, Any] = field(default_factory=dict)
    #: What a good reply does, in words, for an LLM-as-judge Check to grade.
    judge_rubric: Optional[str] = None

    @property
    def customer_turns(self) -> Tuple[str, ...]:
        """The opening message, then each follow-up, in the order they arrive."""
        return (self.opening_message,) + self.follow_ups


def load_case(case_id_or_path: Union[str, Path]) -> Case:
    """Load a Case by id (``"upgrade-to-pro"``) or by path to its JSON file."""
    path = Path(case_id_or_path)
    if not path.suffix:
        path = CASES_DIR / "{}.json".format(case_id_or_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("actions"), list):
        raise ValueError(
            "Case {} does not declare its Actions. Add \"actions\": a list of the "
            "Actions it lets the agent use, from: {}.".format(
                data.get("id", path.stem), ", ".join(ACTION_NAMES)
            )
        )
    repeated = sorted({name for name in data["actions"] if data["actions"].count(name) > 1})
    if repeated:
        raise ValueError(
            "Case {} declares {} more than once.".format(
                data.get("id", path.stem), ", ".join(repeated)
            )
        )
    unknown = [name for name in data["actions"] if name not in ACTION_NAMES]
    if unknown:
        raise ValueError(
            "Case {} declares {}, which the Backend does not have. It has: {}.".format(
                data.get("id", path.stem), ", ".join(unknown), ", ".join(ACTION_NAMES)
            )
        )
    knowledge_base = data.get("knowledge_base", False)
    if not isinstance(knowledge_base, bool):
        raise ValueError(
            "Case {}: \"knowledge_base\" is true or false, not {!r}.".format(
                data.get("id", path.stem), knowledge_base
            )
        )
    judge_rubric = data.get("judge_rubric")
    if judge_rubric is not None and not (isinstance(judge_rubric, str) and judge_rubric.strip()):
        raise ValueError(
            "Case {}: \"judge_rubric\" is a sentence saying what a good reply does, "
            "not {!r}.".format(data.get("id", path.stem), judge_rubric)
        )
    return Case(
        id=data["id"],
        customer_account_id=data["customer"]["account_id"],
        opening_message=data["opening_message"],
        expected_state_change=dict(data.get("expected_state_change") or {}),
        actions=tuple(data["actions"]),
        forbidden_actions=tuple(data.get("forbidden_actions") or ()),
        follow_ups=tuple(data.get("follow_ups") or ()),
        knowledge_base=knowledge_base,
        recording=data.get("recording"),
        tags=dict(data.get("tags") or {}),
        judge_rubric=judge_rubric,
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
    """What an agent works with: the Case's tools, the model and its memory.

    ``tools`` is the Actions the Case declares, plus ``search_knowledge_base``
    when the Case lets the agent use the Knowledge Base; ``act`` runs any of
    them. ``knowledge`` is that Case's retriever, or ``None``. ``complete`` is
    the provider layer's ``llm.complete`` with the model and transport already
    chosen, so the same agent code runs Offline or live. Every call counts
    toward the step limit and toward usage and cost. ``memory`` is the agent's
    own, for the length of one Case: empty at the first turn, kept across the
    follow-ups, gone before the next Case.
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
        knowledge: Optional[Retriever] = None,
    ) -> None:
        self._actions = actions
        self.knowledge = knowledge
        self.memory: Dict[str, Any] = {}
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
        if self.knowledge is None:
            return self._actions.tools
        return self._actions.tools + (SEARCH_TOOL,)

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
        """Run one tool call. Refusals come back as an error result for the model.

        A Knowledge Base search reads and changes nothing, so it is not an
        Action: it goes into the transcript but not into the Action log.
        """
        if self.knowledge is not None and call.name == SEARCH_TOOL.name:
            result = run_search(self.knowledge, call)
        else:
            result = self._actions.run(call)
        last = self.transcript[-1] if self.transcript else None
        if last is not None and last.role == ROLE_TOOL:
            self.transcript[-1] = Message.tool(last.tool_results + (result,))
        else:
            self.transcript.append(Message.tool([result]))
        return result


# ── What comes back ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Verdict:
    """How one run of a Case measures up to it: the one definition of resolved.

    ``Outcome.resolved`` is ``verdict.resolved``, and each Graded Check reports
    one part of the verdict, so a run the Checks fail is never counted as
    resolved by the Budgeted Checks or the Scoreboard. Each part is a tuple of
    findings, written for the Reader; an empty part is a right one.
    """

    #: Why the agent did not finish the Case: the step limit, the Spend Cap,
    #: or a customer turn it answered with an empty reply.
    unfinished: Tuple[str, ...] = ()
    #: Expected changes that did not happen, or happened with another value.
    missing: Tuple[str, ...] = ()
    #: Backend changes the Case did not ask for.
    unexpected: Tuple[str, ...] = ()
    #: Attempts at an Action the Case forbids, whether or not they ran.
    forbidden: Tuple[str, ...] = ()
    #: Changes the agent sent more than once with the same arguments, whether
    #: or not the Backend let the repeat through.
    repeated: Tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return not (
            self.unfinished or self.missing or self.unexpected or self.forbidden
            or self.repeated
        )


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
    #: Whether the run did what the Case expects, and if not, what went wrong.
    verdict: Verdict
    #: The run's Spend Cap stopped this Case before the agent finished.
    spend_cap_reached: bool = False
    #: The agent's reply to each customer turn it answered, in order.
    replies: Tuple[Optional[str], ...] = ()

    @property
    def resolved(self) -> bool:
        """The agent finished, reached exactly the expected state, tried nothing
        the Case forbids and sent no change twice: ``verdict.resolved``."""
        return self.verdict.resolved


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


def _call(record: ActionRecord) -> str:
    return "{}({})".format(
        record.name,
        ", ".join("{}={!r}".format(k, v) for k, v in sorted(record.arguments.items())),
    )


def _verdict_for(
    case: Case,
    diff: Mapping[str, Mapping[str, Any]],
    attempted: Sequence[ActionRecord],
    replies: Sequence[Optional[str]],
    *,
    step_limit: int,
    step_limit_reached: bool,
    spend_cap_reached: bool,
) -> Verdict:
    """The Verdict on one run of ``case``.

    The agent must answer every customer turn with a reply that says something:
    a run stopped by the step limit, or a turn answered with nothing, is
    unfinished whatever the Backend looks like. Every expected path must have
    changed to exactly its expected value, and nothing else may have changed:
    a Case that expects no change is met only by a run that changed nothing.
    A forbidden Action counts when it was attempted, even if the Backend
    refused it, and so does a change sent twice: the Backend refusing the
    second refund this time is luck, not a property of the agent.
    """
    unfinished = []
    if step_limit_reached:
        unfinished.append("the agent hit the step limit ({} model calls)".format(step_limit))
    if spend_cap_reached:
        unfinished.append("the Spend Cap stopped the run mid-Case")
    for turn, reply in enumerate(replies, start=1):
        if not (reply or "").strip():
            unfinished.append("the agent's reply to customer turn {} was empty".format(turn))
    missing = []
    for path, value in sorted(case.expected_state_change.items()):
        change = diff.get(path)
        if change is None or change["after"] != value:
            missing.append("{}: expected {!r}, got {!r}".format(
                path, value, change["after"] if change else "no change"))
    unexpected = [
        "{}: {!r} -> {!r}".format(path, change["before"], change["after"])
        for path, change in sorted(diff.items())
        if path not in case.expected_state_change
    ]
    forbidden = [
        "{}, {}".format(
            _call(record), "which ran" if record.executed else "which the Backend refused")
        for record in attempted
        if record.name in case.forbidden_actions
    ]
    sent: Dict[str, int] = {}
    for record in attempted:
        if record.name not in READ_ONLY_ACTIONS:
            sent[_call(record)] = sent.get(_call(record), 0) + 1
    repeated = [
        "{} {} times".format(call, times) for call, times in sent.items() if times > 1
    ]
    return Verdict(
        unfinished=tuple(unfinished),
        missing=tuple(missing),
        unexpected=tuple(unexpected),
        forbidden=tuple(forbidden),
        repeated=tuple(repeated),
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


def _knowledge_for(case: Case, retriever: Optional[Retriever]) -> Optional[Retriever]:
    if not case.knowledge_base:
        return None
    return retriever if retriever is not None else LexicalRetriever(load_articles())


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
    retriever: Optional[Retriever] = None,
) -> Outcome:
    """Run ``case`` against ``agent`` on a fresh Backend and report the Outcome.

    Each customer turn goes to the agent in order, with the same environment.
    A Case that declares the Knowledge Base searches it with ``retriever``,
    by default the lexical one; pass an ``EmbeddingsRetriever`` to upgrade.

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
    actions = backend.actions_for(case.customer_account_id, allowed=case.actions)
    env = Environment(
        actions=actions,
        model=model,
        transport=transport if transport is not None else _transport_for(case, mode, model, registry),
        registry=registry,
        step_limit=step_limit,
        options=options,
        spend_cap=spend_cap,
        knowledge=_knowledge_for(case, retriever),
    )

    replies: List[Optional[str]] = []
    step_limit_reached = False
    spend_cap_reached = False
    try:
        for turn in case.customer_turns:
            env.transcript.append(Message.user(turn))
            replies.append(run_agent(turn, env))
    except StepLimitReached:
        step_limit_reached = True
    except SpendCapReached:
        spend_cap_reached = True
    # The final reply is the answer to the last turn; a Case stopped before it has none.
    reply = replies[-1] if len(replies) == len(case.customer_turns) else None

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
        verdict=_verdict_for(
            case, diff, attempted, replies, step_limit=step_limit,
            step_limit_reached=step_limit_reached, spend_cap_reached=spend_cap_reached,
        ),
        spend_cap_reached=spend_cap_reached,
        replies=tuple(replies),
    )
