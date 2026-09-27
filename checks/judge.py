"""The LLM-as-judge: a model that grades a reply against a Case's rubric.

State assertions decide whether the agent did the right thing. A judge is for
what state cannot show: whether the reply told the customer the truth, in
words they can act on. It is a model call, so it is live only, it costs money,
and it can be wrong. The Checks CLI reports it as skipped Offline, and counts
its calls against the run's Spend Cap live.

The CLI sets the judge for a live run; a Check asks for it with
``current_judge()``. To judge one Outcome yourself, on your own key::

    from checks.judge import Judge, judging_with
    with judging_with(Judge(model="claude-sonnet-5")):
        reply_meets_the_judge_rubric(outcome)
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import re
from dataclasses import dataclass
from typing import Iterator, Optional

from company.runner import Outcome, SpendCap
from llm import load_registry
from llm.call import complete
from llm.registry import Registry
from llm.types import CallOptions, Message

SYSTEM = (
    "You grade one support agent's handling of one customer Case against a rubric. "
    "Read the conversation and the Actions the agent took, then decide whether the "
    "agent's replies meet every part of the rubric. Answer with one JSON object and "
    'nothing else: {"verdict": "pass" or "fail", "reason": "one sentence"}.'
)

#: Enough room for one short verdict. A judge that writes an essay is not grading.
OPTIONS = CallOptions(effort="low", max_output_tokens=300)


@dataclass(frozen=True)
class Judgement:
    passed: bool
    reason: str


def _conversation(outcome: Outcome) -> str:
    """Each customer turn and the agent's reply to it, then every Action attempted."""
    lines = []
    for turn, reply in zip(outcome.case.customer_turns, outcome.replies):
        lines.append("Customer: {}".format(turn))
        lines.append("Agent: {}".format(reply or "(no reply)"))
    for record in outcome.actions_attempted:
        lines.append("Action {}({}): {} - {}".format(
            record.name, json.dumps(dict(record.arguments), sort_keys=True),
            "ran" if record.executed else "refused", record.result))
    return "\n".join(lines)


def _parse(text: str) -> Optional[Judgement]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    verdict = str(data.get("verdict", "")).strip().lower() if isinstance(data, dict) else ""
    if verdict not in ("pass", "fail"):
        return None
    return Judgement(verdict == "pass", str(data.get("reason", "")).strip())


class Judge:
    """Grades a Case's replies against its rubric, with one model call.

    Pass the run's ``SpendCap`` so the judge's calls count toward it: a
    course about budgets does not get to hide the cost of its own grading.
    Without a ``transport``, the first grade opens an HTTPS one to the
    model's provider, on the Reader's key.
    """

    def __init__(
        self,
        *,
        model: str,
        transport=None,
        registry: Optional[Registry] = None,
        spend_cap: Optional[SpendCap] = None,
    ) -> None:
        self.model = model
        self._transport = transport
        self._registry = registry or load_registry()
        self._spend_cap = spend_cap

    def grade(self, rubric: str, outcome: Outcome) -> Judgement:
        """The judge's verdict. Raises ``SpendCapReached`` if the cap is spent."""
        if self._spend_cap is not None:
            self._spend_cap.check()
        if self._transport is None:
            import llm.transport

            self._transport = llm.transport.HttpTransport(
                provider=self._registry.get(self.model).provider)
        prompt = "Rubric:\n{}\n\nThe Case:\n{}".format(rubric, _conversation(outcome))
        response = complete(
            model=self.model,
            messages=[Message.user(prompt)],
            system=SYSTEM,
            options=OPTIONS,
            transport=self._transport,
            registry=self._registry,
        )
        if self._spend_cap is not None:
            self._spend_cap.charge(response.cost_usd)
        judgement = _parse(response.text or "")
        if judgement is None:
            return Judgement(False, "The judge's answer was not a verdict: {!r}".format(
                (response.text or "")[:200]))
        return judgement


_JUDGE: contextvars.ContextVar[Optional[Judge]] = contextvars.ContextVar("judge", default=None)


def current_judge() -> Optional[Judge]:
    """The judge for this run, or ``None`` outside a live run."""
    return _JUDGE.get()


@contextlib.contextmanager
def judging_with(judge: Optional[Judge]) -> Iterator[None]:
    """Make ``judge`` the one live-only Checks use, for the length of the block."""
    token = _JUDGE.set(judge)
    try:
        yield
    finally:
        _JUDGE.reset(token)
