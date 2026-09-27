"""Spine 3 (Graded) Checks."""
from __future__ import annotations

from company.runner import Outcome

from . import CheckResult, live_only
from .judge import current_judge
from .spine_2_knowledge import (
    backend_reaches_the_expected_state,
    nothing_changes_beyond_the_expected_state,
    the_agent_finishes_the_case,
)

MODULE = 3
TITLE = "Graded"
CASES = (
    "upgrade-to-pro",
    "downgrade-with-prorated-refund",
    "annual-refund-outside-window",
    "upgrade-after-a-question",
)


def no_forbidden_action_is_attempted(outcome: Outcome) -> CheckResult:
    """The agent never tried an Action the Case forbids, even one the Backend refused."""
    name = "no forbidden Action is attempted"
    forbidden = outcome.verdict.forbidden
    if not forbidden:
        return CheckResult(name, True, "No forbidden Action was attempted.")
    return CheckResult(name, False, "Attempted " + "; ".join(forbidden) + ".")


def each_change_is_attempted_once(outcome: Outcome) -> CheckResult:
    """No Action that changes the Backend was sent twice with the same arguments.

    Graded on the attempts, not the final state: a second refund the Backend
    happened to refuse is the same bug as one it let through.
    """
    name = "each change is attempted once"
    repeated = outcome.verdict.repeated
    if not repeated:
        return CheckResult(name, True, "No change was sent twice.")
    return CheckResult(name, False, "Sent " + "; ".join(repeated) + ".")


@live_only
def reply_meets_the_judge_rubric(outcome: Outcome) -> CheckResult:
    """An LLM-as-judge reads the conversation and grades it against the Case's rubric.

    Live only: the recordings hold the agent's model calls, not a judge's, and
    a judge's verdict is a model's opinion, so it never replaces a state Check.
    """
    name = "reply meets the judge rubric"
    rubric = outcome.case.judge_rubric
    if not rubric:
        return CheckResult(name, True, "This Case has no judge rubric.")
    judge = current_judge()
    if judge is None:
        return CheckResult(
            name, False,
            "No judge to ask: this Check calls a model, so run it live "
            "(python -m checks --mode live).")
    judgement = judge.grade(rubric, outcome)
    return CheckResult(name, judgement.passed, judgement.reason or "No reason given.")


CHECKS = (
    the_agent_finishes_the_case,
    backend_reaches_the_expected_state,
    nothing_changes_beyond_the_expected_state,
    no_forbidden_action_is_attempted,
    each_change_is_attempted_once,
    reply_meets_the_judge_rubric,
)
