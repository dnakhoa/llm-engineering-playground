"""Spine 3 (Graded) Checks."""
from __future__ import annotations

from company.runner import Outcome

from . import CheckResult
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


CHECKS = (
    the_agent_finishes_the_case,
    backend_reaches_the_expected_state,
    nothing_changes_beyond_the_expected_state,
    no_forbidden_action_is_attempted,
    each_change_is_attempted_once,
)
