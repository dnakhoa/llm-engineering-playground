"""Spine 3 (Graded) Checks."""
from __future__ import annotations

from company.runner import Outcome

from . import CheckResult
from .spine_2_knowledge import (
    backend_reaches_the_expected_state,
    nothing_changes_beyond_the_expected_state,
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


CHECKS = (
    backend_reaches_the_expected_state,
    nothing_changes_beyond_the_expected_state,
    no_forbidden_action_is_attempted,
)
