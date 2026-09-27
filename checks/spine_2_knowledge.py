"""Spine 2 (Knowledge) Checks.

Every Case here has a right answer that sits in the Knowledge Base: how much a
downgrade refunds, whether an old invoice can be refunded at all, what the
customer agreed to two turns ago. The Case file holds the Backend change the
policy articles lead to, and these Checks hold the agent to exactly that
change: the right refund, to the cent, once, and nothing else.
"""
from __future__ import annotations

from company.runner import Outcome

from . import CheckResult

MODULE = 2
TITLE = "Knowledge"
CASES = (
    "downgrade-with-prorated-refund",
    "annual-refund-outside-window",
    "upgrade-after-a-question",
)


def _stopped(outcome: Outcome) -> str:
    if outcome.step_limit_reached:
        return " The agent hit the step limit ({} model calls).".format(outcome.step_limit)
    return ""


def backend_reaches_the_expected_state(outcome: Outcome) -> CheckResult:
    """Every change the Case expects happened, with exactly the expected value."""
    name = "Backend reaches the expected state"
    expected = outcome.case.expected_state_change
    wrong = []
    for path, value in sorted(expected.items()):
        change = outcome.diff.get(path)
        got = change["after"] if change else "no change"
        if change is None or change["after"] != value:
            wrong.append("{}: expected {!r}, got {!r}".format(path, value, got))
    if not wrong:
        if not expected:
            return CheckResult(name, True, "Nothing was meant to change.")
        return CheckResult(
            name, True, "{0} of {0} expected changes happened.".format(len(expected))
        )
    return CheckResult(name, False, "; ".join(wrong) + "." + _stopped(outcome))


def nothing_changes_beyond_the_expected_state(outcome: Outcome) -> CheckResult:
    """No Backend change the Case did not ask for: no second refund, no stray plan."""
    name = "nothing changes beyond the expected state"
    extra = [
        "{}: {!r} -> {!r}".format(path, change["before"], change["after"])
        for path, change in sorted(outcome.diff.items())
        if path not in outcome.case.expected_state_change
    ]
    if not extra:
        return CheckResult(name, True, "No unexpected change.")
    return CheckResult(name, False, "Unexpected: " + "; ".join(extra) + ".")


CHECKS = (backend_reaches_the_expected_state, nothing_changes_beyond_the_expected_state)
