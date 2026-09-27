"""Spine 2 (Knowledge) Checks.

Every Case here has a right answer that sits in the Knowledge Base: how much a
downgrade refunds, whether an old invoice can be refunded at all, what the
customer agreed to two turns ago. The Case file holds the Backend change the
policy articles lead to, and these Checks hold the agent to exactly that
change: the right refund, to the cent, once, and nothing else.

Each Check reports one part of the Outcome's ``verdict``, the same verdict
``Outcome.resolved`` is read from, so the two cannot disagree. Each holds on
every Case (``on_every_case``), so the Checks CLI also grades every later
suite's Cases on them, such as the Graded suite's ``upgrade-to-pro``, which is
not one of the Cases here.
"""
from __future__ import annotations

from company.runner import Outcome

from . import CheckResult, on_every_case

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


@on_every_case
def the_agent_finishes_the_case(outcome: Outcome) -> CheckResult:
    """The agent answered every customer turn, within the step limit, with a reply.

    Without it, an agent that stops early passes any Case where nothing was
    meant to change, and a customer who got silence counts as helped.
    """
    name = "the agent finishes the Case"
    unfinished = outcome.verdict.unfinished
    if not unfinished:
        turns = len(outcome.case.customer_turns)
        return CheckResult(name, True, "It answered the customer's {}.".format(
            "turn" if turns == 1 else "{} turns".format(turns)))
    return CheckResult(name, False, "Unfinished: " + "; ".join(unfinished) + ".")


@on_every_case
def backend_reaches_the_expected_state(outcome: Outcome) -> CheckResult:
    """Every change the Case expects happened, with exactly the expected value."""
    name = "Backend reaches the expected state"
    expected = outcome.case.expected_state_change
    missing = outcome.verdict.missing
    if not missing:
        if not expected:
            return CheckResult(name, True, "Nothing was meant to change.")
        return CheckResult(
            name, True, "{0} of {0} expected changes happened.".format(len(expected))
        )
    return CheckResult(name, False, "; ".join(missing) + "." + _stopped(outcome))


@on_every_case
def nothing_changes_beyond_the_expected_state(outcome: Outcome) -> CheckResult:
    """No Backend change the Case did not ask for: no second refund, no stray plan."""
    name = "nothing changes beyond the expected state"
    unexpected = outcome.verdict.unexpected
    if not unexpected:
        return CheckResult(name, True, "No unexpected change.")
    return CheckResult(name, False, "Unexpected: " + "; ".join(unexpected) + ".")


CHECKS = (
    the_agent_finishes_the_case,
    backend_reaches_the_expected_state,
    nothing_changes_beyond_the_expected_state,
)
