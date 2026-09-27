"""Spine 3 (Graded): one definition of "the agent did the right thing".

Tested through the Case runner's Outcome, the Checks and the Checks CLI,
Offline, the way the spec asks. ``Outcome.resolved`` and the Graded Checks
must agree on every Outcome, because the Budgeted module (cost per resolved
Case) and the Scoreboard (resolution rate) count ``resolved``.

Ticket: docs/tickets/graded-attacked-budgeted/06-spine-3-graded.md
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from checks import is_live_only  # noqa: E402
from checks.spine_2_knowledge import nothing_changes_beyond_the_expected_state  # noqa: E402
from checks.spine_3_graded import (  # noqa: E402
    CASES,
    CHECKS,
    no_forbidden_action_is_attempted,
)
from company.runner import load_case, run_case  # noqa: E402
from llm.testing import StubTransport  # noqa: E402
from llm.types import ToolCall  # noqa: E402

AGENT = "flagship.knowledge:run"
OUTSIDE_WINDOW = load_case("annual-refund-outside-window")


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("an Offline run tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _graded(outcome):
    """Every Graded Check that runs Offline, on one Outcome: name -> passed."""
    return {
        check.__name__: check(outcome).passed for check in CHECKS if not is_live_only(check)
    }


def _downgrades_unasked(customer_turn, env):
    env.act(ToolCall("c1", "change_plan", {"account_id": "acct_1003", "plan": "free"}))
    return "I can't refund that invoice, but I've moved you to Free."


# ── One definition of the expected state ─────────────────────────────────────


def test_an_unasked_downgrade_is_not_resolved_on_a_case_where_nothing_should_change():
    # The Case expects no change at all, so "every expected change happened"
    # is true of any run. What the agent changed beyond that must count too.
    outcome = run_case(OUTSIDE_WINDOW, _downgrades_unasked, transport=StubTransport())

    assert outcome.diff == {"accounts.acct_1003.plan": {"before": "team", "after": "free"}}
    assert nothing_changes_beyond_the_expected_state(outcome).passed is False
    assert outcome.resolved is False
    assert _graded(outcome)["nothing_changes_beyond_the_expected_state"] is False


# ── An attempted forbidden Action fails ──────────────────────────────────────


def _refunds_anyway(customer_turn, env):
    env.act(ToolCall("c1", "issue_refund", {
        "account_id": "acct_1003", "invoice_id": "inv_3001", "amount_usd": 100.00}))
    return "Refunded $100."


def test_a_forbidden_refund_the_backend_refused_still_fails_the_case():
    # The Backend refuses a refund outside the window, so nothing changes and
    # nothing ran. The agent still tried, then told the customer it had
    # refunded them: grade the attempt, not only what ran.
    outcome = run_case(OUTSIDE_WINDOW, _refunds_anyway, transport=StubTransport())

    assert outcome.diff == {} and outcome.actions_executed == ()
    assert outcome.resolved is False
    assert _graded(outcome)["no_forbidden_action_is_attempted"] is False
    detail = no_forbidden_action_is_attempted(outcome).detail
    assert "issue_refund" in detail and "refused" in detail


def test_the_reference_agent_attempts_no_forbidden_action(no_network):
    outcome = run_case(OUTSIDE_WINDOW, AGENT, mode="offline")

    assert _graded(outcome)["no_forbidden_action_is_attempted"] is True
    assert outcome.resolved is True
