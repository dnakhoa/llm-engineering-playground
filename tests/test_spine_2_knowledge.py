"""Spine 2 (Knowledge): Cases whose right answer is in the Knowledge Base.

Tested through the Case runner's Outcome and the Checks, Offline, the way the
spec asks: never on retriever scores or prompt text.

The seed's "today" is 2026-09-20. The worked numbers:

* prorated refund: Northwind Studio paid $12.00 for a 30-day Pro month on
  2026-09-13. 7 days used, 23 unused: 12.00 x 23 / 30 = $9.20.
* outside the window: Harbor Robotics Club paid for a Team year on 2026-08-06,
  45 days ago. The window is 30 days, so nothing can be refunded.

Ticket: docs/tickets/graded-attacked-budgeted/05-spine-2-knowledge.md
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from checks.cli import run_checks  # noqa: E402
from checks.spine_2_knowledge import (  # noqa: E402
    CASES,
    CHECKS,
    backend_reaches_the_expected_state,
    nothing_changes_beyond_the_expected_state,
)
from company.knowledge import EmbeddingsRetriever, load_articles  # noqa: E402
from company.runner import load_case, run_case  # noqa: E402
from llm.replay import ReplayMismatchError  # noqa: E402
from llm.testing import StubTransport  # noqa: E402
from llm.types import ToolCall  # noqa: E402

AGENT = "flagship.knowledge:run"
PRORATED = load_case("downgrade-with-prorated-refund")
OUTSIDE_WINDOW = load_case("annual-refund-outside-window")
MULTI_TURN = load_case("upgrade-after-a-question")


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("an Offline run tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _passes_every_check(outcome):
    results = [check(outcome) for check in CHECKS]
    return all(result.passed for result in results), [r.detail for r in results]


# ── The prorated refund ───────────────────────────────────────────────────────


def test_the_prorated_refund_case_passes_with_the_policy_amount(no_network):
    outcome = run_case(PRORATED, AGENT, mode="offline")

    passed, details = _passes_every_check(outcome)
    assert passed, details
    assert outcome.resolved is True
    assert outcome.final_state["refunds"] == {
        "rf_0001": {"account_id": "acct_1002", "invoice_id": "inv_2002", "amount_cents": 920}
    }
    assert outcome.final_state["accounts"]["acct_1002"]["plan"] == "free"


def test_the_prorated_refund_case_is_answered_from_the_knowledge_base(no_network):
    outcome = run_case(PRORATED, AGENT, mode="offline")

    searched = [
        result.content
        for message in outcome.transcript
        for result in message.tool_results
        if result.name == "search_knowledge_base"
    ]
    assert searched and "(article refund-policy)" in searched[0]


def _refunds(amount_usd):
    """An agent that downgrades and refunds ``amount_usd``, whatever the policy says."""

    def agent(customer_turn, env):
        env.act(ToolCall("c1", "change_plan", {"account_id": "acct_1002", "plan": "free"}))
        env.act(ToolCall("c2", "issue_refund", {
            "account_id": "acct_1002", "invoice_id": "inv_2002", "amount_usd": amount_usd}))
        return "Done."

    return agent


def test_a_refund_the_policy_allows_but_the_article_does_not_give_fails_the_check():
    # $5.00 is inside what the Action allows, so it runs. It is still wrong:
    # the policy article gives $9.20, and the Check holds the agent to that.
    outcome = run_case(PRORATED, _refunds(5.00), transport=StubTransport())

    assert [a.executed for a in outcome.actions_attempted] == [True, True]
    result = backend_reaches_the_expected_state(outcome)
    assert result.passed is False
    assert "920" in result.detail and "500" in result.detail
    assert outcome.resolved is False


def test_a_full_refund_is_refused_by_the_action_and_fails_the_check():
    outcome = run_case(PRORATED, _refunds(12.00), transport=StubTransport())

    assert [(a.name, a.executed) for a in outcome.actions_attempted] == [
        ("change_plan", True), ("issue_refund", False)]
    assert outcome.final_state["refunds"] == {}
    assert backend_reaches_the_expected_state(outcome).passed is False


def test_two_refunds_that_add_up_to_the_policy_amount_fail_the_check():
    # The invoice ends with $9.20 refunded, but in two refunds, not one.
    def refunds_twice(customer_turn, env):
        _refunds(9.00)(customer_turn, env)
        env.act(ToolCall("c3", "issue_refund", {
            "account_id": "acct_1002", "invoice_id": "inv_2002", "amount_usd": 0.20}))
        return "Done."

    outcome = run_case(PRORATED, refunds_twice, transport=StubTransport())

    assert outcome.final_state["accounts"]["acct_1002"]["invoices"]["inv_2002"][
        "refunded_cents"] == 920
    result = nothing_changes_beyond_the_expected_state(outcome)
    assert result.passed is False
    assert "rf_0002" in result.detail
    assert backend_reaches_the_expected_state(outcome).passed is False


# ── The refund outside the window ─────────────────────────────────────────────


def test_the_refund_outside_the_window_case_passes_with_no_refund(no_network):
    outcome = run_case(OUTSIDE_WINDOW, AGENT, mode="offline")

    passed, details = _passes_every_check(outcome)
    assert passed, details
    assert outcome.resolved is True
    assert outcome.diff == {}
    assert "30 days" in outcome.reply


def test_a_refund_outside_the_window_is_refused_by_the_action_whatever_the_agent_thinks():
    def refunds_anyway(customer_turn, env):
        env.act(ToolCall("c1", "issue_refund", {
            "account_id": "acct_1003", "invoice_id": "inv_3001", "amount_usd": 100.00}))
        return "Refunded $100."

    outcome = run_case(OUTSIDE_WINDOW, refunds_anyway, transport=StubTransport())

    assert [(a.name, a.executed) for a in outcome.actions_attempted] == [("issue_refund", False)]
    assert outcome.diff == {}


# ── Memory across a Case's turns ──────────────────────────────────────────────


def test_a_multi_turn_case_that_needs_an_earlier_turns_detail_passes(no_network):
    outcome = run_case(MULTI_TURN, AGENT, mode="offline")

    passed, details = _passes_every_check(outcome)
    assert passed, details
    assert outcome.resolved is True
    # The second turn never names the account; only memory of the first can.
    assert "acct_1001" not in MULTI_TURN.follow_ups[0]
    assert [a.arguments["account_id"] for a in outcome.actions_executed] == [
        "acct_1001", "acct_1001"]
    assert len(outcome.replies) == 2
    assert outcome.reply == outcome.replies[-1]


def test_the_spine_1_agent_forgets_the_first_turn(no_network):
    # Spine 1's loop starts each turn from scratch, so its second request holds
    # only the second turn: a request no recording of this Case ever saw.
    with pytest.raises(ReplayMismatchError, match="body.messages"):
        run_case(MULTI_TURN, "flagship.loop:run", mode="offline")


def test_the_transcript_holds_every_customer_turn_in_order(no_network):
    outcome = run_case(MULTI_TURN, AGENT, mode="offline")

    customer_turns = [m.text for m in outcome.transcript if m.role == "user"]
    assert customer_turns == [MULTI_TURN.opening_message, *MULTI_TURN.follow_ups]


def test_memory_does_not_leak_from_one_case_into_the_next(no_network):
    run_case(MULTI_TURN, AGENT, mode="offline")

    # A fresh Case starts with empty memory, so its replay still matches.
    assert run_case(PRORATED, AGENT, mode="offline").resolved is True


# ── What the Case offers ──────────────────────────────────────────────────────


def test_a_case_that_declares_the_knowledge_base_offers_the_search_tool():
    stub = StubTransport()

    run_case(PRORATED, AGENT, transport=stub)

    offered = [tool["name"] for tool in stub.last_request.body["tools"]]
    assert offered == ["look_up_account", "change_plan", "issue_refund", "search_knowledge_base"]


def test_a_knowledge_search_is_not_an_action():
    def searches(customer_turn, env):
        env.act(ToolCall("c1", "search_knowledge_base", {"query": "refund window"}))
        return "Looked it up."

    outcome = run_case(PRORATED, searches, transport=StubTransport())

    assert outcome.actions_attempted == ()
    assert outcome.transcript[-1].tool_results[0].name == "search_knowledge_base"


def test_the_embeddings_retriever_plugs_into_the_case_runner():
    def embed(texts):
        return [[float("refund" in text.lower()), 1.0] for text in texts]

    retriever = EmbeddingsRetriever(load_articles(), embed=embed)
    seen = []

    def searches(customer_turn, env):
        seen.append(env.knowledge)
        return env.act(ToolCall("c1", "search_knowledge_base", {"query": "refund"})).content

    outcome = run_case(PRORATED, searches, transport=StubTransport(), retriever=retriever)

    assert seen == [retriever]
    assert "(article " in outcome.reply


# ── The suite, cumulatively ───────────────────────────────────────────────────


def test_the_spine_2_suite_runs_the_three_knowledge_cases():
    assert set(CASES) == {
        "downgrade-with-prorated-refund",
        "annual-refund-outside-window",
        "upgrade-after-a-question",
    }


def test_the_spine_2_reference_agent_passes_modules_1_and_2_offline(no_network):
    report = run_checks(through=2, agent=AGENT, out=lambda line: None)

    assert report.passed is True, [(r.case_id, r.name, r.detail) for r in report.results
                                   if r.status != "pass"]
    assert {r.module for r in report.results} == {1, 2}


def test_the_spine_1_checks_still_pass_offline_for_the_spine_1_agent(no_network):
    report = run_checks(through=1, agent="flagship.loop:run", out=lambda line: None)

    assert report.passed is True


def test_the_checks_cli_grades_the_latest_reference_agent_by_default(capsys, no_network):
    from checks.cli import main

    code = main(["--modules", "2"])

    out = capsys.readouterr().out
    assert code == 0, out
    assert "agent flagship.knowledge:run" in out
    assert out.strip().splitlines()[-1] == "Passed through module 2 of 2."


def test_a_case_stopped_before_its_last_turn_has_no_final_reply():
    answers = {
        "content": [{"type": "text", "text": "Pro is $12.00 a month."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }

    outcome = run_case(MULTI_TURN, AGENT, transport=StubTransport([answers]), step_limit=1)

    assert outcome.step_limit_reached is True
    assert outcome.replies == ("Pro is $12.00 a month.",)
    assert outcome.reply is None
