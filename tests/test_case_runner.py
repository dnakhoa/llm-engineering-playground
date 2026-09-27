"""Seam 1: run a Case against an agent, get an Outcome.

Tested the way the spec says: through the Outcome, in Offline mode, never
against loop internals or prompt text.
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from checks.spine_1_loop import plan_changed_to_pro_exactly_once  # noqa: E402
from company.runner import load_case, run_case  # noqa: E402
from llm.replay import ReplayMismatchError  # noqa: E402
from llm.testing import StubTransport  # noqa: E402

AGENTS = "tests.fixtures.case_runner_agents"
UPGRADE = load_case("upgrade-to-pro")


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("an Offline run tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_the_reference_agent_passes_the_offline_check_with_no_network(no_network):
    outcome = run_case(UPGRADE, "flagship.loop:run", mode="offline")

    assert outcome.resolved is True
    assert outcome.diff == {"accounts.acct_1001.plan": {"before": "free", "after": "pro"}}
    assert [a.name for a in outcome.actions_executed] == ["look_up_account", "change_plan"]
    assert outcome.step_limit_reached is False
    assert plan_changed_to_pro_exactly_once(outcome).passed is True


def test_the_offline_outcome_carries_transcript_usage_and_cost(no_network):
    outcome = run_case(UPGRADE, "flagship.loop:run")

    # The recording's three turns: 812 + 921 + 1003 in, 61 + 58 + 47 out.
    assert outcome.usage.input_tokens == 2736
    assert outcome.usage.output_tokens == 166
    # claude-sonnet-5 at $2 in / $10 out per million tokens.
    assert outcome.cost_usd == pytest.approx(0.007132)
    assert outcome.transcript[0].text == UPGRADE.opening_message
    assert outcome.transcript[-1].text == outcome.reply
    assert "Pro" in outcome.reply


def test_an_offline_run_on_a_model_the_recording_never_saw_fails_loudly(no_network):
    with pytest.raises(ReplayMismatchError):
        run_case(UPGRADE, "flagship.loop:run", model="claude-opus-5-5")


def test_an_agent_that_never_calls_the_action_is_unresolved_and_fails_the_check():
    outcome = run_case(
        UPGRADE, AGENTS + ":chats_but_never_acts", transport=StubTransport()
    )

    assert outcome.resolved is False
    assert outcome.actions_executed == ()
    assert outcome.diff == {}
    assert plan_changed_to_pro_exactly_once(outcome).passed is False


def _looks_up_the_account():
    return {
        "content": [
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "look_up_account",
                "input": {"account_id": "acct_1001"},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def test_the_step_limit_stops_a_looping_agent_and_is_reported():
    stub = StubTransport([_looks_up_the_account() for _ in range(50)])

    outcome = run_case(UPGRADE, AGENTS + ":loops_for_ever", transport=stub, step_limit=3)

    assert outcome.step_limit_reached is True
    assert outcome.steps == 3
    assert len(stub.requests) == 3
    assert outcome.resolved is False
    assert "step limit" in plan_changed_to_pro_exactly_once(outcome).detail


def test_the_reference_loop_is_stopped_by_the_step_limit_too():
    stub = StubTransport([_looks_up_the_account() for _ in range(50)])

    outcome = run_case(UPGRADE, "flagship.loop:run", transport=stub, step_limit=4)

    assert outcome.step_limit_reached is True
    assert outcome.steps == 4
    assert outcome.resolved is False


def _says(text):
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def test_an_agent_that_claims_the_upgrade_without_acting_fails_the_check():
    # The reply claims success. Only Backend state can tell it is a lie.
    stub = StubTransport([_says("Done! You're on Pro now.")])

    outcome = run_case(UPGRADE, AGENTS + ":claims_without_acting", transport=stub)

    assert outcome.reply == "Done! You're on Pro now."
    assert outcome.resolved is False
    assert outcome.actions_attempted == ()
    assert outcome.final_state["accounts"]["acct_1001"]["plan"] == "free"
    assert plan_changed_to_pro_exactly_once(outcome).passed is False


def test_upgrading_downgrading_and_upgrading_again_fails_the_check():
    outcome = run_case(
        UPGRADE,
        AGENTS + ":upgrades_downgrades_and_upgrades_again",
        transport=StubTransport(),
    )

    # The account does end on Pro, which is why the Check counts the changes.
    assert outcome.final_state["accounts"]["acct_1001"]["plan"] == "pro"
    assert [a.arguments["plan"] for a in outcome.actions_executed] == ["pro", "free", "pro"]
    result = plan_changed_to_pro_exactly_once(outcome)
    assert result.passed is False
    assert "2 change_plan calls to Pro" in result.detail


def test_upgrading_a_different_account_fails_the_check():
    outcome = run_case(
        UPGRADE, AGENTS + ":upgrades_someone_elses_account", transport=StubTransport()
    )

    assert outcome.actions_executed == ()
    assert outcome.final_state["accounts"]["acct_1003"]["plan"] == "team"
    assert outcome.final_state["accounts"]["acct_1001"]["plan"] == "free"
    assert plan_changed_to_pro_exactly_once(outcome).passed is False


def test_a_cross_account_change_fails_the_check():
    outcome = run_case(
        UPGRADE, AGENTS + ":changes_someone_elses_plan", transport=StubTransport()
    )

    assert plan_changed_to_pro_exactly_once(outcome).passed is False


def test_the_transcript_holds_every_turn_of_the_offline_run(no_network):
    outcome = run_case(UPGRADE, "flagship.loop:run")

    turns = [
        (m.role, m.text, [(c.name, dict(c.arguments)) for c in m.tool_calls],
         [(r.name, r.content, r.is_error) for r in m.tool_results])
        for m in outcome.transcript
    ]
    assert turns == [
        ("user", UPGRADE.opening_message, [], []),
        ("assistant", "Let me look up your account first.",
         [("look_up_account", {"account_id": "acct_1001"})], []),
        ("tool", None, [],
         [("look_up_account",
           '{"account_id": "acct_1001", "name": "Juniper Lane Bakery", '
           '"plan": "free", "seats": 1}', False)]),
        ("assistant", "Juniper Lane Bakery is on Free. Moving it to Pro now.",
         [("change_plan", {"account_id": "acct_1001", "plan": "pro"})], []),
        ("tool", None, [], [("change_plan", "acct_1001 moved from free to pro.", False)]),
        ("assistant", "Done! Juniper Lane Bakery (acct_1001) is now on the Pro plan.", [], []),
    ]


def test_the_transcript_keeps_a_refused_action_and_the_turns_after_it():
    wrong_account = {
        "content": [
            {"type": "tool_use", "id": "call_1", "name": "change_plan",
             "input": {"account_id": "acct_1002", "plan": "pro"}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    stub = StubTransport([wrong_account, _says("I can only change your own account.")])

    outcome = run_case(UPGRADE, "flagship.loop:run", transport=stub)

    turns = [
        (m.role, m.text, [c.name for c in m.tool_calls],
         [(r.content, r.is_error) for r in m.tool_results])
        for m in outcome.transcript
    ]
    assert turns == [
        ("user", UPGRADE.opening_message, [], []),
        ("assistant", None, ["change_plan"], []),
        ("tool", None, [],
         [("Refused: acct_1002 is not the account of the customer on this Case.", True)]),
        ("assistant", "I can only change your own account.", [], []),
    ]


def test_a_cross_account_plan_change_is_attempted_but_never_executed():
    outcome = run_case(
        UPGRADE, AGENTS + ":changes_someone_elses_plan", transport=StubTransport()
    )

    assert [a.name for a in outcome.actions_attempted] == ["change_plan"]
    assert outcome.actions_executed == ()
    assert outcome.final_state["accounts"]["acct_1002"]["plan"] == "pro"
    assert outcome.diff == {}


# ── Each Case declares its Actions (ADR 0005) ─────────────────────────────────


def test_adding_issue_refund_leaves_the_upgrade_to_pro_replay_unchanged(no_network):
    # The Backend now has an Action Spine 1 never had...
    from company.backend import ACTION_NAMES

    assert "issue_refund" in ACTION_NAMES
    # ...but the old Case offers only what it declares, so its request, and
    # therefore its reviewed recording and its Offline Check, stay as they were.
    assert UPGRADE.actions == ("look_up_account", "change_plan")
    for agent in ("flagship.loop:run", "flagship.knowledge:run"):
        outcome = run_case(UPGRADE, agent, mode="offline")
        assert outcome.resolved is True, agent
        assert plan_changed_to_pro_exactly_once(outcome).passed is True, agent


def test_the_upgrade_to_pro_request_offers_only_the_declared_actions():
    stub = StubTransport()

    run_case(UPGRADE, "flagship.loop:run", transport=stub)

    offered = [tool["name"] for tool in stub.last_request.body["tools"]]
    assert offered == ["look_up_account", "change_plan"]


def test_offering_every_action_would_have_broken_the_old_recording(no_network):
    from dataclasses import replace

    from company.backend import ACTION_NAMES

    every_action = replace(UPGRADE, actions=ACTION_NAMES)

    with pytest.raises(ReplayMismatchError, match="body.tools"):
        run_case(every_action, "flagship.loop:run", mode="offline")


def test_a_call_to_an_undeclared_action_is_refused_and_recorded_as_attempted():
    outcome = run_case(
        UPGRADE, AGENTS + ":refunds_on_a_case_that_offers_no_refunds",
        transport=StubTransport(),
    )

    assert [(a.name, a.executed) for a in outcome.actions_attempted] == [
        ("issue_refund", False)
    ]
    assert outcome.actions_executed == ()
    assert "not available on this Case" in outcome.actions_attempted[0].result
    assert outcome.final_state["refunds"] == {}


def test_the_system_prompt_is_part_of_the_recording(no_network):
    # Changing the system prompt changes the request, so it needs a re-recording.
    with pytest.raises(ReplayMismatchError, match="body.system"):
        run_case(UPGRADE, AGENTS + ":loop_with_another_system_prompt", mode="offline")


def test_a_case_that_does_not_declare_its_actions_will_not_load(tmp_path):
    case_file = tmp_path / "undeclared.json"
    case_file.write_text(
        '{"id": "undeclared", "customer": {"account_id": "acct_1001"}, '
        '"opening_message": "Hi", "expected_state_change": {}}'
    )

    with pytest.raises(ValueError, match="declare"):
        load_case(case_file)


def test_a_case_that_declares_an_action_the_backend_lacks_will_not_load(tmp_path):
    case_file = tmp_path / "unknown.json"
    case_file.write_text(
        '{"id": "unknown", "customer": {"account_id": "acct_1001"}, '
        '"opening_message": "Hi", "expected_state_change": {}, '
        '"actions": ["look_up_account", "delete_everything"]}'
    )

    with pytest.raises(ValueError, match="delete_everything"):
        load_case(case_file)


def test_a_case_that_declares_an_action_twice_will_not_load(tmp_path):
    case_file = tmp_path / "twice.json"
    case_file.write_text(
        '{"id": "twice", "customer": {"account_id": "acct_1001"}, '
        '"opening_message": "Hi", "expected_state_change": {}, '
        '"actions": ["change_plan", "change_plan"]}'
    )

    with pytest.raises(ValueError, match="more than once"):
        load_case(case_file)
