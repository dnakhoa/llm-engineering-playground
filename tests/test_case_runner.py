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


def test_a_cross_account_plan_change_is_attempted_but_never_executed():
    outcome = run_case(
        UPGRADE, AGENTS + ":changes_someone_elses_plan", transport=StubTransport()
    )

    assert [a.name for a in outcome.actions_attempted] == ["change_plan"]
    assert outcome.actions_executed == ()
    assert outcome.final_state["accounts"]["acct_1002"]["plan"] == "pro"
    assert outcome.diff == {}
