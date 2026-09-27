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
    reply_meets_the_judge_rubric,
)
from checks.cli import run_checks  # noqa: E402
from checks.suites import Suite  # noqa: E402
from company.runner import StepLimitReached, load_case, run_case  # noqa: E402
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


# ── An agent that never finishes fails ───────────────────────────────────────


def _replies_empty(customer_turn, env):
    return ""


def _replies_blank(customer_turn, env):
    return "  \n"


def _replies_nothing(customer_turn, env):
    return None


def _hits_the_step_limit(customer_turn, env):
    raise StepLimitReached(env.step_limit)


NEVER_FINISHES = (_replies_empty, _replies_blank, _replies_nothing, _hits_the_step_limit)


@pytest.mark.parametrize("agent", NEVER_FINISHES, ids=lambda agent: agent.__name__)
@pytest.mark.parametrize("case_id", CASES)
def test_an_agent_that_never_finishes_fails_every_case(agent, case_id):
    # Including the Case where nothing was meant to change, which an agent
    # that does nothing would otherwise reach by doing nothing.
    outcome = run_case(load_case(case_id), agent, transport=StubTransport())

    assert outcome.resolved is False
    assert _graded(outcome)["the_agent_finishes_the_case"] is False


@pytest.mark.parametrize("agent", NEVER_FINISHES, ids=lambda agent: agent.__name__)
def test_the_checks_cli_fails_every_case_of_an_agent_that_never_finishes(agent, no_network):
    report = run_checks(through=3, agent=agent, out=lambda line: None)

    failed = {(r.module, r.case_id) for r in report.results if r.status == "fail"}
    graded = {(r.module, r.case_id) for r in report.results}
    assert failed == graded
    assert report.passed_through == 0


# ── A seeded regression: refunding twice ─────────────────────────────────────

REFUNDS_TWICE = "tests.fixtures.graded_agents:refunds_twice"
PRORATED = load_case("downgrade-with-prorated-refund")


def test_refunding_twice_leaves_the_right_state_but_is_not_resolved(no_network):
    # The Backend refuses the second refund, because the first used up what the
    # invoice owes. A $5.00 refund sent twice would have run twice.
    outcome = run_case(PRORATED, REFUNDS_TWICE, mode="offline")

    assert [(a.name, a.executed) for a in outcome.actions_attempted] == [
        ("look_up_account", True), ("change_plan", True),
        ("issue_refund", True), ("issue_refund", False)]
    assert outcome.verdict.missing == () and outcome.verdict.unexpected == ()
    assert outcome.resolved is False
    assert _graded(outcome)["each_change_is_attempted_once"] is False


def test_the_seeded_regression_fails_the_graded_suite(capsys, no_network):
    from checks.cli import main

    code = main(["--modules", "3", "--agent", REFUNDS_TWICE])

    out = capsys.readouterr().out
    failed = [line.split() for line in out.splitlines() if line.strip().startswith("FAIL")]
    assert code == 1, out
    assert [row[1] for row in failed] == [PRORATED.id + ":"], out
    assert "issue_refund" in "\n".join(line for line in out.splitlines() if "FAIL" in line)
    # The Knowledge suite reads only the Backend's final state, which is right.
    assert out.strip().splitlines()[-1] == "Passed through module 2 of 3."


# ── The Graded suite ──────────────────────────────────────────────────────────


def test_the_reference_agent_passes_modules_1_to_3_offline(no_network):
    report = run_checks(through=3, agent=AGENT, out=lambda line: None)

    assert report.passed is True, [(r.case_id, r.name, r.detail) for r in report.results
                                   if r.status not in ("pass", "skip")]
    assert {r.module for r in report.results} == {1, 2, 3}
    assert report.passed_through == 3


def test_the_graded_suite_covers_every_action_built_so_far(no_network):
    from company.backend import ACTION_NAMES

    ran = {
        record.name
        for case_id in CASES
        for record in run_case(load_case(case_id), AGENT, mode="offline").actions_executed
    }
    assert set(ACTION_NAMES) == {"look_up_account", "change_plan", "issue_refund"}
    assert ran == set(ACTION_NAMES)


def test_every_graded_case_has_a_recording_so_it_runs_offline():
    assert all(load_case(case_id).recording for case_id in CASES)


def test_every_graded_case_carries_a_judge_rubric():
    assert all(load_case(case_id).judge_rubric for case_id in CASES)


# ── The judge rubric: live only ───────────────────────────────────────────────


def test_the_judge_rubric_is_live_only_and_skipped_offline(no_network):
    report = run_checks(through=3, agent=AGENT, out=lambda line: None)

    judged = [r for r in report.results if r.name == reply_meets_the_judge_rubric.__name__]
    assert is_live_only(reply_meets_the_judge_rubric)
    assert [r.case_id for r in judged] == list(CASES)
    assert {r.status for r in judged} == {"skip"}
    assert report.passed is True


def _verdict(text, input_tokens=1000, output_tokens=100):
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _upgrades_without_the_model(customer_turn, env):
    env.act(ToolCall("c1", "look_up_account", {"account_id": "acct_1001"}))
    env.act(ToolCall("c2", "change_plan", {"account_id": "acct_1001", "plan": "pro"}))
    return "UPGRADED-REPLY: you're on Pro now."


def _judged(transport, spend_cap_usd=1.00):
    """A live run of one Graded Case whose only model call is the judge's."""
    suite = Suite(module=3, title="Graded", cases=("upgrade-to-pro",),
                  checks=(reply_meets_the_judge_rubric,))
    return run_checks(through=3, agent=_upgrades_without_the_model, mode="live",
                      transport=transport, suites=(suite,), spend_cap_usd=spend_cap_usd,
                      out=lambda line: None)


def test_live_the_judge_reads_the_rubric_and_the_reply_and_its_verdict_counts():
    stub = StubTransport([_verdict('{"verdict": "pass", "reason": "Confirms the new plan."}')])

    report = _judged(stub)

    (result,) = report.results
    assert (result.status, result.detail) == ("pass", "Confirms the new plan.")
    (request,) = stub.requests
    sent = str(request.body)
    assert load_case("upgrade-to-pro").judge_rubric in sent
    assert "UPGRADED-REPLY" in sent
    assert "change_plan" in sent  # the judge sees what actually ran


def test_live_a_failing_verdict_fails_the_check_with_the_judges_reason():
    report = _judged(StubTransport([
        _verdict('Here is my verdict: {"verdict": "fail", "reason": "Never names the plan."}')]))

    (result,) = report.results
    assert (result.status, result.detail) == ("fail", "Never names the plan.")
    assert report.passed is False


def test_live_an_answer_that_is_not_a_verdict_fails_the_check():
    report = _judged(StubTransport([_verdict("Looks fine to me!")]))

    (result,) = report.results
    assert result.status == "fail"
    assert "Looks fine to me!" in result.detail


def test_live_the_judges_calls_count_against_the_spend_cap():
    # 1,000 in and 100 out on claude-sonnet-5: $0.002 + $0.001.
    report = _judged(StubTransport([_verdict('{"verdict": "pass", "reason": "ok"}')]))

    assert report.spent_usd == pytest.approx(0.003)


def test_live_a_spend_cap_the_run_has_used_up_stops_the_judge():
    stub = StubTransport([_verdict('{"verdict": "pass", "reason": "ok"}')])

    report = _judged(stub, spend_cap_usd=0.0)

    (result,) = report.results
    assert result.status == "stopped"
    assert stub.requests == []
    assert report.stopped_by_spend_cap is True


def test_without_a_judge_the_rubric_check_says_to_run_it_live(no_network):
    outcome = run_case(load_case("upgrade-to-pro"), AGENT, mode="offline")

    result = reply_meets_the_judge_rubric(outcome)
    assert result.passed is False
    assert "live" in result.detail


PANEL = (
    AGENT,
    REFUNDS_TWICE,
    _downgrades_unasked,
    _refunds_anyway,
    *NEVER_FINISHES,
    "tests.fixtures.knowledge_agents:refunds_five_dollars",
    "tests.fixtures.knowledge_agents:refunds_the_policy_amount_in_two_parts",
    "tests.fixtures.case_runner_agents:upgrades_downgrades_and_upgrades_again",
    "tests.fixtures.case_runner_agents:changes_someone_elses_plan",
)


@pytest.mark.parametrize("agent", PANEL, ids=lambda a: a if isinstance(a, str) else a.__name__)
@pytest.mark.parametrize("case_id", CASES)
def test_resolved_is_exactly_every_offline_graded_check_passing(agent, case_id):
    # One definition: the Budgeted Checks and the Scoreboard count `resolved`,
    # so it must never call resolved a run the Graded Checks fail, or the
    # other way round.
    case = load_case(case_id)
    transport = None if agent in (AGENT, REFUNDS_TWICE) else StubTransport()
    outcome = run_case(case, agent, transport=transport)

    assert outcome.resolved is all(_graded(outcome).values()), _graded(outcome)
