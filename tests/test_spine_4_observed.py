"""Spine 4 (Observed): every Case run leaves an OpenTelemetry trace.

Tested through the Case runner's Outcome and the Checks, Offline, the way the
spec asks. The worked numbers come from the upgrade-to-pro recording
(company/recordings/upgrade-to-pro.recording.json): three model calls, msg_hand_01
to msg_hand_03, using 812 + 921 + 1003 = 2736 input and 61 + 58 + 47 = 166
output tokens. claude-sonnet-5 is $2 in and $10 out per million tokens, so the
Case costs 2736 x 2 / 1e6 + 166 x 10 / 1e6 = $0.007132.

Ticket: docs/tickets/graded-attacked-budgeted/07-spine-4-observed.md
"""
from __future__ import annotations

import dataclasses
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from checks import spine_1_loop, spine_2_knowledge, spine_3_graded  # noqa: E402
from checks.cli import DEFAULT_AGENT, run_checks  # noqa: E402
from checks.spine_4_observed import (  # noqa: E402
    CASES,
    CHECKS,
    every_model_call_has_a_chat_span,
    every_tool_call_has_a_tool_span,
    span_tokens_and_cost_match_the_outcome,
    spans_follow_the_pinned_conventions,
)
from company.runner import load_case, run_case  # noqa: E402
from llm.types import ToolCall  # noqa: E402

AGENT = "flagship.knowledge:run"
OBSERVED_AGENT = "flagship.observed:run"
UPGRADE = load_case("upgrade-to-pro")
PRORATED = load_case("downgrade-with-prorated-refund")


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("an Offline run tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _op(span):
    return span.attributes["gen_ai.operation.name"]


# ── The trace on the Outcome ──────────────────────────────────────────────────


def test_a_case_run_leaves_an_agent_span_over_chat_and_tool_spans(no_network):
    trace = run_case(UPGRADE, AGENT, mode="offline").trace

    (agent,) = [s for s in trace.spans if _op(s) == "invoke_agent"]
    chats = [s for s in trace.spans if _op(s) == "chat"]
    tools = [s for s in trace.spans if _op(s) == "execute_tool"]

    assert agent.parent_id is None
    assert [s.name for s in chats] == ["chat claude-sonnet-5"] * 3
    assert [s.attributes["gen_ai.response.id"] for s in chats] == [
        "msg_hand_01", "msg_hand_02", "msg_hand_03"]
    assert [(s.name, s.attributes["gen_ai.tool.call.id"]) for s in tools] == [
        ("execute_tool look_up_account", "toolu_hand_01"),
        ("execute_tool change_plan", "toolu_hand_02"),
    ]
    assert {s.parent_id for s in chats + tools} == {agent.span_id}
    assert {s.trace_id for s in trace.spans} == {agent.trace_id}


def test_the_chat_spans_carry_each_calls_tokens_and_cost(no_network):
    outcome = run_case(UPGRADE, AGENT, mode="offline")
    chats = [s for s in outcome.trace.spans if _op(s) == "chat"]

    assert [s.attributes["gen_ai.usage.input_tokens"] for s in chats] == [812, 921, 1003]
    assert [s.attributes["gen_ai.usage.output_tokens"] for s in chats] == [61, 58, 47]
    assert sum(s.attributes["acme.cost_usd"] for s in chats) == pytest.approx(0.007132)
    assert outcome.cost_usd == pytest.approx(0.007132)


def test_the_agent_span_carries_the_cases_totals(no_network):
    (agent,) = run_case(UPGRADE, AGENT, mode="offline").trace.agent_spans

    assert agent.attributes["gen_ai.usage.input_tokens"] == 2736
    assert agent.attributes["gen_ai.usage.output_tokens"] == 166
    assert agent.attributes["acme.cost_usd"] == pytest.approx(0.007132)
    assert agent.attributes["gen_ai.conversation.id"] == "upgrade-to-pro"


def test_the_trace_comes_with_the_verdict_not_instead_of_it(no_network):
    outcome = run_case(PRORATED, AGENT, mode="offline")

    assert outcome.verdict.resolved is True and outcome.resolved is True
    assert len(outcome.trace.chat_spans) == outcome.steps


def test_an_exporter_receives_the_same_spans_as_the_outcome(no_network):
    # Any OpenTelemetry backend is an exporter; an in-memory one stands in for it.
    backend = InMemorySpanExporter()

    outcome = run_case(UPGRADE, AGENT, mode="offline", exporters=[backend])

    exported = backend.get_finished_spans()
    assert sorted(s.name for s in exported) == sorted(s.name for s in outcome.trace.spans)
    assert len(exported) == 6  # one agent, three chat and two tool spans


def test_the_trace_is_pinned_to_a_named_convention_version(no_network):
    trace = run_case(UPGRADE, AGENT, mode="offline").trace

    assert trace.semconv_version == "1.41.0"
    assert {s.schema_url for s in trace.spans} == {"https://opentelemetry.io/schemas/1.41.0"}


def test_a_step_limit_stop_is_an_error_on_the_agent_span(no_network):
    outcome = run_case(UPGRADE, AGENT, mode="offline", step_limit=1)

    (agent,) = outcome.trace.agent_spans
    assert agent.status == "ERROR"
    assert agent.attributes["error.type"] == "step_limit_reached"
    assert outcome.verdict.unfinished  # the verdict says so too


def test_a_refused_action_is_an_error_on_its_tool_span(no_network):
    def refunds_unasked(customer_turn, env):
        # upgrade-to-pro does not declare issue_refund, so the Backend refuses it.
        env.act(ToolCall("x1", "issue_refund", {
            "account_id": "acct_1001", "invoice_id": "inv_1001", "amount_usd": 5.00}))
        return "Refunded!"

    (tool,) = run_case(UPGRADE, refunds_unasked, mode="offline").trace.tool_spans

    assert tool.status == "ERROR"
    assert tool.attributes["error.type"] == "tool_error"


# ── The Spine 4 Checks ────────────────────────────────────────────────────────


def _results(outcome):
    return {check.__name__: check(outcome) for check in CHECKS}


def _without(outcome, span):
    spans = tuple(s for s in outcome.trace.spans if s is not span)
    return dataclasses.replace(outcome, trace=dataclasses.replace(outcome.trace, spans=spans))


@pytest.mark.parametrize("case_id", CASES)
def test_the_spine_4_reference_agent_passes_every_observed_check(no_network, case_id):
    outcome = run_case(load_case(case_id), OBSERVED_AGENT, mode="offline")

    failed = {name: r.detail for name, r in _results(outcome).items() if not r.passed}
    assert failed == {}
    assert outcome.resolved is True


def test_the_suite_runs_every_case_so_far():
    earlier = set(spine_1_loop.CASES) | set(spine_2_knowledge.CASES) | set(spine_3_graded.CASES)
    assert set(CASES) == earlier
    assert len(CASES) == len(set(CASES))


def test_the_suites_cases_are_derived_not_hand_copied():
    # ADR 0006: copies drift. No Case id is written into the suite's source.
    from checks.suites import cases_before

    source = (Path(__file__).parent.parent / "checks" / "spine_4_observed.py").read_text(
        encoding="utf-8")
    assert CASES == cases_before(4)
    assert not [case for case in CASES if '"{}"'.format(case) in source]


def test_cases_before_a_module_are_every_earlier_suites_cases_once_in_order():
    from checks.suites import cases_before

    assert cases_before(1) == ()
    assert cases_before(2) == spine_1_loop.CASES
    assert cases_before(4) == tuple(dict.fromkeys(
        spine_1_loop.CASES + spine_2_knowledge.CASES + spine_3_graded.CASES))


def test_every_trace_check_holds_on_every_case():
    from checks import is_on_every_case

    assert [c.__name__ for c in CHECKS if not is_on_every_case(c)] == []


def test_a_case_a_later_suite_adds_gets_the_trace_checks(no_network):
    # A module 5 suite that lists none of Spine 4's Checks, on a Case of its own.
    from checks.suites import Suite, discover_suites
    from tests.fixtures.checks_cli_suites import AGAIN

    later = Suite(module=5, title="Later", cases=(AGAIN,), checks=())
    lines = []

    report = run_checks(through=5, agent=OBSERVED_AGENT,
                        suites=discover_suites() + (later,), out=lines.append)

    graded = [(r.name, r.status) for r in report.results
              if r.module == 5 and r.case_id == "upgrade-to-pro-again"]
    assert {name for name, _ in graded} >= {check.__name__ for check in CHECKS}
    assert all(status == "pass" for _, status in graded), graded
    assert report.passed is True, "\n".join(lines)


#: ADR 0006: offline, each module's Cases replay against that module's own
#: reference agent, the agent whose requests the recording holds.
_OWN_REFERENCE_AGENT = {1: "flagship.loop:run", 2: "flagship.knowledge:run",
                        3: "flagship.knowledge:run", 4: OBSERVED_AGENT}


@pytest.mark.parametrize("case_id", CASES)
def test_each_earlier_case_passes_the_trace_checks_on_its_own_modules_agent(
        no_network, case_id):
    # The older reference agents do not name themselves: that is a warning,
    # never a failure, so the trace Checks on every Case and ADR 0006 agree.
    case = load_case(case_id)
    agent = _OWN_REFERENCE_AGENT[case.tags["module"]]

    results = _results(run_case(case, agent, mode="offline"))

    assert {name: r.detail for name, r in results.items() if not r.passed} == {}
    warned = {name for name, r in results.items() if r.warning}
    assert warned == (set() if agent == OBSERVED_AGENT
                      else {"the_agent_span_names_the_agent"})


def test_the_suite_lists_only_its_own_checks():
    # The Checks CLI already runs every earlier suite; re-listing their Checks
    # here would print each earlier result twice.
    earlier = set(spine_1_loop.CHECKS) | set(spine_2_knowledge.CHECKS) | set(spine_3_graded.CHECKS)
    assert not set(CHECKS) & earlier
    assert all(check.__module__ == "checks.spine_4_observed" for check in CHECKS)


def test_a_tool_call_with_no_span_fails(no_network):
    # The agent answers Knowledge Base searches itself, not through env.act: the
    # Backend ends right and the replay still matches, but the search is untraced.
    outcome = run_case(
        PRORATED, "tests.fixtures.observed_agents:searches_without_a_span", mode="offline")

    result = every_tool_call_has_a_tool_span(outcome)

    assert outcome.resolved is True
    assert result.passed is False
    assert "search_knowledge_base" in result.detail


def test_the_checks_cli_fails_module_4_for_a_tool_call_with_no_span(no_network):
    # What the Reader runs: the same untraced agent passes modules 1 to 3, and
    # the command stops at module 3 on the Knowledge Base searches it hid.
    lines = []

    report = run_checks(
        through=4, agent="tests.fixtures.observed_agents:searches_without_a_span",
        out=lines.append)

    failed = [r for r in report.results if r.status == "fail"]
    assert report.passed_through == 3, "\n".join(lines)
    assert lines[-1] == "Passed through module 3 of 4."
    assert {(r.module, r.name) for r in failed} == {(4, "every_tool_call_has_a_tool_span")}
    assert all("search_knowledge_base" in r.detail for r in failed)
    fail_lines = [line for line in lines if line.strip().startswith("FAIL")]
    assert len(fail_lines) == len(failed) > 0
    assert all("every tool call has a tool span" in line and "search_knowledge_base" in line
               for line in fail_lines), fail_lines


def test_a_dropped_tool_span_fails(no_network):
    outcome = run_case(UPGRADE, OBSERVED_AGENT, mode="offline")
    (change_plan,) = [s for s in outcome.trace.tool_spans if s.name == "execute_tool change_plan"]

    result = every_tool_call_has_a_tool_span(_without(outcome, change_plan))

    assert result.passed is False
    assert "change_plan (toolu_hand_02)" in result.detail


def test_span_tokens_that_disagree_with_the_outcome_fail(no_network):
    outcome = run_case(UPGRADE, OBSERVED_AGENT, mode="offline")
    assert span_tokens_and_cost_match_the_outcome(outcome).passed is True

    first_chat = outcome.trace.chat_spans[0]
    doctored = dataclasses.replace(first_chat, attributes={
        **first_chat.attributes, "gen_ai.usage.output_tokens": 1})
    spans = tuple(doctored if s is first_chat else s for s in outcome.trace.spans)
    wrong = dataclasses.replace(outcome, trace=dataclasses.replace(outcome.trace, spans=spans))

    result = span_tokens_and_cost_match_the_outcome(wrong)
    assert result.passed is False
    assert "output tokens" in result.detail


def test_span_cost_that_disagrees_with_the_outcome_fails(no_network):
    outcome = run_case(UPGRADE, OBSERVED_AGENT, mode="offline")

    wrong = dataclasses.replace(outcome, cost_usd=outcome.cost_usd * 2)

    result = span_tokens_and_cost_match_the_outcome(wrong)
    assert result.passed is False
    assert "cost" in result.detail


def test_a_model_call_with_no_chat_span_fails(no_network):
    outcome = run_case(UPGRADE, OBSERVED_AGENT, mode="offline")

    result = every_model_call_has_a_chat_span(_without(outcome, outcome.trace.chat_spans[-1]))

    assert result.passed is False
    assert "3 model calls" in result.detail


def test_a_span_on_another_convention_version_fails(no_network):
    outcome = run_case(UPGRADE, OBSERVED_AGENT, mode="offline")
    first = outcome.trace.spans[0]
    moved = dataclasses.replace(first, schema_url="https://opentelemetry.io/schemas/1.44.0")
    spans = (moved,) + outcome.trace.spans[1:]
    wrong = dataclasses.replace(outcome, trace=dataclasses.replace(outcome.trace, spans=spans))

    result = spans_follow_the_pinned_conventions(wrong)

    assert result.passed is False
    assert "1.41.0" in result.name and "1.44.0" in result.detail


def test_an_agent_that_never_names_itself_is_warned_not_failed(no_network):
    # gen_ai.agent.name is set "if provided by the application": leaving it
    # out breaks no convention, so a correct Spine 1 to 3 agent is not failed.
    outcome = run_case(UPGRADE, AGENT, mode="offline")  # the Spine 2 agent

    results = _results(outcome)
    assert [name for name, r in results.items() if not r.passed] == []
    assert [name for name, r in results.items() if r.warning] == [
        "the_agent_span_names_the_agent"]
    assert "env.describe_agent" in results["the_agent_span_names_the_agent"].detail


def test_an_agent_that_names_itself_gets_no_warning(no_network):
    outcome = run_case(UPGRADE, OBSERVED_AGENT, mode="offline")

    assert not any(r.warning for r in _results(outcome).values())


def test_the_spine_2_agent_passes_module_4_with_a_naming_warning(no_network):
    lines = []

    report = run_checks(through=4, agent=AGENT, out=lines.append)

    assert report.passed is True, "\n".join(lines)
    assert lines[-1] == "Passed through module 4 of 4."
    warned = [r for r in report.results if r.status == "warn"]
    assert {r.name for r in warned} == {"the_agent_span_names_the_agent"}
    assert report.warnings == len(warned) > 0


def test_the_checks_cli_passes_the_reference_agent_through_module_4(no_network):
    lines = []

    report = run_checks(through=4, agent=OBSERVED_AGENT, out=lambda line: lines.append(line))

    assert report.passed is True, "\n".join(lines)
    assert lines[-1] == "Passed through module 4 of 4."


def test_each_check_prints_once_per_case(no_network):
    lines = []

    run_checks(through=4, agent=OBSERVED_AGENT, out=lambda line: lines.append(line))

    results = [line.strip() for line in lines if line.strip().startswith(("PASS", "SKIP"))]
    assert len(results) == len(set(results)), [r for r in results if results.count(r) > 1]


def test_the_checks_cli_defaults_to_the_spine_4_agent():
    assert DEFAULT_AGENT == OBSERVED_AGENT
