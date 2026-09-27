"""Spine 4 (Observed) Checks.

They hold the Outcome's trace to the OpenTelemetry GenAI semantic conventions,
at the version ``company.tracing`` pins. A trace that is merely present passes
nothing here: every model call needs its chat span, every tool call the model
asked for needs its tool span, and the tokens and cost on the spans must add
up to the Outcome's, because the Budgeted module prices the agent from them.

They run on every Case so far, so that multi-turn Cases and Knowledge Base
searches are traced as completely as a single-turn upgrade.
"""
from __future__ import annotations

import math
from typing import List

from company.runner import Outcome
from company.tracing import (
    AGENT_NAME,
    CHAT,
    COST_USD,
    EXECUTE_TOOL,
    INVOKE_AGENT,
    OPERATION_NAME,
    PROVIDER_NAME,
    REQUEST_MODEL,
    SCHEMA_URL,
    SEMCONV_VERSION,
    TOOL_CALL_ID,
    TOOL_NAME,
    USAGE_CACHE_CREATION_INPUT_TOKENS,
    USAGE_CACHE_READ_INPUT_TOKENS,
    USAGE_INPUT_TOKENS,
    USAGE_OUTPUT_TOKENS,
    SpanRecord,
)
from llm.types import ROLE_ASSISTANT

from . import CheckResult

MODULE = 4
TITLE = "Observed"
CASES = (
    "upgrade-to-pro",
    "downgrade-with-prorated-refund",
    "annual-refund-outside-window",
    "upgrade-after-a-question",
)

#: What each operation's span must look like under the pinned conventions:
#: its kind, and the attributes the conventions require or this course relies on.
_SHAPES = {
    INVOKE_AGENT: ("INTERNAL", (OPERATION_NAME, PROVIDER_NAME)),
    CHAT: ("CLIENT", (OPERATION_NAME, PROVIDER_NAME, REQUEST_MODEL)),
    EXECUTE_TOOL: ("INTERNAL", (OPERATION_NAME, TOOL_NAME, TOOL_CALL_ID)),
}


def _describe(span: SpanRecord) -> str:
    return "'{}'".format(span.name)


def trace_is_one_tree_under_the_agent_span(outcome: Outcome) -> CheckResult:
    """One trace, one ``invoke_agent`` span at its root, every other span below it."""
    name = "trace is one tree under the agent span"
    trace = outcome.trace
    if not trace.spans:
        return CheckResult(name, False, "The Outcome has no trace: no span was recorded.")
    agents = trace.agent_spans
    if len(agents) != 1:
        return CheckResult(
            name, False, "Expected one invoke_agent span, found {}.".format(len(agents)))
    roots = trace.roots
    if [span.span_id for span in roots] != [agents[0].span_id]:
        return CheckResult(name, False, "The root spans are {}, not the invoke_agent span.".format(
            ", ".join(_describe(span) for span in roots)))
    trace_ids = {span.trace_id for span in trace.spans}
    if len(trace_ids) != 1:
        return CheckResult(name, False, "The spans belong to {} traces, not one.".format(
            len(trace_ids)))
    return CheckResult(name, True, "{} spans under {}.".format(
        len(trace.spans) - 1, _describe(agents[0])))


def spans_follow_the_pinned_conventions(outcome: Outcome) -> CheckResult:
    """Every GenAI span has the pinned schema, its operation's name, kind and attributes."""
    name = "spans follow GenAI conventions {}".format(SEMCONV_VERSION)
    wrong: List[str] = []
    for span in outcome.trace.spans:
        if span.schema_url != SCHEMA_URL:
            wrong.append("{} has schema {!r}, not {}".format(
                _describe(span), span.schema_url or None, SCHEMA_URL))
        shape = _SHAPES.get(span.operation or "")
        if shape is None:
            continue  # a span of the agent's own: the conventions say nothing about it
        kind, required = shape
        missing = [key for key in required if key not in span.attributes]
        if missing:
            wrong.append("{} lacks {}".format(_describe(span), ", ".join(missing)))
        if span.kind != kind:
            wrong.append("{} is {}, not {}".format(_describe(span), span.kind, kind))
        subject = {
            CHAT: span.attributes.get(REQUEST_MODEL),
            EXECUTE_TOOL: span.attributes.get(TOOL_NAME),
            INVOKE_AGENT: span.attributes.get(AGENT_NAME),
        }[span.operation or ""]
        expected = "{} {}".format(span.operation, subject) if subject else span.operation
        if span.name != expected:
            wrong.append("{} should be named '{}'".format(_describe(span), expected))
    if wrong:
        return CheckResult(name, False, "; ".join(wrong) + ".")
    return CheckResult(name, True, "{} spans checked.".format(len(outcome.trace.spans)))


def every_model_call_has_a_chat_span(outcome: Outcome) -> CheckResult:
    """One ``chat`` span per model call the agent made, each under the agent span."""
    name = "every model call has a chat span"
    chats = outcome.trace.chat_spans
    agents = {span.span_id for span in outcome.trace.agent_spans}
    if len(chats) != outcome.steps:
        return CheckResult(name, False, "{} model calls, but {} chat spans.".format(
            outcome.steps, len(chats)))
    stray = [_describe(span) for span in chats if span.parent_id not in agents]
    if stray:
        return CheckResult(name, False, "Not under the agent span: {}.".format(", ".join(stray)))
    return CheckResult(name, True, "{0} model calls, {0} chat spans.".format(outcome.steps))


def every_tool_call_has_a_tool_span(outcome: Outcome) -> CheckResult:
    """Each tool call the model asked for has an ``execute_tool`` span with its call id."""
    name = "every tool call has a tool span"
    asked = [
        call
        for message in outcome.transcript
        if message.role == ROLE_ASSISTANT
        for call in message.tool_calls
    ]
    traced = {
        (span.attributes.get(TOOL_NAME), span.attributes.get(TOOL_CALL_ID))
        for span in outcome.trace.tool_spans
    }
    untraced = [call for call in asked if (call.name, call.id) not in traced]
    if untraced:
        return CheckResult(name, False, "No execute_tool span for {}. Run tool calls "
                           "through env.act, which traces them.".format(
                               ", ".join("{} ({})".format(c.name, c.id) for c in untraced)))
    if not asked:
        return CheckResult(name, True, "The model asked for no tools.")
    return CheckResult(name, True, "{0} tool calls, {0} traced.".format(len(asked)))


def span_tokens_and_cost_match_the_outcome(outcome: Outcome) -> CheckResult:
    """The chat spans' tokens and cost add up to the Outcome's, and so does the agent span."""
    name = "span tokens and cost match the Outcome"
    usage = outcome.usage
    expected = (
        ("input tokens", USAGE_INPUT_TOKENS, usage.input_tokens),
        ("output tokens", USAGE_OUTPUT_TOKENS, usage.output_tokens),
        ("cache-read input tokens", USAGE_CACHE_READ_INPUT_TOKENS, usage.cached_input_tokens),
        ("cache-write input tokens", USAGE_CACHE_CREATION_INPUT_TOKENS,
         usage.cache_write_input_tokens),
        ("cost", COST_USD, outcome.cost_usd),
    )
    wrong: List[str] = []
    for where, spans in (("chat spans", outcome.trace.chat_spans),
                         ("agent span", outcome.trace.agent_spans)):
        for label, key, want in expected:
            got = sum(span.attributes.get(key, 0) for span in spans)
            if not math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12):
                wrong.append("{} on the {}: {} there, {} on the Outcome".format(
                    label, where, _number(got), _number(want)))
    if wrong:
        return CheckResult(name, False, "; ".join(wrong) + ".")
    return CheckResult(name, True, "{} in, {} out, ${:.6f}, on spans and Outcome alike.".format(
        usage.input_tokens, usage.output_tokens, outcome.cost_usd))


def _number(value: float) -> str:
    return str(value) if float(value).is_integer() else "{:.6f}".format(value)


def the_agent_span_names_the_agent(outcome: Outcome) -> CheckResult:
    """The agent says who it is (``env.describe_agent``), so a backend can find its traces."""
    name = "the agent span names the agent"
    agents = outcome.trace.agent_spans
    named = [span.attributes.get(AGENT_NAME) for span in agents if span.attributes.get(AGENT_NAME)]
    if agents and len(named) == len(agents):
        return CheckResult(name, True, "The agent is {!r}.".format(named[0]))
    return CheckResult(name, False, "No {} on the invoke_agent span: call "
                       "env.describe_agent(name) from your agent.".format(AGENT_NAME))


CHECKS = (
    trace_is_one_tree_under_the_agent_span,
    spans_follow_the_pinned_conventions,
    every_model_call_has_a_chat_span,
    every_tool_call_has_a_tool_span,
    span_tokens_and_cost_match_the_outcome,
    the_agent_span_names_the_agent,
)
