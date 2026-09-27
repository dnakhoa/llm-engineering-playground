"""The trace of one Case: OpenTelemetry spans that follow the GenAI conventions.

The Case runner opens one span per Case and the Environment one per model call
and per tool call, so any agent that works through ``env`` is traced without a
line of tracing code of its own:

    invoke_agent acme-support     gen_ai.operation.name = invoke_agent  (root)
    ├── chat claude-sonnet-5      gen_ai.operation.name = chat          tokens, cost
    ├── execute_tool look_up_account                                    call id
    ├── chat claude-sonnet-5
    └── ...

The GenAI conventions are still in Development, so the attribute names can
change between releases. They are pinned here, to OpenTelemetry semantic
conventions 1.41.0 (``SEMCONV_VERSION``), a release of the main
semantic-conventions repository that defines every ``gen_ai.*`` name used
below. Later releases mark those names deprecated there, because the GenAI
conventions moved to their own repository. Every span says which version it
follows, in its instrumentation scope's schema URL.

The conventions define tokens but not money, so cost goes on ``acme.cost_usd``,
in our own namespace rather than ``gen_ai.*``, which is theirs.

A finished trace comes back on the Outcome as a ``Trace``: plain records, so a
Check reads it without the SDK. The same spans go, as they end, to any
exporter passed in, which is how a trace reaches Jaeger, Honeycomb, Langfuse
or any other OpenTelemetry backend.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

try:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer
except ImportError as error:  # pragma: no cover - depends on what is installed
    raise ImportError(
        "The Case runner traces every run with OpenTelemetry, which is not "
        "installed: pip install opentelemetry-sdk (it is in requirements.txt)."
    ) from error

from llm.types import Response, ToolCall, ToolResult

#: The OpenTelemetry semantic conventions release every span follows.
SEMCONV_VERSION = "1.41.0"
SCHEMA_URL = "https://opentelemetry.io/schemas/{}".format(SEMCONV_VERSION)

#: The instrumentation scope: who emitted the spans.
INSTRUMENTATION_NAME = "acme-notes.case-runner"
SERVICE_NAME = "acme-notes-support-agent"

# ── Attribute names, as semantic conventions 1.41.0 spell them ────────────────

OPERATION_NAME = "gen_ai.operation.name"
PROVIDER_NAME = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_ID = "gen_ai.response.id"
RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
CONVERSATION_ID = "gen_ai.conversation.id"
AGENT_NAME = "gen_ai.agent.name"
AGENT_VERSION = "gen_ai.agent.version"
AGENT_DESCRIPTION = "gen_ai.agent.description"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_TYPE = "gen_ai.tool.type"
TOOL_DESCRIPTION = "gen_ai.tool.description"
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
ERROR_TYPE = "error.type"

#: Not in the conventions: what the tokens cost, in US dollars, at registry prices.
COST_USD = "acme.cost_usd"

INVOKE_AGENT = "invoke_agent"
CHAT = "chat"
EXECUTE_TOOL = "execute_tool"

#: The registry's provider names as ``gen_ai.provider.name`` spells them. A
#: provider the conventions do not list keeps the registry's name, which they allow.
PROVIDER_NAMES = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "gcp.gemini",
    "deepseek": "deepseek",
    "xai": "x_ai",
}

#: The provider layer's stop reasons as finish reasons.
_FINISH_REASONS = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}


def provider_name(registry_provider: str) -> str:
    return PROVIDER_NAMES.get(registry_provider, registry_provider)


# ── What comes back on the Outcome ────────────────────────────────────────────


@dataclass(frozen=True)
class SpanRecord:
    """One finished span, as plain data."""

    name: str
    kind: str  # "INTERNAL", "CLIENT", ...
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    attributes: Mapping[str, Any]
    status: str  # "UNSET", "OK" or "ERROR"
    schema_url: str
    start_ns: int = 0
    end_ns: int = 0

    @property
    def operation(self) -> Optional[str]:
        """``gen_ai.operation.name``, or ``None`` on a span that is not GenAI."""
        return self.attributes.get(OPERATION_NAME)


@dataclass(frozen=True)
class Trace:
    """Every span one Case run produced, in the order they started."""

    spans: Tuple[SpanRecord, ...] = ()
    semconv_version: str = SEMCONV_VERSION

    def operation(self, name: str) -> Tuple[SpanRecord, ...]:
        """The spans whose ``gen_ai.operation.name`` is ``name``."""
        return tuple(span for span in self.spans if span.operation == name)

    @property
    def agent_spans(self) -> Tuple[SpanRecord, ...]:
        return self.operation(INVOKE_AGENT)

    @property
    def chat_spans(self) -> Tuple[SpanRecord, ...]:
        return self.operation(CHAT)

    @property
    def tool_spans(self) -> Tuple[SpanRecord, ...]:
        return self.operation(EXECUTE_TOOL)

    @property
    def roots(self) -> Tuple[SpanRecord, ...]:
        """Spans with no parent in this trace."""
        ids = {span.span_id for span in self.spans}
        return tuple(span for span in self.spans if span.parent_id not in ids)


def _hex(value: int, width: int) -> str:
    return format(value, "0{}x".format(width))


def _record(span: ReadableSpan) -> SpanRecord:
    context = span.get_span_context()
    scope = span.instrumentation_scope
    return SpanRecord(
        name=span.name,
        kind=span.kind.name,
        trace_id=_hex(context.trace_id, 32) if context else "",
        span_id=_hex(context.span_id, 16) if context else "",
        parent_id=_hex(span.parent.span_id, 16) if span.parent else None,
        attributes=dict(span.attributes or {}),
        status=span.status.status_code.name,
        schema_url=(scope.schema_url or "") if scope else "",
        start_ns=span.start_time or 0,
        end_ns=span.end_time or 0,
    )


# ── Emitting the spans ────────────────────────────────────────────────────────


def _set_error(span: Span, error_type: str, description: str) -> None:
    span.set_attribute(ERROR_TYPE, error_type)
    span.set_status(Status(StatusCode.ERROR, description))


@dataclass
class CaseTracer:
    """Traces one Case: keeps its spans in memory and sends them to ``exporters``.

    Each Case gets its own tracer provider, so its trace holds its spans and no
    other Case's. An exporter is only ever flushed, never shut down, so one
    exporter can serve every Case in a run.
    """

    exporters: Sequence[SpanExporter] = ()
    _memory: InMemorySpanExporter = field(default_factory=InMemorySpanExporter, init=False)

    def __post_init__(self) -> None:
        service = os.environ.get("OTEL_SERVICE_NAME") or SERVICE_NAME
        self._provider = TracerProvider(
            resource=Resource.create({"service.name": service}),
            shutdown_on_exit=False,
        )
        self._provider.add_span_processor(SimpleSpanProcessor(self._memory))
        for exporter in self.exporters:
            self._provider.add_span_processor(SimpleSpanProcessor(exporter))
        self.tracer: Tracer = self._provider.get_tracer(
            INSTRUMENTATION_NAME, schema_url=SCHEMA_URL
        )

    @contextmanager
    def agent(self, *, case_id: str, model: str, provider: str) -> Iterator[Span]:
        """The root span: one agent working one Case, every turn of it."""
        with self.tracer.start_as_current_span(
            INVOKE_AGENT,
            kind=SpanKind.INTERNAL,
            attributes={
                OPERATION_NAME: INVOKE_AGENT,
                PROVIDER_NAME: provider_name(provider),
                REQUEST_MODEL: model,
                CONVERSATION_ID: case_id,
            },
        ) as span:
            yield span

    @contextmanager
    def chat(self, *, model: str, provider: str) -> Iterator[Span]:
        """One model call. ``record_response`` puts its tokens and cost on it."""
        with self.tracer.start_as_current_span(
            "{} {}".format(CHAT, model),
            kind=SpanKind.CLIENT,
            attributes={
                OPERATION_NAME: CHAT,
                PROVIDER_NAME: provider_name(provider),
                REQUEST_MODEL: model,
            },
            record_exception=True,
            set_status_on_exception=False,
        ) as span:
            try:
                yield span
            except Exception as error:
                _set_error(span, type(error).__name__, str(error))
                raise

    @contextmanager
    def tool(self, call: ToolCall, description: Optional[str]) -> Iterator[Span]:
        """One tool call. ``record_tool_result`` marks a refusal as an error."""
        attributes = {
            OPERATION_NAME: EXECUTE_TOOL,
            TOOL_NAME: call.name,
            TOOL_CALL_ID: call.id,
            TOOL_TYPE: "function",
        }
        if description:
            attributes[TOOL_DESCRIPTION] = description
        with self.tracer.start_as_current_span(
            "{} {}".format(EXECUTE_TOOL, call.name),
            kind=SpanKind.INTERNAL,
            attributes=attributes,
            record_exception=True,
            set_status_on_exception=False,
        ) as span:
            try:
                yield span
            except Exception as error:
                _set_error(span, type(error).__name__, str(error))
                raise

    def finish(self) -> Trace:
        """Flush every exporter and return this Case's spans, in start order."""
        self._provider.force_flush()
        spans = sorted(self._memory.get_finished_spans(), key=lambda s: s.start_time or 0)
        return Trace(spans=tuple(_record(span) for span in spans))


def record_response(span: Span, response: Response) -> None:
    """A model call's answer on its chat span: ids, finish reason, tokens, cost."""
    raw = response.raw or {}
    response_id = raw.get("id") if isinstance(raw, Mapping) else None
    if isinstance(response_id, str):
        span.set_attribute(RESPONSE_ID, response_id)
    reported_model = raw.get("model") if isinstance(raw, Mapping) else None
    span.set_attribute(
        RESPONSE_MODEL, reported_model if isinstance(reported_model, str) else response.model
    )
    span.set_attribute(
        RESPONSE_FINISH_REASONS,
        [_FINISH_REASONS.get(response.stop_reason, response.stop_reason)],
    )
    record_usage(
        span,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=response.usage.cached_input_tokens,
        cache_creation_tokens=response.usage.cache_write_input_tokens,
        cost_usd=response.cost_usd,
    )


def record_usage(
    span: Span,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    cost_usd: float,
) -> None:
    """Tokens the way the conventions count them: input includes cached tokens."""
    span.set_attribute(USAGE_INPUT_TOKENS, input_tokens)
    span.set_attribute(USAGE_OUTPUT_TOKENS, output_tokens)
    span.set_attribute(USAGE_CACHE_READ_INPUT_TOKENS, cache_read_tokens)
    span.set_attribute(USAGE_CACHE_CREATION_INPUT_TOKENS, cache_creation_tokens)
    span.set_attribute(COST_USD, cost_usd)


def record_tool_result(span: Span, result: ToolResult) -> None:
    """A refused or failed tool call is an error on its span, with the reason."""
    if result.is_error:
        _set_error(span, "tool_error", result.content)


def record_stop(span: Span, reason: str, description: str) -> None:
    """The agent did not finish: the step limit or the Spend Cap stopped it."""
    _set_error(span, reason, description)


def describe_agent(
    span: Span,
    name: str,
    *,
    version: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
    """Name the agent on its span, which the conventions then call ``invoke_agent {name}``."""
    span.set_attribute(AGENT_NAME, name)
    span.update_name("{} {}".format(INVOKE_AGENT, name))
    if version:
        span.set_attribute(AGENT_VERSION, version)
    if description:
        span.set_attribute(AGENT_DESCRIPTION, description)
