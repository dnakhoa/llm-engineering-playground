"""The one call. Normalized in, normalized out, whichever model answers."""
from __future__ import annotations

from typing import Optional, Sequence

from .capabilities import RequestPlan, plan_request
from .registry import ApiSurface, ModelSpec, Registry, load_registry
from .surfaces import (
    anthropic_messages,
    google_gemini,
    openai_chat_completions,
    openai_responses,
)
from .types import CallOptions, Message, ProviderRequest, Response, ToolSpec

_SURFACES = {
    ApiSurface.ANTHROPIC_MESSAGES: anthropic_messages,
    ApiSurface.OPENAI_RESPONSES: openai_responses,
    ApiSurface.OPENAI_CHAT_COMPLETIONS: openai_chat_completions,
    ApiSurface.GOOGLE_GEMINI: google_gemini,
}

_DEFAULT_REGISTRY: Optional[Registry] = None


def _registry() -> Registry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = load_registry()
    return _DEFAULT_REGISTRY


def build_request(
    spec: ModelSpec,
    *,
    messages: Sequence[Message],
    system: Optional[str] = None,
    tools: Sequence[ToolSpec] = (),
    plan: Optional[RequestPlan] = None,
    options: Optional[CallOptions] = None,
) -> ProviderRequest:
    """What we would send. Exposed so Checks can look without sending."""
    plan = plan if plan is not None else plan_request(spec, options)
    return _SURFACES[spec.api_surface].build_request(
        spec, messages=messages, system=system, tools=tools, plan=plan
    )


def complete(
    *,
    model: str,
    messages: Sequence[Message],
    transport,
    tools: Sequence[ToolSpec] = (),
    system: Optional[str] = None,
    options: Optional[CallOptions] = None,
    registry: Optional[Registry] = None,
) -> Response:
    """Ask ``model`` to continue ``messages``, with ``tools`` available.

    Options the model cannot act on are dropped or translated first, and what
    happened to them comes back on ``Response.adjustments``.
    """
    spec = (registry or _registry()).get(model)
    plan = plan_request(spec, options)
    surface = _SURFACES[spec.api_surface]

    request = surface.build_request(
        spec, messages=messages, system=system, tools=tools, plan=plan
    )
    payload = transport.send(request)
    text, tool_calls, usage, stop_reason = surface.parse_response(spec, payload)

    return Response(
        model=spec.model_id,
        text=text,
        tool_calls=tool_calls,
        usage=usage,
        stop_reason=stop_reason,
        cost_usd=spec.cost_usd(
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
        ),
        adjustments=plan.adjustments,
        raw=payload,
    )
