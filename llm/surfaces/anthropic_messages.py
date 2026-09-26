"""Anthropic Messages API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..types import (
    STOP_END_TURN,
    STOP_MAX_TOKENS,
    STOP_OTHER,
    STOP_REFUSAL,
    STOP_STOP_SEQUENCE,
    STOP_TOOL_USE,
    Message,
    ProviderRequest,
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ToolCall,
    ToolSpec,
    Usage,
)

PATH = "/v1/messages"
DEFAULT_BASE_URL = "https://api.anthropic.com"

_STOP_REASONS = {
    "end_turn": STOP_END_TURN,
    "tool_use": STOP_TOOL_USE,
    "max_tokens": STOP_MAX_TOKENS,
    "stop_sequence": STOP_STOP_SEQUENCE,
    "refusal": STOP_REFUSAL,
}


def _message(message: Message) -> Dict[str, Any]:
    if message.role == ROLE_TOOL:
        # Anthropic carries tool results on a user turn.
        results = [
            {
                "type": "tool_result",
                "tool_use_id": result.call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in message.tool_results
        ]
        return {"role": "user", "content": results}

    content: List[Dict[str, Any]] = []
    if message.text:
        content.append({"type": "text", "text": message.text})
    for call in message.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": dict(call.arguments),
            }
        )
    role = "assistant" if message.role == ROLE_ASSISTANT else "user"
    return {"role": role, "content": content}


def build_request(
    spec,
    *,
    messages: Sequence[Message],
    system: Optional[str],
    tools: Sequence[ToolSpec],
    plan,
) -> ProviderRequest:
    body: Dict[str, Any] = {
        "model": spec.model_id,
        "max_tokens": plan.required_max_output_tokens(spec),
        "messages": [_message(m) for m in messages],
    }
    if system:
        body["system"] = system
    if tools:
        body["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            }
            for tool in tools
        ]
    if plan.effort is not None:
        body["output_config"] = {"effort": plan.effort}
    if plan.temperature is not None:
        body["temperature"] = plan.temperature
    return ProviderRequest(
        surface=spec.api_surface,
        model=spec.model_id,
        path=PATH,
        body=body,
        base_url=spec.base_url or DEFAULT_BASE_URL,
    )


def parse_response(spec, payload: Dict[str, Any]):
    text_parts: List[str] = []
    calls: List[ToolCall] = []
    for block in payload.get("content") or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input") or {},
                )
            )
    raw_usage = payload.get("usage") or {}
    usage = Usage(
        input_tokens=int(raw_usage.get("input_tokens") or 0),
        output_tokens=int(raw_usage.get("output_tokens") or 0),
        cached_input_tokens=int(raw_usage.get("cache_read_input_tokens") or 0),
    )
    stop = _STOP_REASONS.get(payload.get("stop_reason"), STOP_OTHER)
    return "".join(text_parts), tuple(calls), usage, stop
