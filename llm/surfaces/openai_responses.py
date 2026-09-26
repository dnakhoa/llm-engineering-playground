"""OpenAI Responses API.

Current OpenAI models need this surface for tool calls: Chat Completions only
supports function calling at ``reasoning_effort: "none"``. That is why the
registry pins the GPT-6 family to ``openai_responses`` rather than treating
OpenAI as just another OpenAI-compatible server.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ..types import (
    STOP_END_TURN,
    STOP_MAX_TOKENS,
    STOP_REFUSAL,
    STOP_TOOL_USE,
    Message,
    ProviderRequest,
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ToolCall,
    ToolSpec,
    Usage,
)

PATH = "/v1/responses"
DEFAULT_BASE_URL = "https://api.openai.com"


def _input_items(message: Message) -> List[Dict[str, Any]]:
    if message.role == ROLE_TOOL:
        return [
            {
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": result.content,
            }
            for result in message.tool_results
        ]

    items: List[Dict[str, Any]] = []
    if message.text:
        part_type = "output_text" if message.role == ROLE_ASSISTANT else "input_text"
        items.append(
            {
                "role": "assistant" if message.role == ROLE_ASSISTANT else "user",
                "content": [{"type": part_type, "text": message.text}],
            }
        )
    for call in message.tool_calls:
        items.append(
            {
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": json.dumps(dict(call.arguments), sort_keys=True),
            }
        )
    return items


def build_request(
    spec,
    *,
    messages: Sequence[Message],
    system: Optional[str],
    tools: Sequence[ToolSpec],
    plan,
) -> ProviderRequest:
    items: List[Dict[str, Any]] = []
    for message in messages:
        items.extend(_input_items(message))

    body: Dict[str, Any] = {"model": spec.model_id, "input": items}
    if system:
        body["instructions"] = system
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.input_schema),
            }
            for tool in tools
        ]
    if plan.effort is not None:
        body["reasoning"] = {"effort": plan.effort}
    if plan.max_output_tokens is not None:
        body["max_output_tokens"] = plan.max_output_tokens
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
    for item in payload.get("output") or []:
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    text_parts.append(part.get("text", ""))
        elif item_type == "function_call":
            raw_args = item.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args)
            except (TypeError, ValueError):
                arguments = {"__unparsed__": raw_args}
            calls.append(
                ToolCall(
                    id=item.get("call_id") or item.get("id", ""),
                    name=item.get("name", ""),
                    arguments=arguments,
                )
            )

    raw_usage = payload.get("usage") or {}
    details = raw_usage.get("input_tokens_details") or {}
    usage = Usage(
        input_tokens=int(raw_usage.get("input_tokens") or 0),
        output_tokens=int(raw_usage.get("output_tokens") or 0),
        cached_input_tokens=int(details.get("cached_tokens") or 0),
    )

    if calls:
        stop = STOP_TOOL_USE
    elif payload.get("status") == "incomplete":
        reason = (payload.get("incomplete_details") or {}).get("reason")
        stop = STOP_MAX_TOKENS if reason == "max_output_tokens" else STOP_REFUSAL
    else:
        stop = STOP_END_TURN
    return "".join(text_parts), tuple(calls), usage, stop
