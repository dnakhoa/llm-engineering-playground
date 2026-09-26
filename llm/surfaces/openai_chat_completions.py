"""OpenAI-compatible Chat Completions: DeepSeek, xAI and local servers."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ..types import (
    STOP_END_TURN,
    STOP_MAX_TOKENS,
    STOP_OTHER,
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

PATH = "/chat/completions"

_FINISH_REASONS = {
    "stop": STOP_END_TURN,
    "tool_calls": STOP_TOOL_USE,
    "function_call": STOP_TOOL_USE,
    "length": STOP_MAX_TOKENS,
    "content_filter": STOP_REFUSAL,
}


def _messages(messages: Sequence[Message]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for message in messages:
        if message.role == ROLE_TOOL:
            for result in message.tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.content,
                    }
                )
            continue
        if message.role == ROLE_ASSISTANT:
            entry: Dict[str, Any] = {"role": "assistant", "content": message.text}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                dict(call.arguments), sort_keys=True
                            ),
                        },
                    }
                    for call in message.tool_calls
                ]
            out.append(entry)
            continue
        out.append({"role": "user", "content": message.text or ""})
    return out


def build_request(
    spec,
    *,
    messages: Sequence[Message],
    system: Optional[str],
    tools: Sequence[ToolSpec],
    plan,
) -> ProviderRequest:
    wire_messages: List[Dict[str, Any]] = []
    if system:
        wire_messages.append({"role": "system", "content": system})
    wire_messages.extend(_messages(messages))

    body: Dict[str, Any] = {"model": spec.model_id, "messages": wire_messages}
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.input_schema),
                },
            }
            for tool in tools
        ]
    if plan.effort is not None:
        body["reasoning_effort"] = plan.effort
    if plan.max_output_tokens is not None:
        body["max_tokens"] = plan.max_output_tokens
    if plan.temperature is not None:
        body["temperature"] = plan.temperature
    return ProviderRequest(
        surface=spec.api_surface,
        model=spec.model_id,
        path=PATH,
        body=body,
        base_url=spec.base_url,
    )


def parse_response(spec, payload: Dict[str, Any]):
    choices = payload.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""

    calls: List[ToolCall] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args)
        except (TypeError, ValueError):
            arguments = {"__unparsed__": raw_args}
        calls.append(
            ToolCall(id=call.get("id", ""), name=function.get("name", ""), arguments=arguments)
        )

    raw_usage = payload.get("usage") or {}
    details = raw_usage.get("prompt_tokens_details") or {}
    usage = Usage(
        input_tokens=int(raw_usage.get("prompt_tokens") or 0),
        output_tokens=int(raw_usage.get("completion_tokens") or 0),
        # DeepSeek reports cache hits in its own field rather than the details.
        cached_input_tokens=int(
            details.get("cached_tokens") or raw_usage.get("prompt_cache_hit_tokens") or 0
        ),
    )
    stop = _FINISH_REASONS.get(choice.get("finish_reason"), STOP_OTHER)
    return text, tuple(calls), usage, stop
