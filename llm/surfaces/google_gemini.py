"""Google Gemini generateContent.

Gemini identifies a tool response by the function's *name*, not by a call id,
and has no id on the call it sends back. The normalized ToolResult carries both
``call_id`` and ``name`` so a transcript survives a move to or from Gemini; ids
for incoming calls are synthesized from the name and position.
"""
from __future__ import annotations

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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

_FINISH_REASONS = {
    "STOP": STOP_END_TURN,
    "MAX_TOKENS": STOP_MAX_TOKENS,
    "SAFETY": STOP_REFUSAL,
    "PROHIBITED_CONTENT": STOP_REFUSAL,
    "BLOCKLIST": STOP_REFUSAL,
}


def path_for(model_id: str) -> str:
    return "/v1beta/models/{}:generateContent".format(model_id)


def _content(message: Message) -> Dict[str, Any]:
    if message.role == ROLE_TOOL:
        # Gemini carries function responses on a user turn, keyed by name.
        responses = [
            {
                "functionResponse": {
                    "name": result.name,
                    "response": {
                        "error" if result.is_error else "output": result.content
                    },
                }
            }
            for result in message.tool_results
        ]
        return {"role": "user", "parts": responses}

    parts: List[Dict[str, Any]] = []
    if message.text:
        parts.append({"text": message.text})
    for call in message.tool_calls:
        parts.append({"functionCall": {"name": call.name, "args": dict(call.arguments)}})
    role = "model" if message.role == ROLE_ASSISTANT else "user"
    return {"role": role, "parts": parts}


def build_request(
    spec,
    *,
    messages: Sequence[Message],
    system: Optional[str],
    tools: Sequence[ToolSpec],
    plan,
) -> ProviderRequest:
    body: Dict[str, Any] = {"contents": [_content(m) for m in messages]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.input_schema),
                    }
                    for tool in tools
                ]
            }
        ]

    generation_config: Dict[str, Any] = {}
    if plan.max_output_tokens is not None:
        generation_config["maxOutputTokens"] = plan.max_output_tokens
    if plan.temperature is not None:
        generation_config["temperature"] = plan.temperature
    if plan.effort is not None:
        generation_config["thinkingConfig"] = {"thinkingLevel": plan.effort}
    if generation_config:
        body["generationConfig"] = generation_config

    return ProviderRequest(
        surface=spec.api_surface,
        model=spec.model_id,
        path=path_for(spec.model_id),
        body=body,
        base_url=spec.base_url or DEFAULT_BASE_URL,
    )


def parse_response(spec, payload: Dict[str, Any]):
    candidates = payload.get("candidates") or [{}]
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []

    text_parts: List[str] = []
    calls: List[ToolCall] = []
    for index, part in enumerate(parts):
        if "text" in part:
            text_parts.append(part.get("text") or "")
        elif "functionCall" in part:
            function_call = part["functionCall"]
            name = function_call.get("name", "")
            calls.append(
                ToolCall(
                    id="{}-{}".format(name, index),
                    name=name,
                    arguments=function_call.get("args") or {},
                )
            )

    metadata = payload.get("usageMetadata") or {}
    usage = Usage(
        input_tokens=int(metadata.get("promptTokenCount") or 0),
        output_tokens=int(metadata.get("candidatesTokenCount") or 0),
        cached_input_tokens=int(metadata.get("cachedContentTokenCount") or 0),
    )

    if calls:
        stop = STOP_TOOL_USE
    else:
        stop = _FINISH_REASONS.get(candidate.get("finishReason"), STOP_OTHER)
    return "".join(text_parts), tuple(calls), usage, stop
