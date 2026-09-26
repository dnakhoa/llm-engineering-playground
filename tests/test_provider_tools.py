"""Seam 2 Checks: normalized tools round-trip into each provider's shape.

    "Normalized tools round-trip into each provider's tool shape, and tool
     calls come back in normalized form."
    "Tool calls for current OpenAI models go through the Responses API,
     verified by an outgoing-request test."

Ticket: docs/tickets/graded-attacked-budgeted/01-model-registry-and-provider-layer.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from provider import ApiSurface, load_registry  # noqa: E402
from provider.call import complete  # noqa: E402
from provider.surfaces import openai_responses  # noqa: E402
from provider.testing import StubTransport  # noqa: E402
from provider.types import (  # noqa: E402
    STOP_TOOL_USE,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
)

REGISTRY = load_registry()
ALL_MODELS = REGISTRY.models

ISSUE_REFUND = ToolSpec(
    name="issue_refund",
    description="Refund an invoice on the customer's own account.",
    input_schema={
        "type": "object",
        "properties": {
            "invoice_id": {"type": "string"},
            "amount_usd": {"type": "number"},
        },
        "required": ["invoice_id", "amount_usd"],
        "additionalProperties": False,
    },
)

#: Where each surface keeps the declared tools, and what it calls the schema.
TOOL_SHAPE = {
    ApiSurface.ANTHROPIC_MESSAGES: (
        lambda body: body["tools"],
        "name",
        "description",
        "input_schema",
    ),
    ApiSurface.OPENAI_RESPONSES: (
        lambda body: body["tools"],
        "name",
        "description",
        "parameters",
    ),
    ApiSurface.OPENAI_CHAT_COMPLETIONS: (
        lambda body: [t["function"] for t in body["tools"]],
        "name",
        "description",
        "parameters",
    ),
    ApiSurface.GOOGLE_GEMINI: (
        lambda body: body["tools"][0]["functionDeclarations"],
        "name",
        "description",
        "parameters",
    ),
}


def _ask(spec, messages, transport=None, tools=(ISSUE_REFUND,)):
    transport = transport or StubTransport()
    response = complete(
        model=spec.model_id,
        messages=messages,
        tools=tools,
        system="You are Acme Notes support.",
        transport=transport,
        registry=REGISTRY,
    )
    return transport.last_request, response


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.model_id)
def test_a_normalized_tool_lands_in_the_providers_tool_shape(spec):
    request, _ = _ask(spec, [Message.user("Refund invoice INV-7.")])
    read, name_key, desc_key, schema_key = TOOL_SHAPE[spec.api_surface]
    declared = read(request.body)
    assert len(declared) == 1
    assert declared[0][name_key] == ISSUE_REFUND.name
    assert declared[0][desc_key] == ISSUE_REFUND.description
    assert declared[0][schema_key] == ISSUE_REFUND.input_schema


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.model_id)
def test_a_tool_call_and_its_result_survive_the_round_trip(spec):
    """An assistant turn with a call, then the result, then the next request."""
    call = ToolCall(
        id="call_abc123",
        name="issue_refund",
        arguments={"invoice_id": "INV-7", "amount_usd": 20},
    )
    transcript = [
        Message.user("Refund invoice INV-7."),
        Message.assistant(text=None, tool_calls=[call]),
        Message.tool(
            [
                ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content='{"refunded_usd": 20, "invoice_id": "INV-7"}',
                )
            ]
        ),
    ]
    request, _ = _ask(spec, transcript)
    body = json.dumps(request.body)
    # The call's identity, its arguments and the result all reach the provider.
    assert "issue_refund" in body
    assert "INV-7" in body
    assert "refunded_usd" in body
    if spec.api_surface is not ApiSurface.GOOGLE_GEMINI:
        # Gemini matches a tool response by name, not by id (see the surface).
        assert call.id in body


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.model_id)
def test_a_tool_call_comes_back_normalized(spec):
    transport = StubTransport([_tool_call_payload(spec)])
    _, response = _ask(spec, [Message.user("Refund invoice INV-7.")], transport)

    assert response.stop_reason == STOP_TOOL_USE
    assert len(response.tool_calls) == 1
    returned = response.tool_calls[0]
    assert returned.name == "issue_refund"
    assert returned.arguments == {"invoice_id": "INV-7", "amount_usd": 20}
    assert returned.id  # every surface yields something usable as a call id


def _tool_call_payload(spec):
    arguments = {"invoice_id": "INV-7", "amount_usd": 20}
    if spec.api_surface is ApiSurface.ANTHROPIC_MESSAGES:
        return {
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "issue_refund", "input": arguments}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 120, "output_tokens": 30},
        }
    if spec.api_surface is ApiSurface.OPENAI_RESPONSES:
        return {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "issue_refund",
                    "arguments": json.dumps(arguments),
                }
            ],
            "usage": {"input_tokens": 120, "output_tokens": 30},
        }
    if spec.api_surface is ApiSurface.OPENAI_CHAT_COMPLETIONS:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "issue_refund",
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        }
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "issue_refund", "args": arguments}}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 30},
    }


OPENAI_MODELS = [s for s in ALL_MODELS if s.provider == "openai"]


@pytest.mark.parametrize("spec", OPENAI_MODELS, ids=lambda s: s.model_id)
def test_openai_tool_calls_go_through_the_responses_api(spec):
    """Chat Completions only does function calling at effort "none", so never use it."""
    request, _ = _ask(spec, [Message.user("Refund invoice INV-7.")])
    assert request.surface is ApiSurface.OPENAI_RESPONSES
    assert request.path == openai_responses.PATH == "/v1/responses"
    assert "messages" not in request.body  # not a Chat Completions body
    assert request.body["tools"][0]["type"] == "function"


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.model_id)
def test_usage_and_cost_come_back_priced_from_the_registry(spec):
    transport = StubTransport([_tool_call_payload(spec)])
    _, response = _ask(spec, [Message.user("Refund invoice INV-7.")], transport)
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 30
    assert response.cost_usd == pytest.approx(
        spec.cost_usd(input_tokens=120, output_tokens=30)
    )
