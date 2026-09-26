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

from llm import ApiSurface, load_registry  # noqa: E402
from llm.call import complete  # noqa: E402
from llm.surfaces import openai_responses  # noqa: E402
from llm.testing import StubTransport  # noqa: E402
from llm.types import (  # noqa: E402
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


def _check_anthropic_round_trip(body, call, result_content):
    messages = body["messages"]
    # system rides beside the transcript on this surface.
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == [
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": dict(call.arguments),
        }
    ]
    assert messages[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": result_content,
            "is_error": False,
        }
    ]


def _check_responses_round_trip(body, call, result_content):
    items = body["input"]
    assert len(items) == 3
    assert items[0]["role"] == "user"
    assert "type" not in items[0]  # the user turn is a plain message, not a call
    function_call = items[1]
    assert function_call["type"] == "function_call"
    assert function_call["call_id"] == call.id
    assert function_call["name"] == call.name
    assert json.loads(function_call["arguments"]) == dict(call.arguments)
    output = items[2]
    assert output["type"] == "function_call_output"
    assert output["call_id"] == call.id  # linked back to the call above
    assert output["output"] == result_content


def _check_chat_completions_round_trip(body, call, result_content):
    messages = body["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"] == [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(dict(call.arguments), sort_keys=True),
            },
        }
    ]
    tool_message = messages[3]
    assert tool_message["tool_call_id"] == call.id
    assert tool_message["content"] == result_content


def _check_gemini_round_trip(body, call, result_content):
    contents = body["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"] == [
        {"functionCall": {"name": call.name, "args": dict(call.arguments)}}
    ]
    response_parts = contents[2]["parts"]
    assert len(response_parts) == 1
    # Gemini matches a tool response by the function's name, not by a call id.
    assert response_parts[0] == {
        "functionResponse": {
            "name": call.name,
            "response": {"output": result_content},
        }
    }


#: How each surface must re-encode a prior tool call and its result.
TRANSCRIPT_SHAPE = {
    ApiSurface.ANTHROPIC_MESSAGES: _check_anthropic_round_trip,
    ApiSurface.OPENAI_RESPONSES: _check_responses_round_trip,
    ApiSurface.OPENAI_CHAT_COMPLETIONS: _check_chat_completions_round_trip,
    ApiSurface.GOOGLE_GEMINI: _check_gemini_round_trip,
}


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.model_id)
def test_a_tool_call_and_its_result_survive_the_round_trip(spec):
    """An assistant turn with a call, then the result, then the next request.

    Asserts the typed blocks themselves — the assistant's tool-call block, the
    tool result at the role the surface expects, and the id (or, on Gemini, the
    name) that links the two. A substring check over the serialized body would
    pass on untyped text, so it is deliberately not used here.
    """
    call = ToolCall(
        id="call_abc123",
        name="issue_refund",
        arguments={"invoice_id": "INV-7", "amount_usd": 20},
    )
    result_content = '{"refunded_usd": 20, "invoice_id": "INV-7"}'
    transcript = [
        Message.user("Refund invoice INV-7."),
        Message.assistant(text=None, tool_calls=[call]),
        Message.tool(
            [ToolResult(call_id=call.id, name=call.name, content=result_content)]
        ),
    ]
    request, _ = _ask(spec, transcript)
    TRANSCRIPT_SHAPE[spec.api_surface](request.body, call, result_content)


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
