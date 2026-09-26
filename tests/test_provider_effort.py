"""Seam 2 Checks: effort is mapped to each provider's own control.

    "... and that effort is mapped to that provider's control."

Ticket: docs/tickets/graded-attacked-budgeted/01-model-registry-and-provider-layer.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from provider import ApiSurface, load_registry  # noqa: E402
from provider.call import CallOptions, complete  # noqa: E402
from provider.testing import StubTransport  # noqa: E402
from provider.types import ACTION_DROPPED, ACTION_TRANSLATED, Message  # noqa: E402

REGISTRY = load_registry()
ALL_MODELS = REGISTRY.models

#: Each provider spells "how hard should it think" differently.
EFFORT_PATH = {
    ApiSurface.ANTHROPIC_MESSAGES: ("output_config", "effort"),
    ApiSurface.OPENAI_RESPONSES: ("reasoning", "effort"),
    ApiSurface.OPENAI_CHAT_COMPLETIONS: ("reasoning_effort",),
    ApiSurface.GOOGLE_GEMINI: ("generationConfig", "thinkingConfig", "thinkingLevel"),
}


def _dig(body, path):
    node = body
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _ask(spec, **option_kwargs):
    transport = StubTransport()
    response = complete(
        model=spec.model_id,
        messages=[Message.user("Refund my last invoice.")],
        options=CallOptions(**option_kwargs),
        transport=transport,
        registry=REGISTRY,
    )
    return transport.last_request, response


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.model_id)
@pytest.mark.parametrize("requested", ["low", "medium", "high", "max"])
def test_effort_reaches_the_providers_control_or_is_reported(spec, requested):
    request, response = _ask(spec, effort=requested)
    sent = _dig(request.body, EFFORT_PATH[spec.api_surface])
    notes = [a for a in response.adjustments if a.option == "effort"]

    if not spec.effort_levels:
        assert sent is None
        assert [a.action for a in notes] == [ACTION_DROPPED]
        return

    # Whatever is sent must be a level this model actually has.
    assert sent in spec.effort_levels, (spec.model_id, requested, sent)
    if requested in spec.effort_levels:
        assert sent == requested
        assert notes == []
    else:
        assert [a.action for a in notes] == [ACTION_TRANSLATED]
        assert notes[0].sent_as == sent
        assert notes[0].requested == requested


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.model_id)
def test_no_effort_control_is_sent_when_none_is_asked_for(spec):
    request, response = _ask(spec)
    assert _dig(request.body, EFFORT_PATH[spec.api_surface]) is None
    assert [a for a in response.adjustments if a.option == "effort"] == []


def test_a_level_a_model_lacks_snaps_to_its_nearest_neighbour():
    # DeepSeek exposes low / high / max: "medium" has to become one of them,
    # and a translation never silently spends more than was asked for.
    request, response = _ask(REGISTRY.get("deepseek-flash"), effort="medium")
    assert request.body["reasoning_effort"] == "low"
    assert response.adjustments[0].action == ACTION_TRANSLATED

    # Gemini 3.8 Flash has no "max"; the closest it has is "high".
    request, _ = _ask(REGISTRY.get("gemini-3.8-flash"), effort="max")
    assert request.body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"


def test_an_unknown_effort_level_is_a_programming_error():
    with pytest.raises(ValueError):
        _ask(REGISTRY.get("claude-sonnet-5"), effort="ludicrous")


def test_max_output_tokens_is_clamped_to_the_models_ceiling():
    spec = REGISTRY.get("claude-haiku-4-5")  # 64k output ceiling
    request, response = _ask(spec, max_output_tokens=200_000)
    assert request.body["max_tokens"] == spec.max_output_tokens
    clamped = [a for a in response.adjustments if a.option == "max_output_tokens"]
    assert [a.action for a in clamped] == [ACTION_TRANSLATED]
