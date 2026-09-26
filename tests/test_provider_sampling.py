"""Seam 2 Checks: the outgoing request, for every model in the registry.

The spec tests the provider layer on the request it would have sent, using a
stub transport and never the network:

    "For each registry model it asserts that unsupported options are dropped or
     translated (no temperature where rejected ...)"

Spec: docs/specs/graded-attacked-budgeted.md ("Testing Decisions").
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from provider import ApiSurface, load_registry  # noqa: E402
from provider.call import CallOptions, complete  # noqa: E402
from provider.testing import StubTransport  # noqa: E402
from provider.types import Message  # noqa: E402

REGISTRY = load_registry()
ALL_MODELS = REGISTRY.models

#: Where each surface puts a sampling temperature when it accepts one.
TEMPERATURE_PATH = {
    ApiSurface.ANTHROPIC_MESSAGES: ("temperature",),
    ApiSurface.OPENAI_RESPONSES: ("temperature",),
    ApiSurface.OPENAI_CHAT_COMPLETIONS: ("temperature",),
    ApiSurface.GOOGLE_GEMINI: ("generationConfig", "temperature"),
}


def _dig(body: dict, path):
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
        messages=[Message.user("Why was my refund declined?")],
        options=CallOptions(**option_kwargs),
        transport=transport,
        registry=REGISTRY,
    )
    return transport.last_request, response


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.model_id)
def test_temperature_is_never_sent_where_it_is_rejected(spec):
    request, response = _ask(spec, temperature=0.7)

    if spec.accepts_sampling_params:
        assert _dig(request.body, TEMPERATURE_PATH[spec.api_surface]) == 0.7
        assert not [a for a in response.adjustments if a.option == "temperature"]
    else:
        assert _dig(request.body, TEMPERATURE_PATH[spec.api_surface]) is None
        # Nothing anywhere in the body carries the value under another name.
        assert "temperature" not in _flatten_keys(request.body)
        dropped = [a for a in response.adjustments if a.option == "temperature"]
        assert len(dropped) == 1
        assert dropped[0].action == "dropped"
        # Story 14: the Reader can see *why*.
        assert dropped[0].reason


def _flatten_keys(node) -> set:
    keys = set()
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(key)
            keys |= _flatten_keys(value)
    elif isinstance(node, list):
        for item in node:
            keys |= _flatten_keys(item)
    return keys


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.model_id)
def test_no_temperature_is_sent_when_the_caller_asks_for_none(spec):
    request, response = _ask(spec)
    assert "temperature" not in _flatten_keys(request.body)
    assert not [a for a in response.adjustments if a.option == "temperature"]


@pytest.mark.parametrize("spec", ALL_MODELS, ids=lambda s: s.model_id)
def test_the_request_names_the_registry_model_and_its_surface(spec):
    request, _ = _ask(spec)
    assert request.model == spec.model_id
    assert request.surface == spec.api_surface


def test_an_unregistered_model_is_refused_before_any_request_is_built():
    transport = StubTransport()
    with pytest.raises(KeyError):
        complete(
            model="gpt-3.5-turbo",  # historical-exception: retired ID on purpose
            messages=[Message.user("hi")],
            transport=transport,
            registry=REGISTRY,
        )
    assert transport.requests == []
