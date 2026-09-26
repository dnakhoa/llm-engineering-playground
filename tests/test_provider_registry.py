"""Registry Checks — the model registry is the one source of truth for model facts.

Spec: docs/specs/graded-attacked-budgeted.md ("Provider layer (seam 2)", story 50).
Ticket: docs/tickets/graded-attacked-budgeted/01-model-registry-and-provider-layer.md
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from provider import ApiSurface, load_registry  # noqa: E402

REGISTRY = load_registry()

KNOWN_EFFORT_LADDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def test_registry_is_not_empty():
    assert len(REGISTRY.models) >= 1


def test_model_ids_are_unique():
    ids = [spec.model_id for spec in REGISTRY.models]
    assert len(ids) == len(set(ids))


def test_every_surface_in_the_spec_is_represented():
    surfaces = {spec.api_surface for spec in REGISTRY.models}
    assert surfaces == {
        ApiSurface.ANTHROPIC_MESSAGES,
        ApiSurface.OPENAI_RESPONSES,
        ApiSurface.GOOGLE_GEMINI,
        ApiSurface.OPENAI_CHAT_COMPLETIONS,
    }


@pytest.mark.parametrize("spec", REGISTRY.models, ids=lambda s: s.model_id)
def test_entry_records_its_source_and_verification_date(spec):
    """Unverified models are left out, and every entry says where its facts came from."""
    assert spec.source.startswith("https://"), spec.model_id
    verified = dt.date.fromisoformat(spec.verified_on)
    assert verified <= dt.date.today()


@pytest.mark.parametrize("spec", REGISTRY.models, ids=lambda s: s.model_id)
def test_entry_carries_prices_and_a_context_window(spec):
    assert spec.input_price_per_mtok > 0
    assert spec.output_price_per_mtok > 0
    assert spec.context_window > 0


@pytest.mark.parametrize("spec", REGISTRY.models, ids=lambda s: s.model_id)
def test_effort_levels_come_from_the_known_ladder(spec):
    assert set(spec.effort_levels) <= set(KNOWN_EFFORT_LADDER), spec.model_id
    # Declared levels keep the ladder's order, so nearest-level mapping is well defined.
    positions = [KNOWN_EFFORT_LADDER.index(level) for level in spec.effort_levels]
    assert positions == sorted(positions), spec.model_id


@pytest.mark.parametrize("spec", REGISTRY.models, ids=lambda s: s.model_id)
def test_a_model_that_refuses_sampling_params_says_why(spec):
    """Story 14: the Reader is told why an option was dropped, not just that it was."""
    if not spec.accepts_sampling_params:
        assert spec.sampling_note, spec.model_id


def test_lookup_by_id_and_unknown_model_error():
    spec = REGISTRY.get("claude-sonnet-5")
    assert spec.provider == "anthropic"
    with pytest.raises(KeyError) as excinfo:
        REGISTRY.get("gpt-4o-mini")  # historical-exception: retired ID on purpose
    assert "gpt-4o-mini" in str(excinfo.value)  # historical-exception


def test_cost_is_computed_from_registry_prices():
    spec = REGISTRY.get("claude-sonnet-5")
    # $2.00 / MTok input, $10.00 / MTok output.
    assert spec.cost_usd(input_tokens=1_000_000, output_tokens=0) == pytest.approx(2.0)
    assert spec.cost_usd(input_tokens=0, output_tokens=500_000) == pytest.approx(5.0)
