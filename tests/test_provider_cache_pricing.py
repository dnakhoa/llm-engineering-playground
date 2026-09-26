"""Cached input is billed at the cache price, not the full input price.

Without this the Spend Cap and the Budgeted module overstate every cached run,
and prompt caching, the first cost lever the Reader pulls, looks like it does
nothing. Tested on the normalized Response and on the Outcome, with a stub
transport replying in each vendor's own usage shape. The expected dollar
amounts are worked by hand from the vendor price pages cited in models.json.

Ticket: docs/tickets/graded-attacked-budgeted/04-checks-cli-spend-cap.md (S4).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from company.runner import load_case, run_case  # noqa: E402
from llm import load_registry  # noqa: E402
from llm.call import complete  # noqa: E402
from llm.testing import StubTransport  # noqa: E402
from llm.types import Message  # noqa: E402

REGISTRY = load_registry()


def _complete(model, payload):
    return complete(
        model=model,
        messages=[Message.user("How much is Pro?")],
        transport=StubTransport([payload]),
        registry=REGISTRY,
    )


@pytest.mark.parametrize("spec", REGISTRY.models, ids=lambda s: s.model_id)
def test_every_entry_has_a_vendor_verified_cache_read_price(spec):
    assert spec.cache_read_price_per_mtok is not None, spec.model_id
    assert 0 < spec.cache_read_price_per_mtok < spec.input_price_per_mtok
    assert spec.cache_price_source.startswith("https://"), spec.model_id
    assert dt.date.fromisoformat(spec.verified_on) <= dt.date.today()


def test_anthropic_cache_reads_and_writes_are_billed_at_their_own_prices():
    # Anthropic reports cache reads and writes beside input_tokens, not inside it.
    response = _complete(
        "claude-sonnet-5",
        {
            "content": [{"type": "text", "text": "Pro is $10 a month."}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 200,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 500,
                "output_tokens": 100,
            },
        },
    )

    assert response.usage.input_tokens == 1700
    assert response.usage.cached_input_tokens == 1000
    assert response.usage.cache_write_input_tokens == 500
    # $2 in, $0.20 cache hit, $2.50 5-minute cache write, $10 out per MTok:
    # 200*2 + 1000*0.20 + 500*2.50 + 100*10 = 2850 millionths of a dollar.
    assert response.cost_usd == pytest.approx(0.00285)


def test_openai_cached_input_inside_input_tokens_is_not_billed_twice():
    # The Responses API counts cached and written tokens inside input_tokens.
    response = _complete(
        "gpt-6-sol",
        {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
            ],
            "usage": {
                "input_tokens": 15000,
                "input_tokens_details": {"cached_tokens": 12000, "cache_write_tokens": 3000},
                "output_tokens": 100,
            },
        },
    )

    assert response.usage.input_tokens == 15000
    assert response.usage.cached_input_tokens == 12000
    assert response.usage.cache_write_input_tokens == 3000
    # $2 in, $0.20 cached, $2.50 cache write, $10 out per MTok:
    # 0*2 + 12000*0.20 + 3000*2.50 + 100*10 = 10900 millionths.
    assert response.cost_usd == pytest.approx(0.0109)


def test_deepseek_cache_hits_are_read_from_its_own_usage_fields():
    response = _complete(
        "deepseek-flash",
        {
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 10000,
                "prompt_cache_hit_tokens": 8000,
                "prompt_cache_miss_tokens": 2000,
                "completion_tokens": 1000,
            },
        },
    )

    assert response.usage.cached_input_tokens == 8000
    # $0.30 miss, $0.006 hit, $1.20 out per MTok (peak):
    # 2000*0.30 + 8000*0.006 + 1000*1.20 = 1848 millionths.
    assert response.cost_usd == pytest.approx(0.001848)


def test_a_model_with_no_write_price_bills_written_tokens_as_input():
    spec = REGISTRY.get("gemini-3.8-flash")
    assert spec.cache_write_price_per_mtok is None
    # 1000 written tokens at the $0.75 input price, nothing else.
    assert spec.cost_usd(
        input_tokens=1000, output_tokens=0, cache_write_input_tokens=1000
    ) == pytest.approx(0.00075)


def test_the_outcome_bills_cached_input_at_the_cache_price():
    turn = {
        "content": [{"type": "text", "text": "Happy to help."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "cache_read_input_tokens": 1_000_000, "output_tokens": 0},
    }
    outcome = run_case(
        load_case("upgrade-to-pro"),
        "tests.fixtures.case_runner_agents:chats_but_never_acts",
        transport=StubTransport([turn]),
    )

    assert outcome.usage.cached_input_tokens == 1_000_000
    # A million cache hits on claude-sonnet-5 cost $0.20, not the $2.00 input price.
    assert outcome.cost_usd == pytest.approx(0.20)
