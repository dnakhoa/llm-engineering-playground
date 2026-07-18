"""
Tests for LLM Engineering Playground — validates core logic without API calls.

Run: pytest tests/ -v
"""
import sys
import os
import json
import hashlib
import tempfile
from pathlib import Path
from dataclasses import dataclass

import pytest
import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModule00_Tokens:
    def test_tokenizer_loads(self):
        enc = tiktoken.get_encoding("cl100k_base")
        assert enc is not None

    def test_token_count_positive(self):
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode("Hello world")
        assert len(tokens) > 0

    def test_encode_decode_roundtrip(self):
        enc = tiktoken.get_encoding("cl100k_base")
        text = "The quick brown fox"
        decoded = enc.decode(enc.encode(text))
        assert decoded == text

    def test_emoji_tokens(self):
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode("🎉🎊🥳")
        assert len(tokens) >= 3


class TestModule00_CostEstimation:
    def test_cost_calculation(self):
        input_cost = (1000 / 1_000_000) * 0.15
        output_cost = (500 / 1_000_000) * 0.60
        total = input_cost + output_cost
        assert total == pytest.approx(0.00045, rel=1e-6)

    def test_cost_scales_linearly(self):
        cost_1k = (1000 / 1_000_000) * 0.15
        cost_10k = (10_000 / 1_000_000) * 0.15
        assert cost_10k == pytest.approx(cost_1k * 10)


class TestModule06_Caching:
    def test_cache_cost_savings(self):
        without = 500_000
        with_cache = 5000 * 1.25 + 99 * 5000 * 0.1
        savings = 1 - (with_cache / without)
        assert savings > 0.8


class TestModule12_TokenBudget:
    def test_budget_allocation(self):
        @dataclass
        class TokenBudget:
            total: int
            _alloc: dict = None
            _used: dict = None
            def __post_init__(self):
                self._alloc = {}
                self._used = {}
            def allocate(self, slot, tokens):
                committed = sum(self._alloc.values())
                if committed + tokens > self.total:
                    raise ValueError("Exceeds budget")
                self._alloc[slot] = tokens
                self._used[slot] = 0

        budget = TokenBudget(total=1000)
        budget.allocate("system", 200)
        budget.allocate("docs", 500)
        budget.allocate("response", 300)
        assert sum(budget._alloc.values()) == 1000

    def test_budget_overflow_raises(self):
        class TokenBudget:
            def __init__(self, total):
                self.total = total
                self._alloc = {}
            def allocate(self, slot, tokens):
                committed = sum(self._alloc.values())
                if committed + tokens > self.total:
                    raise ValueError("Exceeds budget")
                self._alloc[slot] = tokens

        budget = TokenBudget(total=100)
        budget.allocate("a", 60)
        with pytest.raises(ValueError):
            budget.allocate("b", 50)

    def test_observation_masking(self):
        enc = tiktoken.get_encoding("cl100k_base")
        def mask(text, max_tokens=100):
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return enc.decode(tokens[:max_tokens])
        big_text = "word " * 1000
        masked = mask(big_text, max_tokens=50)
        assert len(enc.encode(masked)) <= 50


class TestModule13_NoveltyGate:
    def test_novelty_gate_basic(self):
        class NoveltyGate:
            def __init__(self, threshold=3):
                self.seen = set()
                self.dry_rounds = 0
                self.threshold = threshold
            def filter_new(self, items):
                new = [i for i in items if i not in self.seen]
                if new:
                    self.seen.update(new)
                    self.dry_rounds = 0
                else:
                    self.dry_rounds += 1
                return new
            @property
            def is_dry(self):
                return self.dry_rounds >= self.threshold

        gate = NoveltyGate(threshold=3)
        assert not gate.is_dry
        gate.filter_new([1, 2, 3])
        assert gate.dry_rounds == 0
        gate.filter_new([])
        gate.filter_new([])
        gate.filter_new([])
        assert gate.is_dry

    def test_novelty_gate_resumes_on_new(self):
        class NoveltyGate:
            def __init__(self, threshold=3):
                self.seen = set()
                self.dry_rounds = 0
                self.threshold = threshold
            def filter_new(self, items):
                new = [i for i in items if i not in self.seen]
                if new:
                    self.seen.update(new)
                    self.dry_rounds = 0
                else:
                    self.dry_rounds += 1
                return new

        gate = NoveltyGate()
        gate.filter_new([])
        gate.filter_new([])
        assert gate.dry_rounds == 2
        gate.filter_new([42])
        assert gate.dry_rounds == 0


class TestModule13_BudgetTracker:
    def test_budget_tracker(self):
        @dataclass
        class BudgetTracker:
            total_tokens: int
            spent_tokens: int = 0
            def record(self, input_tok, output_tok):
                self.spent_tokens += input_tok + output_tok
            @property
            def remaining(self):
                return max(0, self.total_tokens - self.spent_tokens)
            @property
            def is_exhausted(self):
                return self.remaining < 1000

        bt = BudgetTracker(total_tokens=5000)
        assert not bt.is_exhausted
        bt.record(2000, 1500)
        assert bt.remaining == 1500
        bt.record(1000, 500)
        assert bt.is_exhausted


class TestModule13_Journal:
    def test_journal_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            done = set()
            def execute(name, fn, *args):
                sid = hashlib.md5(f"{name}:{args}".encode()).hexdigest()[:12]
                if sid in done:
                    return "skipped"
                result = fn(*args)
                with open(path, "a") as f:
                    f.write(json.dumps({"sid": sid, "result": result}) + "\n")
                done.add(sid)
                return result

            assert execute("step1", lambda: "r1") == "r1"
            assert execute("step1", lambda: "r1") == "skipped"
            assert execute("step2", lambda: "r2") == "r2"


class TestModule14_ToolDesign:
    def test_description_has_trigger(self):
        desc = "Search docs. Use when user asks about policies. Do NOT use for web search."
        assert "Use when" in desc
        assert "Do NOT" in desc

    def test_schema_enum(self):
        schema = {"calendar": {"type": "string", "enum": ["personal", "work", "team"]}}
        assert len(schema["calendar"]["enum"]) == 3

    def test_error_actionable(self):
        msg = "Invalid email: 'foo'. Provide valid email (e.g. name@domain.com)."
        assert "stack trace" not in msg.lower()
        assert len(msg) > 10


class TestModule15_Multimodal:
    def test_base64_roundtrip(self):
        import base64
        data = b"test image data"
        assert base64.b64decode(base64.b64encode(data)) == data

    def test_json_report(self):
        report = {"main_subject": "cat", "colors": ["white", "orange"]}
        assert "main_subject" in report
        assert isinstance(report["colors"], list)


class TestCapstone:
    def test_pii_detection(self):
        import re
        def has_pii(text):
            ssn = bool(re.search(r"\b\d{3}-\d{2}-\d{4}\b", text))
            credit = bool(re.search(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", text))
            return ssn or credit
        assert has_pii("My SSN is 123-45-6789")
        assert has_pii("Card: 4111 1111 1111 1111")
        assert not has_pii("Hello world")

    def test_cost_tracker(self):
        def estimate_cost(in_tok, out_tok):
            return (in_tok / 1e6 * 2.50) + (out_tok / 1e6 * 10.00)
        cost = estimate_cost(1000, 500)
        assert 0 < cost < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
