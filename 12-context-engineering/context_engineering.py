"""
Module 12: Context Engineering
Practical examples of context window optimization, observation masking,
prefix caching, token budgeting, and extended thinking.
"""
import os
import re
import time
import json
import hashlib
import tiktoken
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import deque

# ── Provider setup (uses shared provider.py pattern) ──────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from provider import get_llm_client

client, model = get_llm_client()


# ─────────────────────────────────────────────────────────────────────────────
# 1. TOKEN BUDGET TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class TokenBudget:
    """
    Enforces hard per-slot token limits across a context assembly.

    Prevents any single component (docs, history, tools) from crowding
    out others — a common cause of the U-shaped attention problem.
    """

    def __init__(self, total: int, model: str = "cl100k_base"):
        self.total = total
        self._enc = tiktoken.get_encoding(model)
        self._alloc: dict[str, int] = {}
        self._used: dict[str, int] = {}

    def allocate(self, slot: str, tokens: int) -> None:
        committed = sum(self._alloc.values())
        if committed + tokens > self.total:
            raise ValueError(
                f"Cannot allocate {tokens} for '{slot}': "
                f"only {self.total - committed} tokens remaining"
            )
        self._alloc[slot] = tokens
        self._used[slot] = 0

    def consume(self, slot: str, text: str) -> str:
        if slot not in self._alloc:
            raise KeyError(f"Slot '{slot}' not allocated.")
        limit = self._alloc[slot]
        tokens = self._enc.encode(text)
        if len(tokens) > limit:
            tokens = tokens[:limit]
            text = self._enc.decode(tokens)
        self._used[slot] = len(tokens)
        return text

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    @property
    def remaining(self) -> int:
        return self.total - sum(self._used.values())

    def report(self) -> str:
        lines = [f"\n{'─'*40}", f"Token Budget  (total: {self.total:,})"]
        for slot, alloc in self._alloc.items():
            used = self._used.get(slot, 0)
            bar_len = 20
            filled = int(bar_len * used / alloc) if alloc else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            lines.append(f"  {slot:<20} [{bar}] {used:>5}/{alloc:<5} ({used/alloc*100:.0f}%)")
        lines.append(f"  {'UNALLOCATED':<20}  {'':22} {self.remaining:>5} remaining")
        lines.append(f"{'─'*40}")
        return "\n".join(lines)


def demo_token_budget():
    print("\n=== Demo 1: Token Budget ===")
    budget = TokenBudget(total=4000)
    budget.allocate("system_prompt", 400)
    budget.allocate("retrieved_docs", 1800)
    budget.allocate("conversation",   1200)
    budget.allocate("response",        600)

    system_text = budget.consume(
        "system_prompt",
        "You are an expert financial analyst specializing in emerging markets. "
        "Always cite sources. Be precise and quantitative."
    )
    docs_text = budget.consume(
        "retrieved_docs",
        "Annual report excerpt: " + "Revenue grew 24% YoY. " * 200  # long doc
    )
    history_text = budget.consume(
        "conversation",
        "User: What's the revenue trend?\nAssistant: Revenue is growing... " * 20
    )

    print(budget.report())
    print(f"\nDocs truncated to: {budget.count(docs_text):,} tokens")
    print(f"History truncated to: {budget.count(history_text):,} tokens")


# ─────────────────────────────────────────────────────────────────────────────
# 2. OBSERVATION MASKING
# ─────────────────────────────────────────────────────────────────────────────

class ObservationMasker:
    """
    Compresses tool outputs before they enter the context window.

    Prevents context bloat from large tool responses — a web page,
    a database dump, or a log file — crowding out signal.
    """

    def __init__(self, max_tokens: int = 400, llm_compress: bool = False):
        self.max_tokens = max_tokens
        self.llm_compress = llm_compress
        self._enc = tiktoken.get_encoding("cl100k_base")

    def mask(self, raw: str, task_context: str = "") -> str:
        token_count = len(self._enc.encode(raw))
        if token_count <= self.max_tokens:
            return raw  # small enough, pass through unchanged

        # Step 1: structural extraction (fast, zero LLM cost)
        cleaned = self._structural_clean(raw)

        if len(self._enc.encode(cleaned)) <= self.max_tokens:
            return cleaned

        # Step 2: LLM-based semantic compression (optional, costs tokens)
        if self.llm_compress and task_context:
            return self._semantic_compress(cleaned, task_context)

        # Step 3: hard truncate
        tokens = self._enc.encode(cleaned)[: self.max_tokens]
        return self._enc.decode(tokens) + "\n[...truncated]"

    def _structural_clean(self, text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)          # strip HTML
        text = re.sub(r'https?://\S+', '[URL]', text) # compact URLs
        text = re.sub(r'\n{3,}', '\n\n', text)        # collapse blank lines
        return text.strip()

    def _semantic_compress(self, text: str, context: str) -> str:
        prompt = (
            f"Task context: {context}\n\n"
            "Extract ONLY facts relevant to this task from the text below. "
            "Be terse — bullet points only, no prose.\n\n"
            f"TEXT:\n{text}"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


def mask_observation(max_tokens: int = 400):
    """Decorator: auto-masks return value of any tool function."""
    masker = ObservationMasker(max_tokens=max_tokens)

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)
            if isinstance(result, str):
                original_tokens = len(tiktoken.get_encoding("cl100k_base").encode(result))
                compressed = masker.mask(result)
                compressed_tokens = len(tiktoken.get_encoding("cl100k_base").encode(compressed))
                if original_tokens != compressed_tokens:
                    print(f"  [mask] {func.__name__}: {original_tokens} → {compressed_tokens} tokens")
                return compressed
            return result
        return wrapper
    return decorator


@mask_observation(max_tokens=150)
def fake_web_search(query: str) -> str:
    """Simulates a verbose web search result."""
    return (
        f"<html><head><title>Search: {query}</title></head><body>"
        "<nav>Home | About | Contact | Blog | Terms | Privacy</nav>"
        "<div class='ad'>Buy now! Special offer!</div>"
        "<article>"
        "<h1>The Comprehensive Guide to Context Engineering</h1>"
        "<p>Context engineering, as defined by Anthropic researchers in 2025, "
        "is the discipline of constructing the right information and format "
        "for LLM context windows. Key concepts include observation masking, "
        "prefix caching, token budgets, and the U-shaped attention curve.</p>"
        "<p>The term was popularized in a June 2025 blog post that argued "
        "prompt engineering is a subset of the broader context engineering discipline.</p>"
        "</article>"
        "<footer>© 2026 Example Corp. All rights reserved. " + "lorem ipsum " * 200 + "</footer>"
        "</body></html>"
    )


def demo_observation_masking():
    print("\n=== Demo 2: Observation Masking ===")
    result = fake_web_search("context engineering definition")
    print(f"Masked result:\n{result[:300]}...")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONTEXT COMPRESSION — SLIDING WINDOW WITH ANCHORING
# ─────────────────────────────────────────────────────────────────────────────

def sliding_window_compress(
    messages: list[dict],
    anchor_turns: int = 2,
    recent_turns: int = 4,
) -> list[dict]:
    """
    Keeps first N turns (anchor) + last M turns (recency).
    Drops the middle — the lowest-attention zone.

    Ideal for long multi-turn conversations where the early context
    established key facts and the recent context is immediately relevant.
    """
    pairs = len(messages) // 2  # approximate turn count
    if pairs <= anchor_turns + recent_turns:
        return messages  # short enough, no compression needed

    anchors = messages[:anchor_turns * 2]
    recent  = messages[-(recent_turns * 2):]
    dropped = len(messages) - len(anchors) - len(recent)

    summary_marker = {
        "role": "system",
        "content": f"[{dropped} earlier messages omitted — context compressed for efficiency]"
    }
    return anchors + [summary_marker] + recent


def demo_sliding_window():
    print("\n=== Demo 3: Sliding Window Compression ===")
    # Simulate a 20-turn conversation
    messages = []
    for i in range(20):
        messages.append({"role": "user",      "content": f"Turn {i+1} user message"})
        messages.append({"role": "assistant", "content": f"Turn {i+1} assistant response"})

    compressed = sliding_window_compress(messages, anchor_turns=2, recent_turns=3)
    print(f"Original:   {len(messages)} messages ({len(messages)//2} turns)")
    print(f"Compressed: {len(compressed)} messages")
    for m in compressed:
        print(f"  [{m['role']:9s}] {m['content'][:60]}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREFIX CACHING — ANTHROPIC
# ─────────────────────────────────────────────────────────────────────────────

def demo_anthropic_prefix_caching():
    """
    Demonstrates Anthropic's explicit prefix caching via cache_control.

    Cache write: 1.25x input price (5-min TTL) or 2x (1-hour TTL)
    Cache read:  0.1x input price (90% discount)

    Minimum: 1,024 tokens to qualify for caching.
    """
    print("\n=== Demo 4: Anthropic Prefix Caching ===")
    try:
        import anthropic
        anth = anthropic.Anthropic()

        # Large stable reference document (must be ≥1024 tokens to cache)
        reference_doc = """
        ACME Corp Financial Reference Guide — Q1-Q4 2025

        This document contains authoritative data for financial analysis.
        Always cite section numbers when referencing data.

        Section 1: Revenue
        - Q1 2025: $142M (YoY +18%)
        - Q2 2025: $158M (YoY +22%)
        - Q3 2025: $171M (YoY +19%)
        - Q4 2025: $195M (YoY +24%)
        - Full Year 2025: $666M (YoY +21%)

        Section 2: Operating Costs
        - Personnel: 45% of revenue
        - Infrastructure: 12% of revenue
        - Marketing: 8% of revenue
        - R&D: 18% of revenue

        Section 3: Key Metrics
        - Gross margin: 68%
        - Operating margin: 17%
        - Net margin: 13%
        - ARR: $780M (as of Dec 2025)
        - NRR: 118%
        - Customer count: 4,200 enterprise
        """ * 10  # repeat to exceed 1024 token minimum

        # First call: WRITES to cache (costs 1.25x for 5-min TTL)
        response1 = anth.messages.create(
            model="claude-opus-4-8",
            max_tokens=256,
            system=[
                {"type": "text", "text": "You are an expert financial analyst."},
                {
                    "type": "text",
                    "text": reference_doc,
                    "cache_control": {"type": "ephemeral"}  # 5-min TTL
                }
            ],
            messages=[{"role": "user", "content": "What was Q3 2025 revenue?"}]
        )

        print(f"Call 1 (cache write):")
        print(f"  Cache write tokens: {response1.usage.cache_creation_input_tokens:,}")
        print(f"  Cache read tokens:  {response1.usage.cache_read_input_tokens:,}")
        print(f"  Answer: {response1.content[0].text[:100]}")

        # Second call: READS from cache (costs 0.1x = 90% discount)
        response2 = anth.messages.create(
            model="claude-opus-4-8",
            max_tokens=256,
            system=[
                {"type": "text", "text": "You are an expert financial analyst."},
                {
                    "type": "text",
                    "text": reference_doc,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{"role": "user", "content": "What was Q4 2025 revenue?"}]
        )

        print(f"\nCall 2 (cache read — 90% discount):")
        print(f"  Cache write tokens: {response2.usage.cache_creation_input_tokens:,}")
        print(f"  Cache read tokens:  {response2.usage.cache_read_input_tokens:,}")
        print(f"  Answer: {response2.content[0].text[:100]}")

    except ImportError:
        print("  anthropic package not installed — pip install anthropic")
    except Exception as e:
        print(f"  Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXTENDED THINKING (ANTHROPIC)
# ─────────────────────────────────────────────────────────────────────────────

def demo_extended_thinking():
    """
    Extended thinking exposes the model's internal chain-of-thought.
    Thinking tokens are billed at output token rates.

    Use for: multi-step reasoning, math, architecture decisions, analysis.
    Skip for: simple classification, high-volume cheap calls.

    As of Claude Opus 4.7+ / Sonnet 4.6+ / Fable 5:
    - 'adaptive' mode is the default (model decides reasoning depth)
    - 'enabled' + budget_tokens is available for explicit control
    - Thinking blocks appear between tool calls (inter-tool reasoning)
    """
    print("\n=== Demo 5: Extended Thinking ===")
    try:
        import anthropic
        anth = anthropic.Anthropic()

        problem = """
        A company has 3 products: A ($50/unit, 40% margin), B ($120/unit, 60% margin),
        C ($200/unit, 75% margin). They can produce max 1000 units/month total.
        Product demand caps: A=600, B=500, C=300.
        Production costs: A needs 1 worker-hour, B needs 3, C needs 5.
        They have 2,400 worker-hours/month.
        What production mix maximizes profit?
        """

        # Extended thinking with explicit budget
        response = anth.messages.create(
            model="claude-opus-4-8",
            max_tokens=8000,
            thinking={"type": "enabled", "budget_tokens": 4000},
            messages=[{"role": "user", "content": problem}]
        )

        thinking_tokens = 0
        answer = ""
        for block in response.content:
            if block.type == "thinking":
                thinking_tokens = len(block.thinking.split())  # rough estimate
                print(f"  Thinking block ({thinking_tokens} words approx): "
                      f"{block.thinking[:150]}...")
            elif block.type == "text":
                answer = block.text

        print(f"\n  Answer: {answer[:300]}")
        print(f"  Total output tokens: {response.usage.output_tokens:,}")

    except ImportError:
        print("  anthropic package not installed — pip install anthropic")
    except Exception as e:
        print(f"  Note: {e}")
        print("  Demonstrating structure — extended thinking requires claude-opus-4-8+")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONTEXT QUALITY AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def context_quality_audit(messages: list[dict]) -> dict:
    """
    Analyze a message list for common context engineering issues.
    Returns a report with detected problems and recommendations.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    report = {"issues": [], "stats": {}}

    total_tokens = sum(
        len(enc.encode(m.get("content", "") or ""))
        for m in messages
    )
    report["stats"]["total_tokens"] = total_tokens

    # Check for context bloat (single message > 20% of total)
    for i, msg in enumerate(messages):
        content = msg.get("content", "") or ""
        msg_tokens = len(enc.encode(content))
        if msg_tokens > total_tokens * 0.25 and total_tokens > 1000:
            report["issues"].append({
                "type": "context_bloat",
                "severity": "warning",
                "message": f"Message {i} ({msg['role']}) is {msg_tokens} tokens "
                           f"({msg_tokens/total_tokens:.0%} of total). Consider masking."
            })

    # Check for middle-heavy loading (more tokens in middle than ends)
    if len(messages) >= 5:
        n = len(messages)
        first_third  = messages[:n//3]
        middle_third = messages[n//3: 2*n//3]
        last_third   = messages[2*n//3:]

        first_tok  = sum(len(enc.encode(m.get("content","") or "")) for m in first_third)
        middle_tok = sum(len(enc.encode(m.get("content","") or "")) for m in middle_third)
        last_tok   = sum(len(enc.encode(m.get("content","") or "")) for m in last_third)

        report["stats"]["distribution"] = {
            "first_third": first_tok,
            "middle_third": middle_tok,
            "last_third": last_tok
        }

        if middle_tok > first_tok + last_tok:
            report["issues"].append({
                "type": "attention_risk",
                "severity": "warning",
                "message": (
                    f"Middle zone has {middle_tok} tokens vs "
                    f"{first_tok + last_tok} in ends. "
                    "Information in the middle is under-attended."
                )
            })

    # Check for missing system prompt
    if not any(m.get("role") == "system" for m in messages):
        report["issues"].append({
            "type": "missing_system_prompt",
            "severity": "info",
            "message": "No system prompt found. Key instructions in system prompt get primacy attention."
        })

    report["healthy"] = not any(
        i["severity"] == "warning" for i in report["issues"]
    )
    return report


def demo_context_audit():
    print("\n=== Demo 6: Context Quality Audit ===")
    messages = [
        # No system prompt
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        # Big bloated middle message
        {"role": "user", "content": "Here's the full document: " + "data " * 500},
        {"role": "assistant", "content": "I've reviewed it."},
        {"role": "user", "content": "Summarize the key points"},
        {"role": "assistant", "content": "Sure!"},
        {"role": "user", "content": "What's the main finding?"},
    ]

    report = context_quality_audit(messages)
    print(f"  Total tokens: {report['stats']['total_tokens']:,}")
    print(f"  Healthy: {report['healthy']}")
    print(f"  Issues found: {len(report['issues'])}")
    for issue in report["issues"]:
        severity = "⚠️ " if issue["severity"] == "warning" else "ℹ️ "
        print(f"  {severity}{issue['type']}: {issue['message'][:100]}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Module 12: Context Engineering")
    print("=" * 60)

    demo_token_budget()
    demo_observation_masking()
    demo_sliding_window()
    demo_anthropic_prefix_caching()
    demo_extended_thinking()
    demo_context_audit()

    print("\n✅ All demos complete.")
    print("\nKey takeaways:")
    print("  1. Budget tokens per slot — prevent crowding")
    print("  2. Mask large tool outputs before they hit context")
    print("  3. Compress old conversation turns (anchor + recency)")
    print("  4. Cache stable prefixes — 90% cost reduction on reads")
    print("  5. Use extended thinking only for genuinely hard problems")
    print("  6. Audit context distribution — avoid middle-heavy loading")
