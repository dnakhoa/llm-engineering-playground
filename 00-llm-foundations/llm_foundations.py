"""
Module 00: LLM Foundations
Runnable demos covering tokens, embeddings, context windows,
sampling parameters, and cost estimation.
"""
import os
import sys
import time
import base64
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from provider import get_llm_client

client, model = get_llm_client()


# ─────────────────────────────────────────────────────────────────────────────
# 1. TOKENS — what the model actually sees
# ─────────────────────────────────────────────────────────────────────────────

def demo_tokens():
    print("\n=== Demo 1: Tokens ===")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        examples = [
            "Hello",
            "Hello world",
            "The quick brown fox jumps over the lazy dog.",
            "antidisestablishmentarianism",
            "🎉🎊🥳",
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1)+fibonacci(n-2)",
            "你好，世界",  # Chinese: "Hello, world" — more tokens/char than English
            '{"user": {"id": "abc123", "name": "Alice", "role": "admin"}}',
        ]

        print(f"  {'Text':<55} {'Tokens':>6}  {'Tok/char':>8}")
        print("  " + "─" * 72)
        for text in examples:
            tokens = enc.encode(text)
            ratio = len(tokens) / len(text)
            display = text if len(text) < 55 else text[:52] + "..."
            print(f"  {display:<55} {len(tokens):>6}   {ratio:>7.2f}")

        # Show what the model actually sees
        sentence = "The model cannot count characters!"
        token_ids = enc.encode(sentence)
        token_strings = [enc.decode([t]) for t in token_ids]
        print(f"\n  '{sentence}' → {token_strings}")

        # Practical: estimate cost before calling API
        long_text = "word " * 1000
        token_count = len(enc.encode(long_text))
        cost_estimate = (token_count / 1_000_000) * 0.15  # gpt-4o-mini pricing
        print(f"\n  1000-word text: {token_count:,} tokens ≈ ${cost_estimate:.4f} (input, gpt-4o-mini)")

    except ImportError:
        print("  tiktoken not installed: pip install tiktoken")


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDINGS — semantic geometry
# ─────────────────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """Get embedding vector for text using the configured provider."""
    from openai import OpenAI
    oai = OpenAI()
    response = oai.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def demo_embeddings():
    print("\n=== Demo 2: Embeddings & Semantic Similarity ===")
    try:
        from openai import OpenAI
        OpenAI()  # test that key exists

        # Semantically related pairs (should score high)
        pairs_related = [
            ("The cat sat on the mat", "A feline rested on the rug"),
            ("How to fix a memory leak in Python", "Python garbage collection and OOM errors"),
            ("Machine learning model training", "Neural network optimization"),
        ]

        # Unrelated pairs (should score low)
        pairs_unrelated = [
            ("The cat sat on the mat", "Q3 revenue grew 24% YoY"),
            ("How to fix a memory leak", "The history of the Roman Empire"),
        ]

        print("  Related pairs (expect high similarity):")
        for a, b in pairs_related:
            score = cosine_similarity(embed(a), embed(b))
            print(f"    {score:.3f} | '{a[:35]}' ↔ '{b[:35]}'")

        print("\n  Unrelated pairs (expect low similarity):")
        for a, b in pairs_unrelated:
            score = cosine_similarity(embed(a), embed(b))
            print(f"    {score:.3f} | '{a[:35]}' ↔ '{b[:35]}'")

        print("\n  Semantic search demo: finding the best match for a query")
        query = "how to handle memory errors in Python"
        documents = [
            "Python's garbage collector uses reference counting",
            "The best restaurants in San Francisco",
            "Common causes of OOM errors and how to fix them",
            "Introduction to machine learning algorithms",
            "Detecting and resolving memory leaks in production",
        ]
        query_vec = embed(query)
        scores = [(cosine_similarity(query_vec, embed(doc)), doc) for doc in documents]
        scores.sort(reverse=True)
        for score, doc in scores:
            marker = "← TOP MATCH" if score == scores[0][0] else ""
            print(f"    {score:.3f} | {doc[:60]} {marker}")

    except Exception as e:
        print(f"  Skipping (requires OPENAI_API_KEY): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. API ANATOMY — reading a real LLM call
# ─────────────────────────────────────────────────────────────────────────────

def demo_api_anatomy():
    print("\n=== Demo 3: API Call Anatomy ===")

    # A complete call showing every element
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=200,
        messages=[
            # System prompt — instructions, permanent context, constraints
            {
                "role": "system",
                "content": (
                    "You are a concise technical tutor. "
                    "Explain concepts in 2-3 sentences max. "
                    "Use a concrete analogy."
                )
            },
            # Conversation history (simulating a prior turn)
            {"role": "user",      "content": "What is a neural network?"},
            {"role": "assistant", "content": "A neural network is a system of interconnected nodes..."},
            # Current user message — always last
            {"role": "user",      "content": "And what is a transformer?"}
        ]
    )
    latency_ms = (time.time() - start) * 1000

    text         = response.choices[0].message.content
    tokens_in    = response.usage.prompt_tokens
    tokens_out   = response.usage.completion_tokens
    finish       = response.choices[0].finish_reason

    print(f"  Response:     {text[:150]}...")
    print(f"  Tokens in:    {tokens_in}")
    print(f"  Tokens out:   {tokens_out}")
    print(f"  Total tokens: {tokens_in + tokens_out}")
    print(f"  Finish reason: {finish}  (stop=natural end, length=hit max_tokens)")
    print(f"  Latency:      {latency_ms:.0f}ms")

    # Cost calculation
    input_cost  = (tokens_in  / 1_000_000) * 0.15   # gpt-4o-mini pricing
    output_cost = (tokens_out / 1_000_000) * 0.60
    print(f"  Estimated cost: ${input_cost + output_cost:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAMPLING PARAMETERS — controlling output randomness
# ─────────────────────────────────────────────────────────────────────────────

def demo_temperature():
    print("\n=== Demo 4: Temperature Effect ===")

    prompt = "Give me one creative name for a coffee shop."

    print("  temperature=0.0 (deterministic — same every time):")
    results_low = set()
    for _ in range(3):
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=30,
            messages=[{"role": "user", "content": prompt}]
        )
        name = r.choices[0].message.content.strip()
        results_low.add(name)
        print(f"    '{name}'")

    print(f"\n  temperature=1.0 (varied — different every time):")
    results_high = set()
    for _ in range(3):
        r = client.chat.completions.create(
            model=model, temperature=1.0, max_tokens=30,
            messages=[{"role": "user", "content": prompt}]
        )
        name = r.choices[0].message.content.strip()
        results_high.add(name)
        print(f"    '{name}'")

    print(f"\n  Unique responses at temp=0.0: {len(results_low)}/3 (expect 1)")
    print(f"  Unique responses at temp=1.0: {len(results_high)}/3 (expect 3)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. COST ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def estimate_cost(
    model_name: str,
    input_text: str,
    expected_output_words: int = 150,
) -> dict:
    """
    Estimate the cost of a single LLM API call.
    Prices are approximate — check provider docs for current rates.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        input_tokens = len(enc.encode(input_text))
    except ImportError:
        input_tokens = len(input_text.split()) * 1.3  # rough estimate

    output_tokens = int(expected_output_words * 1.3)

    # Approximate pricing per 1M tokens (input/output), mid-2026
    pricing = {
        "gpt-4o-mini":         (0.15,   0.60),
        "gpt-4o":              (2.50,  10.00),
        "gpt-4.1":             (2.00,   8.00),
        "claude-haiku-4-5":    (0.80,   4.00),
        "claude-sonnet-4-6":   (3.00,  15.00),
        "claude-opus-4-8":    (15.00,  75.00),
    }

    # Normalize model name for lookup
    lookup = next((k for k in pricing if k in model_name.lower()), None)
    if lookup:
        price_in, price_out = pricing[lookup]
    else:
        price_in, price_out = 2.50, 10.00  # default: gpt-4o

    input_cost  = (input_tokens  / 1_000_000) * price_in
    output_cost = (output_tokens / 1_000_000) * price_out

    return {
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd":  round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd":  round(input_cost + output_cost, 6),
        "daily_cost_1k_req": round((input_cost + output_cost) * 1000, 4),
    }


def demo_cost_estimator():
    print("\n=== Demo 5: Cost Estimator ===")

    # Typical chatbot message
    sample_input = (
        "You are a helpful customer support agent for Acme Corp. "
        "The user has the following issue: " + "I can't log into my account. " * 5
    )

    for m in ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-6", "claude-opus-4-8"]:
        est = estimate_cost(m, sample_input, expected_output_words=150)
        print(
            f"  {m:<25} | in={est['input_tokens']:>5} tok | "
            f"${est['total_cost_usd']:.5f}/call | "
            f"${est['daily_cost_1k_req']:.2f}/day@1k-calls"
        )

    # Scale calculation
    print("\n  Scale: 5,000 conversations/day with 500-word context, 150-word response:")
    big_input = "word " * 500
    for m in ["gpt-4o-mini", "gpt-4o"]:
        est = estimate_cost(m, big_input, expected_output_words=150)
        daily = est["total_cost_usd"] * 5000
        monthly = daily * 30
        print(f"  {m:<25} | ${daily:.2f}/day | ${monthly:.0f}/month")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Module 00: LLM Foundations")
    print("=" * 60)

    demo_tokens()
    demo_embeddings()
    demo_api_anatomy()
    demo_temperature()
    demo_cost_estimator()

    print("\n✅ Foundations complete.")
    print("\nWhat to remember:")
    print("  Tokens — the unit of cost, context, and capacity")
    print("  Embeddings — meaning as geometry; power behind semantic search")
    print("  Context window — stateless; you re-send everything each call")
    print("  Temperature — 0.0 for deterministic, 0.7 for creative")
    print("  Cost — input is cheap, output is expensive; right-size your model")
