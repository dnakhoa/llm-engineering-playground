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

# The provider layer (llm/) and the .env file live at the repo root.
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, '.env'))
except ImportError:
    pass

from llm import CallOptions, Message, complete, configured_registry, default_model, load_registry
from llm.transport import HttpTransport

# llm/models.json (current IDs, prices, capabilities), plus your local server if
# .env points at one (LLM_PROVIDER=ollama or OPENAI_BASE_URL).
REGISTRY = configured_registry()
# The cheapest hosted model, for cost examples: a local server is priced at $0.
CHEAPEST = min(load_registry().models, key=lambda spec: spec.input_price_per_mtok)


def call(messages, *, system=None, **options):
    """One call through the provider layer, on the model your .env selects."""
    model = default_model(REGISTRY)
    return complete(
        model=model,
        messages=messages,
        system=system,
        options=CallOptions(**options),
        transport=HttpTransport(provider=REGISTRY.get(model).provider),
        registry=REGISTRY,
    )


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
        cost_estimate = CHEAPEST.cost_usd(input_tokens=token_count, output_tokens=0)
        print(f"\n  1000-word text: {token_count:,} tokens ≈ ${cost_estimate:.4f} (input, {CHEAPEST.model_id})")

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
    response = call(
        # System prompt — instructions, permanent context, constraints
        system=(
            "You are a concise technical tutor. "
            "Explain concepts in 2-3 sentences max. "
            "Use a concrete analogy."
        ),
        messages=[
            # Conversation history (simulating a prior turn)
            Message.user("What is a neural network?"),
            Message.assistant("A neural network is a system of interconnected nodes..."),
            # Current user message — always last
            Message.user("And what is a transformer?"),
        ],
        max_output_tokens=200,
        temperature=0.3,  # dropped, with a reason, for models that reject it
    )
    latency_ms = (time.time() - start) * 1000

    tokens_in  = response.usage.input_tokens
    tokens_out = response.usage.output_tokens

    print(f"  Model:        {response.model}")
    print(f"  Response:     {response.text[:150]}...")
    print(f"  Tokens in:    {tokens_in}")
    print(f"  Tokens out:   {tokens_out}")
    print(f"  Total tokens: {tokens_in + tokens_out}")
    print(f"  Stop reason:  {response.stop_reason}  (end_turn=natural end, max_tokens=hit the limit)")
    print(f"  Latency:      {latency_ms:.0f}ms")
    # Cost from the registry's prices for this model
    print(f"  Cost:         ${response.cost_usd:.6f}")
    for note in response.adjustments:
        print(f"  Layer:        {note}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAMPLING PARAMETERS — controlling output randomness
# ─────────────────────────────────────────────────────────────────────────────

def demo_temperature():
    print("\n=== Demo 4: Temperature Effect ===")

    # Whether a temperature is honoured depends on the model AND the effort level.
    # Claude Opus 4.7 and later reject a non-default temperature at any effort;
    # GPT-6 rejects it unless the request runs at effort "none". The provider layer
    # drops it where it would be refused and says why, so this demo asks the
    # registry for an effort level at which this model does take a temperature.
    spec = REGISTRY.get(default_model(REGISTRY))
    effort = None if spec.accepts_sampling_at(None) else next(iter(spec.sampling_effort_levels), None)
    if not spec.accepts_sampling_at(effort):
        print(f"  {spec.model_id} does not act on temperature: {spec.sampling_note}")
        print("  Set LLM_MODEL to a model that does (see llm/models.json) to see the effect.")
        return
    if effort is not None:
        print(f"  {spec.model_id} takes a temperature only at effort '{effort}'; running there.")

    prompt = "Give me one creative name for a coffee shop."
    unique = {}
    for temperature in (0.0, 1.0):
        print(f"\n  temperature={temperature}:")
        names = set()
        for _ in range(3):
            r = call([Message.user(prompt)], temperature=temperature, effort=effort,
                     max_output_tokens=30)
            name = r.text.strip()
            names.add(name)
            print(f"    '{name}'")
        unique[temperature] = len(names)

    print(f"\n  Unique responses at temp=0.0: {unique[0.0]}/3 (expect 1)")
    print(f"  Unique responses at temp=1.0: {unique[1.0]}/3 (expect 3)")


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
    Prices are the registry's standard rates; the registry names each source.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        input_tokens = len(enc.encode(input_text))
    except ImportError:
        input_tokens = len(input_text.split()) * 1.3  # rough estimate

    output_tokens = int(expected_output_words * 1.3)

    # Prices per 1M tokens come from the model registry (llm/models.json), which
    # records the vendor page and date each price was read from.
    spec = REGISTRY.get(model_name)
    input_cost  = spec.cost_usd(input_tokens=int(input_tokens), output_tokens=0)
    output_cost = spec.cost_usd(input_tokens=0, output_tokens=output_tokens)

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

    for m in REGISTRY.ids():
        est = estimate_cost(m, sample_input, expected_output_words=150)
        print(
            f"  {m:<25} | in={est['input_tokens']:>5} tok | "
            f"${est['total_cost_usd']:.5f}/call | "
            f"${est['daily_cost_1k_req']:.2f}/day@1k-calls"
        )

    # Scale calculation
    print("\n  Scale: 5,000 conversations/day with 500-word context, 150-word response:")
    big_input = "word " * 500
    for m in (CHEAPEST.model_id, "claude-sonnet-5"):
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
    print("  Temperature — only some models act on it; reasoning models use effort")
    print("  Cost — input is cheap, output is expensive; right-size your model")
