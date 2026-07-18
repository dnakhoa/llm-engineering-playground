# Module 00: LLM Foundations


> **Why this matters:** Every LLM application is built on tokens, embeddings, and context windows. Understanding these fundamentals lets you estimate costs before building, debug token-related issues, and choose the right model tier for your use case.


> **Start here.** This module builds the mental models you need before touching any code. If you're coming from software engineering or ML, you'll have some of this — but the LLM-specific details (tokens, context windows, sampling) are different from what you're used to.

## 🎯 Learning Objectives
- Understand what tokens are and why they're the fundamental unit of LLM cost and capacity
- Build geometric intuition for embeddings and why semantic search works
- Know the anatomy of a context window and its hard limits
- Control model output with temperature, top-p, and other sampling parameters
- Read an API call fluently: system prompt, user turn, parameters
- Choose the right model tier for a task (speed, cost, capability tradeoffs)

---

## 1. Tokens — The Atomic Unit of LLMs

**Tokens are not words, not characters, not bytes.** They're chunks that a tokenizer has learned tend to appear together in training data.

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4, Claude, most modern models

# Tokenize some text
text = "The quick brown fox jumps over the lazy dog."
tokens = enc.encode(text)
print(f"Text:   '{text}'")
print(f"Tokens: {tokens}")          # list of integer IDs
print(f"Count:  {len(tokens)}")     # 10 tokens for this sentence

# Decode back
print(f"Decoded: {[enc.decode([t]) for t in tokens]}")
# → ['The', ' quick', ' brown', ' fox', ' jumps', ' over', ' the', ' lazy', ' dog', '.']
```

### Why this matters in practice

```
"Hello"        → 1 token
"Hello world"  → 2 tokens
"antidisestablishmentarianism" → 6 tokens
"🎉"           → 3-4 tokens (emoji are expensive!)
```

**The model cannot count characters.** Ask it "how many letters in 'strawberry'?" and it will often fail — because it never sees individual characters, only token chunks.

**Pricing is per token.** A 10,000-word document is ~13,000 tokens. Knowing this lets you estimate costs before building.

**Context limits are token limits.** "128k context window" means 128,000 tokens of input + output combined.

### Token estimation rule of thumb
- English prose: ~1.3 tokens per word
- Code: ~1.5–2 tokens per "word" (identifiers, operators split)
- Chinese/Japanese: 1–2 characters per token (much more expensive per character than English)

---

## 2. Embeddings — The Geometry of Meaning

An embedding is a **high-dimensional vector** (list of ~1,500 numbers) that represents the meaning of text. Texts with similar meanings have vectors that point in similar directions.

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Semantically similar texts → high similarity
v1 = embed("The cat sat on the mat")
v2 = embed("A feline rested on the rug")
v3 = embed("The quarterly revenue report shows 24% growth")

print(f"Cat/feline similarity: {cosine_similarity(v1, v2):.3f}")  # ~0.93
print(f"Cat/revenue similarity: {cosine_similarity(v1, v3):.3f}") # ~0.12
```

### Why RAG works

Embeddings turn "find documents relevant to this question" into **nearest-neighbor search in vector space** — which is fast and scales to millions of documents. Keyword search fails for synonyms; embedding search doesn't.

```
Query: "how to fix memory leak"
─────────────────────────────────────────────────────────
Keyword search finds: only docs containing "memory leak"
Embedding search finds: docs about "heap overflow", "garbage collection", "OOM errors"
```

---

## 3. Context Window Anatomy

Every LLM call has this structure:

```
┌─────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW                       │
│                (e.g. 128,000 tokens max)                │
│                                                         │
│  INPUT TOKENS                      OUTPUT TOKENS        │
│  ─────────────────────────         ─────────────────    │
│  • System prompt                   • Model response     │
│  • Conversation history            (generated here)     │
│  • Retrieved documents                                  │
│  • Tool results                                         │
│  • The current user message                             │
│                                                         │
│  [ALL OF THE ABOVE COUNTED TOGETHER TOWARD THE LIMIT]  │
└─────────────────────────────────────────────────────────┘
```

### Key facts

- **Input tokens** are cheaper than output tokens (typically 3–5× cheaper). Sending a long context is cheap; generating a long response is expensive.
- **The model doesn't "remember" between API calls.** Every call is stateless — you re-send the full conversation history each time.
- **Hitting the context limit truncates your input**, not your output. You'll see the model losing track of early instructions — a sign your context is too long.

### Context limit by model tier (approximate, 2025-2026)

| Model tier | Example | Context limit | Best for |
|-----------|---------|--------------|---------|
| Small / fast | GPT-4o mini, Haiku 4.5 | 128k | High volume, simple tasks |
| Standard | GPT-4o, Sonnet 4.6 | 200k | Most production use cases |
| Large / smart | o3, Opus 4.8, Fable 5 | 200k–1M | Hard reasoning, research |

---

## 4. Sampling Parameters

These control how the model picks the next token. Wrong settings cause outputs to be either robotic (too low) or hallucination-prone (too high).

### Temperature

```python
# Temperature = 0: deterministic, always picks highest-probability token
# Temperature = 1: standard sampling
# Temperature > 1: more random (rarely used in production)

# Low temperature (0.0–0.3): best for
#   - factual Q&A, extraction, classification, code generation
# Medium temperature (0.5–0.7): best for
#   - general chat, summarization, reasoning
# High temperature (0.8–1.0): best for
#   - creative writing, brainstorming, diversity of outputs

response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.0,   # deterministic
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
```

### Top-p (nucleus sampling)

Top-p restricts sampling to the smallest set of tokens whose cumulative probability exceeds p. At `top_p=0.1` the model only considers the top 10% probability mass — very conservative. At `top_p=0.9` it considers a wider vocabulary.

**Rule of thumb**: use temperature OR top-p, not both. Most practitioners use temperature and leave top-p at default (1.0).

### Max tokens

Sets the maximum response length. Always set this — without it, the model may generate indefinitely (and you pay for every token).

```python
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=500,    # hard cap on output
    messages=[{"role": "user", "content": "Explain transformers."}]
)
```

---

## 5. API Call Anatomy

Every LLM API call has the same logical shape:

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    # ── What model to use ────────────────────────────────────
    model="gpt-4o-mini",

    # ── Sampling parameters ──────────────────────────────────
    temperature=0.3,
    max_tokens=1024,

    # ── The messages array (the context window content) ──────
    messages=[
        # System prompt: instructions, role, constraints
        # Placed at the top — gets highest attention
        {
            "role": "system",
            "content": "You are a concise technical writer. Use bullet points."
        },

        # Conversation history (user + assistant turns)
        {"role": "user",      "content": "What is a transformer?"},
        {"role": "assistant", "content": "A transformer is a neural network architecture..."},

        # The current user message (always last)
        {"role": "user",      "content": "How does attention work?"}
    ]
)

# Accessing the response
text = response.choices[0].message.content
tokens_in  = response.usage.prompt_tokens
tokens_out = response.usage.completion_tokens
print(f"Response: {text}")
print(f"Tokens: {tokens_in} in + {tokens_out} out")
```

### The role of the system prompt

The system prompt is the most powerful lever in prompt engineering. It establishes:
- Who the model is ("You are an expert Python developer")
- What constraints apply ("Always cite sources", "Respond in JSON only")
- What context is permanent ("The user's name is Alice")

System prompts are read at the start and get **primacy attention** — the model weights them most heavily. This is why your system prompt should contain your most critical instructions, not your conversation history.

---

## 6. Model Selection — When to Use What

Career-changers often default to the largest model "to be safe." This is wrong. Use the smallest model that can do the job reliably.

```
Task                          → Model tier
─────────────────────────────────────────────────────────
Classify sentiment (simple)   → Small (GPT-4o mini, Haiku 4.5)
Extract JSON from text        → Small / Standard
Summarize a document          → Small / Standard
Write production code         → Standard (GPT-4o, Sonnet 5)
Reason through a math proof   → Large / Reasoning (GPT-5.6, Opus 4.8)
Creative long-form writing    → Standard / Large
Multi-step research task      → Large / Reasoning (GPT-5.6-sol, Fable 5)
```

**Cost ratio example (approximate)**:
- Small: $0.15/1M input tokens
- Standard: $2.50–3/1M input tokens
- Large: $5–10/1M input tokens
- Reasoning: $10–75/1M input tokens (+ reasoning token costs)

A task that runs 10,000 times/day costs $1.50/day on small vs $25–750/day on large. Right-sizing is an economic decision, not just a technical one.

### Model Landscape (July 2026)

| Provider | Small | Standard | Large / Reasoning |
|----------|-------|----------|-------------------|
| OpenAI | gpt-4o-mini | gpt-4o | gpt-5.6, gpt-5.6-sol |
| Anthropic | Haiku 4.5 | Sonnet 5 | Opus 4.8, Fable 5 |
| DeepSeek | deepseek-chat | deepseek-v3 | deepseek-r1 |

---

## 7. The Responses API (Recommended for OpenAI)

OpenAI's **Responses API** is now the recommended way to build with their models, replacing Chat Completions as the primary interface. It provides better state management, reasoning support, and multi-agent orchestration.

```python
from openai import OpenAI
client = OpenAI()

# Responses API — recommended
response = client.responses.create(
    model="gpt-5.6",
    reasoning={"effort": "medium"},  # control reasoning depth
    input=[
        {"role": "user", "content": "Explain how transformers work."}
    ]
)

print(response.output_text)

# Usage includes reasoning tokens
print(f"Reasoning tokens: {response.usage.output_tokens_details.reasoning_tokens}")
```

### Key differences from Chat Completions:

| Feature | Chat Completions | Responses API |
|---------|-----------------|---------------|
| State management | Manual (re-send history) | Built-in (previous_response_id) |
| Reasoning control | Not available | `reasoning.effort` (none/low/medium/high/xhigh) |
| Multi-agent | Manual | Native orchestration |
| Reasoning persistence | Not available | `reasoning.context` (current_turn/all_turns) |
| Background mode | Not available | Long-running tasks with polling |

Chat Completions still works, but Responses API is recommended for new projects.

---

## 8. Putting It Together — Reading a Real System

Now you can read any LLM system and understand what's happening:

```
1. User sends a message
2. Application assembles the context window:
   - System prompt (instructions + constraints)
   - Conversation history (prior turns)
   - Retrieved documents from RAG (if any)
   - Tool results from previous agent steps (if any)
   - The user's current message
3. Sends assembled context to the model API
4. Model generates output tokens (and reasoning tokens for reasoning models)
5. Application parses the response:
   - If tool call → execute tool → add result → repeat from step 2
   - If plain text → return to user
6. Application logs: tokens_in, tokens_out, reasoning_tokens, latency, cost
```

Every module in this course deepens one part of this pipeline.

---


## 📚 Resources

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — visualize how text splits into tokens
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — understand caching mechanics
- [OpenAI Reasoning Models](https://platform.openai.com/docs/guides/reasoning) — effort levels and pro mode
- [3Blue1Brown: But what is a GPT?](https://www.youtube.com/watch?v=wjZofJX0v4M) — visual intro to transformers

## 🧪 Hands-On Exercises

1. **Token Counting**: Use tiktoken to count tokens in 10 different inputs — a tweet, a paragraph, a JSON object, a Python function, an emoji-heavy string. Which has the highest tokens-per-character ratio? Why?

2. **Embedding Space Exploration**: Embed 20 sentences on 5 different topics (4 sentences each). Compute all pairwise cosine similarities. Do within-topic pairs always score higher than cross-topic pairs? What breaks the pattern?

3. **Temperature Experiment**: Run the same creative prompt 5 times at temperature 0.0, then 5 times at temperature 0.8. Measure the variance of responses (word count, unique words, content similarity). At what temperature does output become unreliable?

4. **Cost Calculator**: Build a function `estimate_cost(model, input_text, expected_output_words)` that returns the dollar cost of one API call. Use real pricing from the provider docs. Now estimate the daily cost of a chatbot that handles 5,000 conversations/day with 500-word average context and 150-word average responses.

5. **Context Limit Test**: Create a message list that grows by adding turns until the API returns a context-length error. What error does OpenAI/Anthropic return? At what token count does it fail? How would you detect and handle this gracefully in production?

---

## 🔗 Where This Leads

| Concept learned here | Used in |
|---------------------|---------|
| Tokens & counting | Module 06 (Optimization), Module 12 (Context Engineering) |
| Embeddings | Module 02 (RAG), Module 11 (Memory) |
| Context window | Module 11 (Memory), Module 12 (Context Engineering) |
| Sampling params | Module 01 (Prompt Engineering), Module 04 (Evaluation) |
| API anatomy | Every module |
| Model selection | Module 06 (Optimization) |

**Next**: Module 01 — Prompt Engineering, where you'll use these fundamentals to craft effective LLM inputs.
