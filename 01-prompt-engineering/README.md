# Module 1: Prompt Engineering

## What is Prompt Engineering?

Prompt engineering is the art and science of designing effective prompts to guide Large Language Models (LLMs) to produce desired outputs. It's the foundation of working with LLMs without modifying their weights.

## Key Concepts

### 1. Zero-Shot Prompting
Asking the model to perform a task without any examples.

**Example:**
```
Translate this to French: "Hello, how are you?"
```

### 2. Few-Shot Prompting
Providing examples in the prompt to guide the model's behavior.

**Example:**
```
Convert to formal tone:
Input: "Hey, what's up?"
Output: "Greetings, how are you doing?"

Input: "Thanks a lot!"
Output: "Thank you very much!"

Input: "Can you help me?"
Output: 
```

### 3. Chain-of-Thought (CoT)
Encouraging the model to show its reasoning process before giving the final answer.

**Example:**
```
Q: John has 5 apples. He buys 3 more, then gives away 2. How many does he have?
A: Let's think step by step:
- John starts with 5 apples
- He buys 3 more: 5 + 3 = 8 apples
- He gives away 2: 8 - 2 = 6 apples
- Final answer: 6 apples
```

### 4. Role Prompting
Assigning a specific role or persona to the model.

**Example:**
```
You are an expert Python developer with 10 years of experience. Review the following code and suggest improvements...
```

### 5. Structured Output — Modern Approach

There are three ways to get structured output from an LLM, from least to most reliable:

**Method A — Ask nicely in the prompt (fragile, avoid in production)**
```
Extract information and return as JSON: {"name": ..., "age": ...}
```
Problem: model may add markdown fences, prose, or deviate from the schema.

**Method B — Native JSON Schema (OpenAI, recommended)**
```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class Person(BaseModel):
    name: str
    age: int
    occupation: str

# .parse() + response_format=Pydantic model → guaranteed schema-conformant output
completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    response_format=Person,
    messages=[{
        "role": "user",
        "content": "John Smith is a 35-year-old software engineer."
    }]
)
person: Person = completion.choices[0].message.parsed
print(person.name, person.age)  # → "John Smith", 35
```
With `strict=True`, OpenAI **guarantees** the output matches the schema.

**Method C — Tool use for extraction (Anthropic)**
```python
import anthropic, json

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    tools=[{
        "name": "extract_person",
        "description": "Extract structured person data",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age":  {"type": "integer"},
            },
            "required": ["name", "age"]
        }
    }],
    tool_choice={"type": "tool", "name": "extract_person"},  # force this tool
    messages=[{"role": "user", "content": "John Smith is a 35-year-old engineer."}]
)
tool_block = next(b for b in response.content if b.type == "tool_use")
data = tool_block.input   # → {"name": "John Smith", "age": 35}
```

**The `instructor` library — provider-agnostic extraction**
```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())

class Person(BaseModel):
    name: str
    age: int

# Works identically for Anthropic via instructor.from_anthropic(Anthropic())
person = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Person,
    messages=[{"role": "user", "content": "Jane is 33 years old."}]
)
# → Person(name='Jane', age=33)
# Automatic retry on validation failure
```

## Best Practices

1. **Be Clear and Specific**: Vague prompts lead to vague answers
2. **Provide Context**: Give relevant background information
3. **Use Delimiters**: Separate instructions from content using ```, """, or ---
4. **Specify Format**: Tell the model exactly how you want the output
5. **Iterate**: Refine prompts based on results

## Hands-On Exercises

Two ways to learn hands-on:

1. **Interactive notebook** (recommended): Open `prompt_engineering.ipynb` and run cells one at a time — you'll see LLM responses inline.
2. **Script**: Run `python prompt_examples.py` to execute all 8 examples end-to-end.

**Challenges to try:**
- Modify the few-shot examples in Example 2 and observe how output style changes
- Add a `role` parameter to Example 4 and compare responses (e.g., "pirate", "professor")
- Change the JSON schema in Example 5 to extract different fields
- Build your own chain-of-thought prompt for a math problem

---

## 6. Multimodal — Vision Inputs

Modern models accept images alongside text. This is essential for document parsing, screenshot analysis, UI review, and visual Q&A.

```python
import base64
from openai import OpenAI

client = OpenAI()

# Option A: URL (simplest)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What does this chart show?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
        ]
    }]
)

# Option B: Local file (base64)
with open("screenshot.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "List all the error messages visible."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high"   # "low" for faster/cheaper, "high" for dense images
            }},
        ]
    }]
)
print(response.choices[0].message.content)
```

**Anthropic vision** (same concepts, different shape):
```python
import anthropic, base64

client = anthropic.Anthropic()

with open("screenshot.png", "rb") as f:
    b64 = base64.standard_b64encode(f.read()).decode()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64}
            },
            {"type": "text", "text": "List all the error messages visible."}
        ]
    }]
)
```

**Cost note**: Images are billed as tokens. A 1024×1024 image at `detail=high` costs ~765 input tokens on OpenAI. Keep `detail=low` for thumbnails; use `high` only when small text or detail matters.

---

## 7. Debugging LLM Failures

Career-changers often have no mental model for *why* an LLM output went wrong. Here are the most common failure modes and how to diagnose and fix each.

### Failure 1: Refusals

The model refuses to help with a legitimate request.

```
Symptom: "I can't assist with that." on a benign task.
Cause:   Overly broad safety training; "dangerous" words in a safe context.

Diagnosis: Is your prompt using ambiguous framing?
  ❌ "How do I kill this process?"
  ✅ "How do I terminate a running OS process in Python?"

Fix:
  1. Reframe with explicit, specific context
  2. Add a system prompt: "You are a developer assistant for internal tools."
  3. Use Anthropic if OpenAI over-refuses for your domain (or vice versa)
```

### Failure 2: Confident Hallucination

The model states false facts with full confidence.

```
Symptom: Plausible-sounding answer that is factually wrong.
Cause:   Model fills gaps in its training data with "likely" completions.

Diagnosis: Ask the model to cite sources. Does it? Are they real?

Fix:
  1. Provide the facts in the prompt (RAG — Module 02)
  2. Add: "Only use information from the provided context. If uncertain, say so."
  3. Ask for confidence: "How confident are you? Rate 1-5."
  4. Add evaluation: cross-check key claims with a second LLM call
```

### Failure 3: Instruction Ignoring

The model ignores part of the system prompt.

```
Symptom: "Respond only in JSON" produces prose + JSON.
Cause:   Competing instructions; instruction buried in context middle;
         instruction is vague ("try to" / "when possible").

Diagnosis:
  - Is the instruction unambiguous? "ONLY JSON, no prose" vs "please use JSON"
  - Is it near the start or end of the system prompt? (Middle gets least attention)
  - Does a later turn contradict it?

Fix:
  1. Use imperative, absolute language: "ONLY return valid JSON. No other text."
  2. Move the instruction to the first or last line of the system prompt
  3. Repeat in the user turn: "...return as JSON only"
  4. Use native structured output (Method B/C above) — bypasses this entirely
```

### Failure 4: Output Truncation

Response cuts off mid-sentence.

```
Symptom: Response ends abruptly; finish_reason="length"
Cause:   Hit max_tokens limit.

Diagnosis: Check response.choices[0].finish_reason
  - "stop"   → natural end ✓
  - "length" → hit max_tokens cap — increase it

Fix:
  1. Increase max_tokens
  2. Ask for a shorter response: "In under 200 words, ..."
  3. For long outputs, stream and detect mid-sentence cuts
```

### Quick Debugging Checklist

```python
response = client.chat.completions.create(...)

# Always check these first
finish_reason = response.choices[0].finish_reason
tokens_used   = response.usage.total_tokens
content       = response.choices[0].message.content

if finish_reason == "length":
    print("⚠️  Truncated — increase max_tokens")
if not content or len(content.strip()) < 10:
    print("⚠️  Near-empty response — check for refusal")
if finish_reason == "content_filter":
    print("⚠️  Safety filter triggered — reframe prompt")
```

---

## 8. Reasoning Models & Effort Tuning

Modern models can "think" before answering — spending reasoning tokens to plan, consider alternatives, and solve multi-step problems. This is controlled through **reasoning effort** parameters.

### Reasoning Effort Levels

| Effort | Best for | Tradeoff |
|--------|----------|----------|
| `none` | Latency-critical: classification, retrieval, simple Q&A | Fastest, no reasoning overhead |
| `low` | Tool-use, planning, search, customer support | Modest latency increase |
| `medium` | Agentic coding, research, complex analysis | Balanced quality/speed |
| `high` | Debugging, deep planning, high-value tasks | Higher latency, more tokens |
| `xhigh` | Security review, deep research, enterprise workflows | Highest quality, highest cost |

### OpenAI Reasoning Models (GPT-5.x)

```python
from openai import OpenAI
client = OpenAI()

# Low effort — fast, efficient
response = client.responses.create(
    model="gpt-5.6",
    reasoning={"effort": "low"},
    input=[{"role": "user", "content": "Summarize this document."}]
)

# High effort with pro mode — maximum intelligence
response = client.responses.create(
    model="gpt-5.6",
    reasoning={"effort": "high", "mode": "pro"},
    input=[{"role": "user", "content": "Analyze this architecture for security flaws."}]
)

# View reasoning summary (not raw tokens)
response = client.responses.create(
    model="gpt-5.6",
    reasoning={"effort": "medium", "summary": "auto"},
    input=[{"role": "user", "content": "Solve this optimization problem."}]
)
for item in response.output:
    if item.type == "reasoning":
        print(f"[Reasoning]: {item.summary[0].text[:200]}...")
```

**Pro reasoning mode** (`reasoning.mode: "pro"`) performs more model work than standard, producing higher quality at increased cost. Useful for the hardest problems.

### Anthropic Adaptive Thinking

Claude models (Opus 4.7+, Sonnet 4.6+, Fable 5) use **adaptive thinking** — the model decides how much reasoning to apply based on task complexity.

```python
import anthropic
client = anthropic.Anthropic()

# Adaptive mode (default) — model decides budget
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=8000,
    thinking={"type": "adaptive"},
    messages=[{"role": "user", "content": "What's the optimal strategy for this problem?"}]
)

# Parse thinking and answer
for block in response.content:
    if block.type == "thinking":
        print(f"[Thinking]: {block.thinking[:200]}...")
    elif block.type == "text":
        print(f"[Answer]: {block.text}")

print(f"Thinking tokens: {response.usage.output_tokens_details.thinking_tokens}")
```

### Persisted Reasoning (Reasoning Context)

For multi-turn conversations with reasoning models, you can preserve reasoning across calls:

```python
# OpenAI — preserve reasoning context
first = client.responses.create(
    model="gpt-5.6",
    reasoning={"effort": "medium", "context": "current_turn"},
    input="Inspect this repository and identify the likely bug."
)

second = client.responses.create(
    model="gpt-5.6",
    previous_response_id=first.id,
    reasoning={"context": "all_turns"},  # reuse prior reasoning
    input="Now patch the bug and explain the change."
)
```

### Budget Guidelines

| Task complexity | Recommended budget |
|----------------|-------------------|
| Simple reasoning | 1,024 tokens (minimum) |
| Moderate complexity | 4,000–8,000 tokens |
| Hard problems | 16,000–32,000 tokens |
| Extremely complex | 32,000+ tokens (use Batch API) |

### When to Use Extended Thinking

| Use extended thinking | Use standard mode |
|----------------------|-------------------|
| Multi-step math/logic | Simple Q&A |
| Architecture decisions | Classification |
| Legal/scientific analysis | High-volume cheap calls |
| Plan-then-execute tasks | Latency-sensitive paths |
| Complex debugging | Summarization |

**Cost tradeoff**: Thinking tokens are billed at output token rates. Reserve for tasks where reasoning quality justifiably outweighs cost.

### Prompt Patterns That Work Well with Thinking

```
# Pattern 1: Plan-then-execute (thinking does the planning)
"First think through all the constraints and edge cases.
Then provide the implementation."

# Pattern 2: Multi-perspective analysis
"Consider at least 3 different approaches before recommending one.
Think about tradeoffs carefully."

# Pattern 3: Confidence-aware output
"If you're uncertain about any part of your answer,
note that uncertainty explicitly in your response."
```

---

## Next Steps

After mastering prompt engineering, move to Module 2: RAG Systems, where you'll learn how to augment LLMs with external knowledge.
