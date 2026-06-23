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

### 5. Structured Output
Requesting output in a specific format (JSON, XML, Markdown, etc.)

**Example:**
```
Extract the following information from the text and return as JSON:
- Name
- Age
- Occupation

Text: "John Smith is a 35-year-old software engineer..."
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

## 6. Extended Thinking / Reasoning Models

Some tasks benefit from the model spending time "thinking" before answering — especially multi-step math, logic problems, complex architecture decisions, and careful analysis.

### When to Use Extended Thinking

| Use extended thinking | Use standard mode |
|----------------------|-------------------|
| Multi-step math/logic | Simple Q&A |
| Architecture decisions | Classification |
| Legal/scientific analysis | High-volume cheap calls |
| Plan-then-execute tasks | Latency-sensitive paths |
| Complex debugging | Summarization |

**Cost tradeoff**: Thinking tokens are billed at output token rates (typically 5-10× input rates). A 16,000-token thinking block on Opus 4.8 costs ~$0.40 in thinking alone. Reserve for tasks where reasoning quality justifiably outweighs cost.

### Anthropic Extended Thinking

```python
import anthropic

client = anthropic.Anthropic()

# Adaptive mode (default on Claude Opus 4.7+, Sonnet 4.6+, Fable 5, Mythos 5)
# Model decides how much reasoning to apply based on complexity
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=8000,
    thinking={"type": "adaptive"},  # model decides budget
    messages=[{"role": "user", "content": "What's the optimal strategy for this optimization problem?"}]
)

# Explicit budget mode (Opus 4.5 and older)
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=8000,
    thinking={
        "type": "enabled",
        "budget_tokens": 4000  # min 1024; 16k+ for complex tasks
    },
    messages=[{"role": "user", "content": complex_problem}]
)

# Parse thinking and answer blocks
for block in response.content:
    if block.type == "thinking":
        print(f"[Thinking]: {block.thinking[:200]}...")
    elif block.type == "text":
        print(f"[Answer]: {block.text}")

# Track thinking token usage
print(f"Thinking tokens: {response.usage.output_tokens_details.thinking_tokens}")
```

### Budget Guidelines

| Task complexity | Recommended budget |
|----------------|-------------------|
| Simple reasoning | 1,024 tokens (minimum) |
| Moderate complexity | 4,000–8,000 tokens |
| Hard problems | 16,000–32,000 tokens |
| Extremely complex | 32,000+ tokens (use Batch API) |

**Key facts**:
- Budget is a target, not a hard cap — actual usage varies
- On Claude Fable 5, Mythos 5, Opus 4.8: thinking blocks appear between tool calls automatically (inter-tool reasoning)
- Requests above 32k thinking tokens should use the Batch API to avoid timeouts
- Track thinking tokens via `usage.output_tokens_details.thinking_tokens`

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
