# Module 4: LLM Evaluation

## Why Evaluate LLMs?

Evaluation is critical for understanding model performance, identifying weaknesses, and making informed decisions about which models or approaches to use in production.

## Types of Evaluation

### 1. Automated Metrics

**Traditional NLP Metrics:**
- **Perplexity**: Measures how well the model predicts text (lower is better)
- **BLEU**: Compares generated text to reference translations
- **ROUGE**: Measures overlap between generated and reference summaries
- **BERTScore**: Uses BERT embeddings for semantic similarity

**Limitations:**
- Don't capture semantic meaning well
- Require reference answers
- Poor correlation with human judgment for open-ended tasks

### 2. Benchmark Datasets

**General Capability Benchmarks:**
| Benchmark | What It Tests | Format |
|-----------|---------------|--------|
| **MMLU** | Multi-task language understanding | Multiple choice |
| **GSM8K** | Grade school math | Free response |
| **HumanEval** | Code generation | Function completion |
| **BIG-Bench** | Diverse tasks | Various |
| **HellaSwag** | Common sense reasoning | Multiple choice |

**Domain-Specific Benchmarks:**
- **MedQA**: Medical questions
- **LegalBench**: Legal reasoning
- **FinQA**: Financial reasoning
- **SciQ**: Science questions

### 3. Human Evaluation

**What Humans Can Assess:**
- Helpfulness and relevance
- Factual accuracy
- Tone and style appropriateness
- Safety and bias
- Overall quality

**Methods:**
- Side-by-side comparisons (A/B testing)
- Rating scales (1-5 Likert)
- Preference ranking
- Error annotation

### 4. Model-Based Evaluation

Using a stronger LLM to evaluate outputs from another model:

```python
evaluation_prompt = """
Evaluate the following AI response on these criteria:
1. Accuracy: Is the information correct?
2. Relevance: Does it answer the question?
3. Clarity: Is it easy to understand?
4. Completeness: Does it cover all aspects?

Question: {question}
Response: {response}

Rate each criterion 1-5 and provide brief justification.
"""
```

**Popular Evaluation Models:**
- GPT-4 as a judge
- Claude for evaluation
- Specialized evaluators (Prometheus, Auto-Judge)

## Evaluation Frameworks

### 1. RAGAS (RAG Assessment)

Specifically for RAG systems:
- **Faithfulness**: Does the answer follow from the context?
- **Answer Relevance**: How relevant is the answer to the query?
- **Context Precision**: Is relevant information ranked higher?
- **Context Recall**: Did we retrieve all relevant information?

### 2. TruLens

- Track costs, latency, and quality
- Feedback functions for custom metrics
- Integration with major LLM providers

### 3. LangSmith (LangChain)

- Trace execution
- Compare runs
- Annotate and curate datasets
- A/B testing

### 4. Phoenix (Arize AI)

- Observability for LLMs
- Tracing and debugging
- Drift detection

### 5. DeepEval

- Unit tests for LLMs
- Pre-built metrics
- CI/CD integration

## Key Metrics to Track

### For Chatbots/Assistants:
1. **Response Quality**: Helpfulness, accuracy, relevance
2. **Safety**: Toxicity, bias, harmful content
3. **Engagement**: Conversation length, return rate
4. **Task Success**: Did the user accomplish their goal?

### For RAG Systems:
1. **Retrieval Quality**: Precision@K, Recall@K, MRR
2. **Generation Quality**: Faithfulness, answer relevance
3. **End-to-End**: Task success rate, user satisfaction

### For Content Generation:
1. **Coherence**: Logical flow, consistency
2. **Creativity**: Novelty, diversity
3. **Style Adherence**: Matches desired tone/format
4. **Factual Accuracy**: Hallucination rate

## Hands-On Example

See `evaluation_example.py` for practical evaluation code.

**Interactive notebook**: Open `evaluation.ipynb` for step-by-step walkthrough.

## Hands-On Exercises

1. **Metric Comparison**: Evaluate the same 5 model outputs using ROUGE, BLEU, and BERTScore. When do the metrics disagree? Which correlates best with your human judgment?

2. **Build a Golden Dataset**: Create 20 test cases for a customer support chatbot. Include 5 easy, 10 medium, and 5 hard cases. Label each with expected quality score.

3. **LLM-as-Judge Calibration**: Write an evaluation prompt for GPT-4 to score responses on a 1-5 scale. Run it on 10 examples and compare with your own ratings. What's the agreement rate?

4. **Red Team a Chatbot**: Spend 15 minutes trying to make a chatbot produce harmful, incorrect, or nonsensical outputs. Document every successful attack vector.

5. **A/B Test Design**: Design an A/B test comparing two prompt strategies. How many samples do you need for statistical significance at p=0.05? What metrics would you track?

---

## Advanced: LLM-as-Judge Systems

LLM-as-judge has become the standard for scalable evaluation of open-ended outputs. But naive implementations introduce systematic bias. Here's how to build reliable judge systems.

### Direct Scoring

```python
def llm_judge_direct(
    question: str,
    response: str,
    rubric: str,
    judge_model: str = "gpt-4o",
) -> dict:
    """
    Single-call direct scoring with a detailed rubric.
    """
    prompt = f"""You are an expert evaluator. Score the following response.

RUBRIC:
{rubric}

QUESTION: {question}

RESPONSE: {response}

Score each rubric dimension 1-5 and provide brief justification.
Output JSON: {{"dimensions": {{"dim_name": {{"score": int, "reason": str}}}}, "overall": int}}"""

    response_obj = client.chat.completions.create(
        model=judge_model,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(response_obj.choices[0].message.content)
```

### Pairwise Comparison (More Reliable Than Direct Scoring)

Pairwise comparison removes absolute calibration bias — the judge only needs to say which is better, not assign a number.

```python
import random

def pairwise_compare(
    question: str,
    response_a: str,
    response_b: str,
    judge_model: str = "gpt-4o",
) -> str:  # "A", "B", or "tie"
    """
    Compare two responses — winner is better.
    Randomize order to neutralize position bias.
    """
    # Randomize to prevent position bias (judges prefer first item)
    if random.random() < 0.5:
        first, second = response_a, response_b
        mapping = {"first": "A", "second": "B"}
    else:
        first, second = response_b, response_a
        mapping = {"first": "B", "second": "A"}

    prompt = f"""Which response better answers the question? Be objective.

QUESTION: {question}

RESPONSE 1: {first}

RESPONSE 2: {second}

Output JSON: {{"winner": "first"|"second"|"tie", "reason": str}}"""

    result = client.chat.completions.create(
        model=judge_model,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )
    data = json.loads(result.choices[0].message.content)
    winner_label = data.get("winner", "tie")
    return mapping.get(winner_label, "tie")


def tournament_eval(responses: list[str], question: str, rounds: int = 3) -> list:
    """Run round-robin tournament and rank by win rate."""
    wins = {i: 0 for i in range(len(responses))}

    for _ in range(rounds):
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                winner = pairwise_compare(question, responses[i], responses[j])
                if winner == "A": wins[i] += 1
                elif winner == "B": wins[j] += 1

    return sorted(wins.items(), key=lambda x: x[1], reverse=True)
```

### Evaluator Bias Mitigation

Known biases in LLM judges and how to address them:

| Bias | Description | Mitigation |
|------|-------------|------------|
| **Position bias** | Prefers first item in pairwise | Randomize order, run both orderings |
| **Length bias** | Prefers longer responses | Add "length is irrelevant" to rubric; normalize by length |
| **Verbosity bias** | Rewards confident-sounding prose | Use rubrics with concrete criteria, not "quality" |
| **Self-preference** | GPT-4 prefers GPT-4 outputs | Use a diverse panel of judges or a third model |
| **Anchoring** | First score anchors subsequent ones | Score each response independently |

```python
def debiased_judge(question: str, response: str, reference: str) -> dict:
    """
    Judge with explicit anti-bias instructions.
    """
    prompt = f"""Rate this response on the following criteria. 

IMPORTANT ANTI-BIAS RULES:
- Score based on factual accuracy and helpfulness, NOT length
- A short accurate answer is better than a long vague one
- Do not favor responses that sound confident but lack substance
- Score this response independently — ignore any prior scores you've given

QUESTION: {question}
REFERENCE ANSWER: {reference}
RESPONSE TO EVALUATE: {response}

Rate 1-5 on: accuracy, relevance, completeness, conciseness.
JSON: {{"accuracy": int, "relevance": int, "completeness": int, "conciseness": int, "overall": int}}"""

    result = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(result.choices[0].message.content)
```

### Rubric Calibration

Before using an LLM judge in production, calibrate it against human labels:

```python
def calibrate_judge(human_labeled_examples: list[dict]) -> dict:
    """
    Measure agreement between LLM judge and human labels.
    Returns calibration metrics.
    """
    agreements = []
    for example in human_labeled_examples:
        judge_score = llm_judge_direct(
            question=example["question"],
            response=example["response"],
            rubric=RUBRIC
        )
        human_score = example["human_score"]

        # Normalize both to 1-5 scale
        diff = abs(judge_score["overall"] - human_score)
        agreements.append(diff <= 1)  # within 1 point = agree

    agreement_rate = sum(agreements) / len(agreements)
    return {
        "agreement_rate": agreement_rate,
        "calibration_quality": "good" if agreement_rate > 0.75 else "poor",
        "n_examples": len(human_labeled_examples),
        "recommendation": (
            "Judge is reliable for production use" if agreement_rate > 0.75
            else "Refine rubric or use different judge model"
        )
    }
```

**Calibration rule of thumb**: 75%+ agreement on ±1 scale point = reliable for automation. Below 75%: refine rubric, add examples, or use human review as a gate.

---

## Evaluation Best Practices

1. **Define Clear Objectives**: What does "good" mean for your use case?
2. **Use Multiple Metrics**: No single metric tells the whole story
3. **Create Golden Datasets**: Curated test cases with known good answers
4. **Test Edge Cases**: Unusual inputs, adversarial examples
5. **Monitor in Production**: Real-world usage may differ from tests
6. **Track Over Time**: Performance can drift as usage patterns change
7. **Include Human Review**: Automated metrics have blind spots

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Only using automated metrics | Add human evaluation |
| Testing on training data | Maintain strict train/test split |
| Ignoring edge cases | Create adversarial test sets |
| One-time evaluation | Continuous monitoring |
| Not tracking baseline | Always compare to a reference |
| Optimizing for metrics | Focus on user outcomes |

## A/B Testing in Production

```
┌─────────┐     ┌──────────────┐     ┌──────────┐
│  Users  │────▶│   Router     │────▶│ Model A  │ (50%)
└─────────┘     │ (Random Split)│    └──────────┘
                │              │    ┌──────────┐
                └──────────────┘────▶│ Model B  │ (50%)
                                     └──────────┘
                                            │
                                            ▼
                                   ┌──────────────┐
                                   │   Compare    │
                                   │   Metrics    │
                                   └──────────────┘
```

**Key Considerations:**
- Statistical significance
- Sample size calculation
- Guardrails for bad models
- Gradual rollouts

## Red Teaming

Systematic testing for vulnerabilities:

1. **Prompt Injection**: Try to bypass safety measures
2. **Jailbreaking**: Attempt to make model ignore constraints
3. **Bias Testing**: Check for unfair treatment
4. **Privacy**: Test for information leakage
5. **Adversarial Examples**: Crafted inputs to cause failures

## Next Steps

After mastering evaluation, move to Module 5: Deployment, where you'll learn how to serve LLMs in production environments.
