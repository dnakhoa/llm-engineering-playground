# Module 3: Fine-Tuning LLMs

> ⚠️ **Read this first.** Fine-tuning is powerful but almost always the wrong first move. Most teams that fine-tune when they shouldn't waste weeks and thousands of dollars. Work through the decision tree below before writing any training code.

---

## Should You Fine-Tune? — Decision Tree

```
START: You want your LLM to do something it doesn't do well today.
          │
          ▼
┌─────────────────────────────────────────────┐
│ Have you tried a well-crafted system prompt │
│ with 3–5 few-shot examples?                 │
└─────────────────────────────────────────────┘
          │ NO → Do this first. It's free and takes 1 hour.
          │ YES ↓
          ▼
┌─────────────────────────────────────────────┐
│ Is the problem missing KNOWLEDGE            │
│ (facts the model doesn't know, recent info, │
│ your proprietary docs)?                     │
└─────────────────────────────────────────────┘
          │ YES → Use RAG (Module 02). Fine-tuning doesn't add facts reliably.
          │ NO ↓
          ▼
┌─────────────────────────────────────────────┐
│ Is the problem STYLE or FORMAT?             │
│ (tone, length, schema, specific phrasing)   │
└─────────────────────────────────────────────┘
          │ YES — is it consistent and describable in a prompt?
          │       → Yes: prompt engineering or structured output first
          │       → No / needs > 5 pages of examples: consider fine-tuning
          │ NO ↓
          ▼
┌─────────────────────────────────────────────┐
│ Do you have > 100 high-quality labeled      │
│ examples you can curate?                    │
└─────────────────────────────────────────────┘
          │ NO → You don't have enough data. Collect first.
          │ YES ↓
          ▼
┌─────────────────────────────────────────────┐
│ Do you have a strong eval that will tell    │
│ you if fine-tuning helped or hurt?          │
└─────────────────────────────────────────────┘
          │ NO → Build eval first (Module 04). Blind fine-tuning is gambling.
          │ YES ↓
          ▼
       Fine-tuning is probably the right tool. Continue below.
```

### When fine-tuning IS the right answer

| Signal | Example |
|--------|---------|
| Consistent output format that prompting can't enforce | Always return a specific XML schema with 10 fields |
| Domain-specific style that requires > 50 examples | Legal writing in a specific firm's voice |
| Latency/cost sensitive at high volume | 10M calls/day where a smaller fine-tuned model replaces a large one |
| Task where the base model genuinely lacks capability | Specialized medical coding (ICD-10) with proprietary taxonomy |

### When fine-tuning is NOT the answer

- ❌ "The model doesn't know our company's data" → Use RAG
- ❌ "We want it to sound more professional" → System prompt + few-shot
- ❌ "We have 20 examples" → Not enough; collect more or use few-shot
- ❌ "GPT-4 does it, we want a cheaper model" → Try gpt-4o-mini first; it's 16× cheaper
- ❌ "We want to add new knowledge from 2025" → RAG or use a newer model

---

## What is Fine-Tuning?

Fine-tuning is the process of adapting a pre-trained LLM to specific tasks or domains by continuing training on specialized data. Unlike prompt engineering (which changes inputs) or RAG (which adds context), fine-tuning actually modifies the model's weights.

## Approach Comparison

## When to Fine-Tune vs Other Approaches

| Approach | Best For | Cost | Complexity |
|----------|----------|------|------------|
| **Prompt Engineering** | General tasks, quick iteration | $ | Low |
| **RAG** | Domain knowledge, up-to-date info | $$ | Medium |
| **Fine-Tuning** | Style adaptation, task specialization | $$$ | High |

## Types of Fine-Tuning

### 1. Full Fine-Tuning
- Update all model parameters
- Requires significant GPU memory
- Risk of catastrophic forgetting

### 2. Parameter-Efficient Fine-Tuning (PEFT)
Only update a small subset of parameters:

#### LoRA (Low-Rank Adaptation)
- Inject trainable rank decomposition matrices
- Freeze original weights
- 10,000x fewer parameters to train
- Most popular method today

#### QLoRA (Quantized LoRA)
- LoRA + 4-bit quantization
- Can fine-tune 7B models on single GPU
- Near full fine-tuning performance

#### Prefix Tuning / Prompt Tuning
- Add learnable tokens to input
- Freeze entire model
- Minimal memory footprint

## Fine-Tuning Use Cases

1. **Style Adaptation**: Match company tone, brand voice
2. **Task Specialization**: Medical diagnosis, legal analysis
3. **Format Compliance**: Always output specific JSON schema
4. **Domain Vocabulary**: Technical jargon, industry terms
5. **Multilingual Support**: Extend to underrepresented languages
6. **Instruction Following**: Better adherence to complex instructions

## The Fine-Tuning Process

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Data       │────▶│   Training   │────▶│  Evaluation  │
│ Preparation  │     │   Loop       │     │  & Testing   │
└──────────────┘     └──────────────┘     └──────────────┘
       ▲                                       │
       │                                       ▼
       │                                ┌──────────────┐
       └────────────────────────────────│  Deployment  │
                                        └──────────────┘
```

### Step 1: Data Preparation

**Quality Requirements:**
- Clean, high-quality examples
- Consistent formatting
- Balanced representation
- No sensitive/PII data

**Data Formats:**
```json
// Instruction-following format
{
  "instruction": "Summarize this article",
  "input": "Article text here...",
  "output": "Summary here..."
}

// Completion format
{
  "text": "Q: What is AI?\nA: Artificial Intelligence is..."
}

// Chat format
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

### Step 2: Choose Base Model

**Popular Choices:**
- **Llama 2/3** (Meta): General purpose, good performance
- **Mistral** (Mistral AI): Efficient, strong reasoning
- **Phi** (Microsoft): Small but capable
- **Falcon** (TII): Open weights, permissive license
- **Gemma** (Google): Lightweight, well-documented

### Step 3: Select Training Framework

**Options:**
- **Hugging Face Transformers**: Most flexible, full control
- **PEFT Library**: Easy LoRA/QLoRA implementation
- **Axolotl**: Production-ready fine-tuning pipeline
- **LLaMA Factory**: Web UI, multiple model support
- **Unsloth**: Optimized for speed and memory

### Step 4: Configure Training

**Key Hyperparameters:**
```python
training_args = {
    "learning_rate": 2e-4,        # Start with 1e-4 to 3e-4
    "num_train_epochs": 3,         # Usually 1-5 epochs
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "save_strategy": "epoch",
    "logging_steps": 10,
}
```

### Step 5: Evaluate

**Evaluation Methods:**
- Hold-out test set
- Perplexity metrics
- Task-specific benchmarks
- Human evaluation
- A/B testing in production

## Hands-On Example

See `finetune_example.py` for a complete LoRA fine-tuning script.

**Interactive notebook**: Open `fine_tuning.ipynb` for step-by-step walkthrough.

## Hands-On Exercises

1. **Data Format Conversion**: Take 10 Q&A pairs and convert them to Alpaca instruction format, ChatML format, and completion format. What are the trade-offs?

2. **Hyperparameter Sweep**: Run `finetune_example.py` with 3 different learning rates (1e-5, 2e-4, 1e-3). Which produces the best results and why?

3. **LoRA Rank Experiment**: Change the LoRA rank from 8 to 32 and 64. How does rank affect training time, model size, and quality?

4. **When NOT to Fine-Tune**: Given these 5 scenarios, decide whether to use prompt engineering, RAG, or fine-tuning for each:
   - Company wants its chatbot to always respond in a pirate accent
   - Legal team needs answers grounded in their 10,000-page policy manual
   - Startup needs to ship a Q&A bot by Friday
   - Hospital wants a model that understands rare disease terminology
   - E-commerce site needs product descriptions in a specific format

5. **Dataset Quality Audit**: Download a public fine-tuning dataset from HuggingFace. Find 3 examples with issues (duplicates, contradictory instructions, PII) and explain how you'd fix them.

## Common Pitfalls

1. **Overfitting**: Model memorizes training data
   - Solution: Add regularization, reduce epochs, increase dropout

2. **Catastrophic Forgetting**: Loses general capabilities
   - Solution: Mix general data, use PEFT methods

3. **Data Leakage**: Test data in training set
   - Solution: Strict train/test separation

4. **Insufficient Data**: Too few examples
   - Solution: At least 100-1000 quality examples per task

5. **Wrong Learning Rate**: Too high/low
   - Solution: Learning rate sweeps, start conservative

## Cost Considerations

**Training Costs (Approximate):**
- 7B model with QLoRA: ~$10-50 on cloud GPUs
- 70B model with QLoRA: ~$100-500
- Full fine-tuning: 10-100x more expensive

**Inference Costs:**
- Fine-tuned models may need larger instances
- Consider distillation for production

## Best Practices

1. ✅ Start with prompt engineering first
2. ✅ Try RAG before fine-tuning
3. ✅ Use PEFT (LoRA/QLoRA) unless you have specific needs
4. ✅ Curate high-quality training data
5. ✅ Monitor for overfitting during training
6. ✅ Evaluate on held-out test set
7. ✅ Version your models and datasets
8. ✅ Document training configuration

## Next Steps

After fine-tuning, move to Module 4: Evaluation, where you'll learn how to measure and improve LLM performance systematically.
