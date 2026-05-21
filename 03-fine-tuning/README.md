# Module 3: Fine-Tuning LLMs

## What is Fine-Tuning?

Fine-tuning is the process of adapting a pre-trained LLM to specific tasks or domains by continuing training on specialized data. Unlike prompt engineering (which changes inputs) or RAG (which adds context), fine-tuning actually modifies the model's weights.

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
