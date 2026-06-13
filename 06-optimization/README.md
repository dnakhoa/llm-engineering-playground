# Module 6: LLM Optimization

## Overview

Optimization is about making your LLM applications faster, cheaper, and more efficient without sacrificing quality. This module covers advanced techniques used in production systems.

## Optimization Areas

### 1. Model Optimization

#### Quantization
Reduce precision of model weights to save memory and speed up inference.

| Precision | Memory Usage | Speed | Quality Loss |
|-----------|-------------|-------|--------------|
| FP32 (Full) | 100% | 1x | None |
| FP16 | 50% | 2-3x | Minimal |
| INT8 | 25% | 3-4x | Small |
| INT4 | 12.5% | 4-5x | Noticeable but acceptable |

**Tools:**
- **bitsandbytes**: Easy 4/8-bit quantization
- **GPTQ**: GPU-optimized quantization
- **AWQ**: Activation-aware weight quantization
- **GGUF**: CPU-friendly format (llama.cpp)

#### Pruning
Remove unnecessary weights/neurons from the model.
- Structured pruning: Remove entire layers/heads
- Unstructured pruning: Remove individual weights
- Can achieve 2-10x compression

#### Distillation
Train a smaller "student" model to mimic a larger "teacher".
- Example: DistilBERT (40% smaller, 60% faster than BERT)
- Example: TinyLlama (1.1B trained to match larger models)

### 2. Inference Optimization

#### Speculative Decoding
Use a small draft model to propose tokens, large model verifies.
```
Draft Model (fast):  "The quick brown fox jumps..."
Large Model (verify): ✓ ✓ ✓ ✓ ✓ ...
Result: 2-3x speedup with same quality
```

#### KV Cache Optimization
Reuse key-value caches across requests with similar prefixes.
- Useful for system prompts, few-shot examples
- PagedAttention (vLLM): Efficient memory management

#### Batch Processing
Process multiple requests together to maximize GPU utilization.
- Static batching: Wait to collect N requests
- Dynamic batching: Process as many as fit in memory
- Continuous batching: Start new requests as others finish

### 3. Prompt Optimization

#### Token Efficiency
Every token costs money and latency. Optimize your prompts:

**Before (45 tokens):**
```
I would like you to please help me write a function that can calculate the factorial of a number
```

**After (15 tokens):**
```
Write a factorial function
```

#### Prompt Compression
Techniques to reduce prompt length while preserving meaning:
- Remove filler words
- Use abbreviations
- Compress examples
- Use structured formats

#### Context Window Management
When hitting context limits:
1. **Summarization**: Compress old conversation turns
2. **Sliding Window**: Keep only recent context
3. **RAG**: Retrieve only relevant parts
4. **Map-Reduce**: Process in chunks, combine results

### 4. Caching Strategies

#### Semantic Caching
Cache responses based on semantic similarity, not exact match.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_cache(query, cache, threshold=0.9):
    query_embedding = model.encode(query)
    
    for cached_query, cached_response in cache.items():
        cached_embedding = model.encode(cached_query)
        similarity = cosine_similarity(query_embedding, cached_embedding)
        
        if similarity > threshold:
            return cached_response  # Return cached response
    
    return None  # No cache hit
```

**Benefits:**
- 30-70% cost reduction for repetitive queries
- Faster response times
- Consistent answers

#### Multi-Level Caching
```
┌─────────────────────────────────────┐
│ Level 1: Exact Match Cache (Redis)  │ → 1ms
├─────────────────────────────────────┤
│ Level 2: Semantic Cache (FAISS)     │ → 10ms
├─────────────────────────────────────┤
│ Level 3: LLM Generation             │ → 500-5000ms
└─────────────────────────────────────┘
```

### 5. Cost Optimization

#### Model Routing
Route requests to appropriate models based on complexity:

```python
def route_request(query):
    if is_simple_greeting(query):
        return "tiny-model"  # $0.10/1M tokens
    elif requires_reasoning(query):
        return "medium-model"  # $2/1M tokens
    elif is_complex_task(query):
        return "large-model"  # $30/1M tokens
```

**Savings:** 50-80% cost reduction

#### Response Length Control
Limit max_tokens based on expected response type:
- Greeting: 20 tokens
- Simple Q&A: 100 tokens
- Complex explanation: 500 tokens

#### Early Stopping
Stop generation when:
- Answer is complete
- Confidence is low (avoid wasting tokens)
- User seems satisfied

### 6. Performance Monitoring

**Key Metrics:**
```
Latency:
  - Time to first token (TTFT): < 500ms target
  - Total generation time
  - Tokens per second

Quality:
  - User satisfaction scores
  - Task completion rate
  - Error rate

Cost:
  - Cost per request
  - Cost per successful task
  - Token efficiency ratio
```

## Hands-On Example

See `optimization_example.py` for practical optimization code.

**Interactive notebook**: Open `optimization.ipynb` for step-by-step walkthrough.

## Hands-On Exercises

1. **Prompt Compression**: Take a 200-token prompt and reduce it to under 50 tokens while preserving the same quality output. Measure the cost savings.

2. **Semantic Cache Implementation**: Build a semantic cache using sentence-transformers and cosine similarity. Test it with 20 semantically similar but syntactically different queries. What's your hit rate at threshold 0.85 vs 0.95?

3. **Model Router**: Build a simple router that classifies queries as "simple", "medium", or "complex" using keyword matching or a small model. Route simple queries to gpt-4o-mini, complex to gpt-4o. Estimate the cost savings.

4. **Token Budget**: Given a 4000-token context window, allocate tokens across: system prompt (10%), conversation history (40%), retrieved documents (40%), response (10%). Build a function that trims content to fit each budget.

5. **Batch vs Streaming**: Compare batch processing (collect 10 requests, process together) vs streaming (process immediately). Which has lower latency? Which has higher throughput?

## Best Practices Checklist

### Quick Wins (1-2 days)
- [ ] Implement response caching
- [ ] Optimize prompt templates
- [ ] Set appropriate max_tokens limits
- [ ] Add request logging

### Medium Term (1-2 weeks)
- [ ] Implement semantic caching
- [ ] Add model routing
- [ ] Quantize models where possible
- [ ] Set up monitoring dashboards

### Long Term (1-2 months)
- [ ] Fine-tune smaller models for your tasks
- [ ] Implement speculative decoding
- [ ] Build custom evaluation pipelines
- [ ] Optimize infrastructure (GPU selection, scaling)

## Common Pitfalls

| Pitfall | Impact | Solution |
|---------|--------|----------|
| No caching | 3-10x higher costs | Implement multi-level caching |
| Over-sized models | Wasted compute | Right-size for tasks |
| Verbose prompts | Higher latency, cost | Prompt optimization |
| No monitoring | Blind to issues | Add comprehensive metrics |
| Ignoring batch size | Poor GPU utilization | Tune batch sizes |
| One model fits all | Suboptimal cost/quality | Model routing |

## Tools & Libraries

**Inference Optimization:**
- **vLLM**: High-throughput serving with PagedAttention
- **TGI** (Text Generation Inference): Hugging Face's optimized server
- **llama.cpp**: CPU inference for Llama models
- **TensorRT-LLM**: NVIDIA's optimized inference

**Quantization:**
- **bitsandbytes**: Easy 4/8-bit quantization
- **AutoGPTQ**: GPU-optimized quantization
- **GGML/GGUF**: CPU-friendly formats

**Monitoring:**
- **LangSmith**: End-to-end LLM platform
- **Arize Phoenix**: Observability and debugging
- **Weights & Biases**: Experiment tracking

## Next Steps

**Next:** Move to Module 7: Agentic Workflows, where you'll learn how to build multi-agent collaborative systems with LangGraph.

**Continue Your Journey:**
1. Build a complete project combining all concepts
2. Contribute to open-source LLM projects
3. Stay updated with latest research (arXiv, Twitter)
4. Join communities (Discord, Reddit r/LocalLLaMA)
5. Experiment with new models and techniques

Remember: LLM engineering is rapidly evolving. What's cutting-edge today may be standard tomorrow. Keep learning and building!
