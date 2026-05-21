# Module 5: LLM Deployment

## Overview

Deploying LLMs to production requires careful consideration of infrastructure, performance, cost, and user experience. This module covers the key aspects of putting LLMs into production.

## Deployment Options

### 1. Cloud APIs (Easiest)

**Providers:**
- OpenAI API
- Anthropic Claude API
- Google Vertex AI
- Azure OpenAI Service
- AWS Bedrock

**Pros:**
- No infrastructure management
- Automatic scaling
- Latest models always available
- Pay-per-use pricing

**Cons:**
- Vendor lock-in
- Data privacy concerns
- Less control over latency
- Ongoing costs can add up

**Best For:** MVPs, startups, non-sensitive data

### 2. Self-Hosted Open Source Models

**Popular Models:**
- Llama 2/3 (Meta)
- Mistral/Mixtral (Mistral AI)
- Falcon (TII)
- Gemma (Google)

**Deployment Platforms:**
- **Hugging Face Inference Endpoints**: Managed hosting
- **Replicate**: Serverless deployment
- **Modal**: Python-native cloud functions
- **Your own infrastructure**: Full control

**Pros:**
- Full data control
- Customizable
- No per-token costs
- Can fine-tune freely

**Cons:**
- Infrastructure management
- Need ML expertise
- Responsible for scaling
- Upfront setup time

**Best For:** Production systems, sensitive data, high volume

### 3. Hybrid Approach

Use cloud APIs for some tasks, self-hosted for others:
- Cloud: Complex reasoning, creative tasks
- Self-hosted: Simple Q&A, high-volume tasks

## Key Deployment Considerations

### 1. Latency Optimization

**Strategies:**
```
┌─────────────────────────────────────────────┐
│ Latency Optimization Techniques             │
├─────────────────────────────────────────────┤
│ • Use smaller models when possible          │
│ • Implement caching for repeated queries    │
│ • Stream responses (don't wait for full     │
│   generation)                               │
│ • Use speculative decoding                  │
│ • Optimize prompt length                    │
│ • Deploy closer to users (edge computing)   │
│ • Batch requests when possible              │
└─────────────────────────────────────────────┘
```

**Typical Latencies:**
- Cloud API: 500ms - 5s (depends on model size)
- Self-hosted 7B: 100-500ms (with proper hardware)
- Self-hosted 70B: 500ms - 2s

### 2. Cost Management

**Cost Factors:**
- Compute (GPU hours)
- Memory (VRAM requirements)
- API calls (if using cloud)
- Storage (model weights, cache)

**Optimization Strategies:**
1. **Right-size your model**: Don't use GPT-4 for simple tasks
2. **Implement caching**: Cache frequent queries
3. **Use quantization**: 4-bit or 8-bit models use less memory
4. **Batch requests**: Process multiple queries together
5. **Monitor usage**: Track cost per request

**Example Cost Comparison (per 1M tokens):**
| Model | Cost | Best For |
|-------|------|----------|
| GPT-4 | $30-60 | Complex reasoning |
| GPT-3.5 | $2-4 | General tasks |
| Llama 2 70B (self-hosted) | ~$5 | High volume |
| Llama 2 7B (self-hosted) | ~$0.50 | Simple tasks |

### 3. Scalability

**Scaling Strategies:**

```
Horizontal Scaling:
┌─────────┐     ┌───────────┐     ┌─────────┐
│  Load   │────▶│   Load    │────▶│ Model A │
│Balancer │     │ Balancer  │     └─────────┘
└─────────┘     └───────────┘     ┌─────────┐
                                  │ Model B │
                                  └─────────┘
                                        ...
                                  
Vertical Scaling:
- More powerful GPUs
- Multi-GPU setups
- GPU clusters
```

**Auto-scaling Triggers:**
- Request queue depth
- CPU/GPU utilization
- Response latency
- Time of day patterns

### 4. Security & Privacy

**Best Practices:**
1. **Data Encryption**: In transit (TLS) and at rest
2. **Access Control**: API keys, authentication, rate limiting
3. **Input Validation**: Sanitize user inputs
4. **Output Filtering**: Prevent harmful content
5. **Audit Logging**: Track all requests
6. **PII Detection**: Redact sensitive information
7. **Compliance**: GDPR, HIPAA, SOC2 as needed

**Prompt Injection Prevention:**
```python
def sanitize_input(user_input: str) -> str:
    # Remove potential injection attempts
    dangerous_patterns = [
        "ignore previous instructions",
        "you are now",
        "system message",
        "###"
    ]
    
    for pattern in dangerous_patterns:
        if pattern.lower() in user_input.lower():
            logger.warning(f"Potential injection attempt: {pattern}")
            return "Invalid input detected"
    
    return user_input
```

## Deployment Architectures

### Architecture 1: Simple API Wrapper

```
User → API Gateway → LLM Service → Response
```

**Use Case:** Simple chatbot, Q&A system

### Architecture 2: RAG Pipeline

```
User → API → Retrieval → Context + Query → LLM → Response
              ↑
         Vector DB
```

**Use Case:** Knowledge base, documentation search

### Architecture 3: Multi-Model Router

```
                    ┌→ Model A (small, fast)
User → Router ──────┼→ Model B (medium)
                    └→ Model C (large, capable)
```

**Use Case:** Optimize cost/performance ratio

### Architecture 4: Async Processing

```
User → Queue → Worker → LLM → Store Result → Notify User
```

**Use Case:** Long-running tasks, batch processing

## Hands-On Example

See `deployment_example.py` for a complete FastAPI service.

## Monitoring & Observability

**Key Metrics to Track:**
1. **Latency**: P50, P95, P99 response times
2. **Throughput**: Requests per second
3. **Error Rate**: Failed requests / total requests
4. **Token Usage**: Input/output tokens per request
5. **Cost**: Dollar cost per request
6. **Quality**: User ratings, task success rate

**Tools:**
- **Prometheus + Grafana**: Metrics visualization
- **Jaeger/Zipkin**: Distributed tracing
- **ELK Stack**: Log aggregation
- **LangSmith**: LLM-specific observability
- **Custom dashboards**: Business metrics

## Testing Before Deployment

**Checklist:**
- [ ] Load testing (can it handle peak traffic?)
- [ ] Stress testing (what breaks first?)
- [ ] Security testing (penetration testing)
- [ ] A/B testing (compare model versions)
- [ ] Canary deployment (gradual rollout)
- [ ] Rollback plan (how to revert quickly?)

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| No rate limiting | Implement request throttling |
| Ignoring token limits | Add input validation |
| No error handling | Graceful degradation |
| Not monitoring costs | Set up alerts |
| Single point of failure | Redundancy and failover |
| No versioning | Version your models and APIs |

## Next Steps

After deployment, move to Module 6: Optimization, where you'll learn advanced techniques for making your LLM applications faster, cheaper, and better.
