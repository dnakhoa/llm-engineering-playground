# Module 05: Deployment
> **Why this matters:**  ## Learning Objectives - Cloud APIs vs self-hosti

## Learning Objectives
- Cloud APIs vs self-hosting tradeoffs
- Streaming and latency optimization
- Edge deployment with SLMs
- Cost management strategies

---
Deployment is where LLM apps meet reality — latency, cost, reliability, and user experience all collide. Getting this right means the difference between a demo and a product.


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

### 2. Streaming — The Most Important Latency Fix

Streaming returns tokens to the user as they're generated instead of waiting for the full response. For a 500-token response, this converts a 5-second wait into a response that starts in under 500ms.

**Why it matters**: Users abandon apps that feel slow. Streaming is the single biggest perceived-latency improvement for chat and generation features — and it costs nothing extra.

```python
from openai import OpenAI

client = OpenAI()

# Without streaming: user waits 3-8 seconds for full response
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a short poem about Python."}]
)
print(response.choices[0].message.content)   # whole response at once

# With streaming: user sees first tokens in <500ms
stream = client.chat.completions.create(
    model="gpt-4o",
    stream=True,   # ← the only change
    messages=[{"role": "user", "content": "Write a short poem about Python."}]
)

full_text = ""
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)   # print as it arrives
        full_text += delta

print()   # newline after stream ends
```

**Measuring Time-to-First-Token (TTFT)**:

```python
import time

def stream_with_ttft(prompt: str) -> dict:
    """Stream a response and measure TTFT and total latency."""
    start = time.time()
    first_token_time = None
    full_text = ""

    stream = client.chat.completions.create(
        model="gpt-4o", stream=True,
        messages=[{"role": "user", "content": prompt}]
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.time()
            full_text += delta

    end = time.time()
    return {
        "ttft_ms":    round((first_token_time - start) * 1000) if first_token_time else None,
        "total_ms":   round((end - start) * 1000),
        "chars":      len(full_text),
    }

result = stream_with_ttft("Explain the CAP theorem in 3 sentences.")
print(f"TTFT: {result['ttft_ms']}ms | Total: {result['total_ms']}ms")
```

**Streaming in FastAPI (Server-Sent Events)**:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(body: dict):
    """Stream LLM response as SSE to the client."""
    async def generate():
        stream = client.chat.completions.create(
            model="gpt-4o", stream=True,
            messages=body["messages"]
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                # SSE format: "data: <content>\n\n"
                yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

**Frontend JavaScript** to consume SSE:
```javascript
const es = new EventSource('/chat/stream');
es.onmessage = (e) => {
  if (e.data === '[DONE]') { es.close(); return; }
  document.getElementById('output').textContent += e.data;
};
```

---

### 3. Error Handling & Retry Patterns

Rate limit errors, timeouts, and transient failures are inevitable in production. Handle them gracefully or your app will be fragile.

```python
import time
import random
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

client = OpenAI()

def llm_call_with_retry(
    messages: list,
    model: str = "gpt-4o",
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """
    LLM call with exponential backoff + jitter.
    Retries on rate limits and transient errors.
    Fails fast on non-retriable errors (bad request, auth).
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=30.0,   # always set a timeout
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            # Respect Retry-After header if present
            retry_after = getattr(e, 'retry_after', None)
            wait = retry_after or (base_delay * (2 ** attempt) + random.uniform(0, 1))
            print(f"  Rate limited — waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)

        except APITimeoutError:
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"  Timeout — retrying in {wait:.1f}s")
            time.sleep(wait)

        except APIConnectionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))

        # Don't retry: BadRequestError (400), AuthenticationError (401), etc.

    raise RuntimeError("Max retries exceeded")


def llm_with_fallback(messages: list, primary: str = "gpt-4o", fallback: str = "gpt-4o-mini") -> str:
    """Try primary model; fall back to cheaper model on failure."""
    try:
        return llm_call_with_retry(messages, model=primary)
    except Exception as e:
        print(f"  Primary model failed ({e}), trying fallback: {fallback}")
        return llm_call_with_retry(messages, model=fallback)
```

**Graceful degradation hierarchy**:
```
1. Try primary model (e.g. gpt-4o)
2. On failure → try fallback model (e.g. gpt-4o-mini)
3. On failure → try cached response if available
4. On failure → return a helpful error message (never expose stack traces)
```

> **Note on caching**: Response caching (semantic cache, prompt caching) is covered in Module 06, which focuses specifically on cost and performance optimization. This module covers the serving infrastructure; Module 06 covers the optimization layer on top.

---

### 4. Cost Management

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

**Interactive notebook**: Open `deployment.ipynb` for step-by-step walkthrough.


## Resources
- [vLLM](https://github.com/vllm-project/vllm)
- [MLC LLM](https://github.com/mlc-ai/mlc-llm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Hands-On Exercises

1. **Latency Benchmark**: Run 100 identical requests to your LLM API and measure p50, p95, p99 latencies. What's the variance? What causes spikes?

2. **Caching Strategy**: Implement exact-match caching for a Q&A endpoint. Then upgrade to semantic caching. Measure the hit rate difference on a realistic query set.

3. **Cost Calculator**: Build a function that takes model name, input tokens, and output tokens and returns the cost. Test it against 3 different models and 3 different use cases.

4. **Streaming Response**: Modify the FastAPI endpoint to stream tokens back to the client using Server-Sent Events. Measure the improvement in time-to-first-token.

5. **Graceful Degradation**: Design a fallback strategy: if the primary model is down, route to a backup model. If that's also down, return a cached response. If no cache, return a helpful error message. Implement it.

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



## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| High TTFT (time to first token) | Use streaming, reduce context length, try faster model |
| Rate limit errors (429) | Implement exponential backoff with jitter |
| Streaming cuts off mid-response | Check `finish_reason`; "length" means hit max_tokens |
| Ollama connection refused | Start Ollama first: `ollama serve` |

## 📚 Resources

- [vLLM](https://github.com/vllm-project/vllm) — high-throughput LLM serving
- [MLC LLM](https://github.com/mlc-ai/mlc-llm) — edge deployment framework
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — CPU inference

## Next Steps

After deployment, move to Module 6: Optimization, where you'll learn advanced techniques for making your LLM applications faster, cheaper, and better.


---

## Edge Deployment — LLMs on Devices

Running LLMs on edge devices (phones, browsers, embedded systems) is a growing trend. SLMs (Small Language Models) make this practical.

### Why Edge?

| Benefit | Description |
|---------|-------------|
| **Privacy** | Data never leaves the device |
| **Latency** | No network round-trip |
| **Offline** | Works without internet |
| **Cost** | No API fees after deployment |

### Frameworks

| Framework | Platform | Key Feature |
|-----------|----------|-------------|
| **MLC LLM** | Browser, Android, iOS | Compile to WebAssembly, runs in browser |
| **mnn-llm** | Mobile (Android/iOS) | Alibaba's lightweight inference engine |
| **llama.cpp** | CPU (any platform) | GGUF format, runs on anything |
| **Ollama** | Desktop | Local server, easy API |
| **ONNX Runtime** | Cross-platform | Optimized for specific hardware |

### SLMs for Edge

| Model | Parameters | Best For |
|-------|-----------|----------|
| Phi-3 Mini | 3.8B | General assistant on mobile |
| Gemma 2B | 2B | Fast, Google-optimized |
| Llama 3.2 1B | 1B | Ultra-lightweight |
| SmolLM2 | 1.3B | Hugging Face's tiny model |
| Qwen2-0.5B | 0.5B | Chinese + English |

### Deployment Pattern

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Train/    │────▶│  Quantize    │────▶│  Deploy to  │
│   Fine-tune │     │  (GGUF/ONNX) │     │  Edge       │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │ 4-bit quant │
                    │ (1.5B → ~1GB)│
                    └─────────────┘
```

### When to Use Edge vs Cloud

```
Use edge when:                    Use cloud when:
─────────────                     ────────────────
Privacy is critical               Need frontier model quality
Offline capability needed         Complex reasoning tasks
Low latency required              Large context windows
Cost sensitivity (high volume)    Rapid iteration/prototyping
Simple tasks (classification,     Multi-step agent workflows
 extraction, summarization)
```

