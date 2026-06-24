# LLM Engineering Playground

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-blue)](https://www.kaggle.com/)

> **The only open-source course that covers the full production lifecycle of LLM engineering** — from foundations through context engineering, agent harness, MCP, guardrails, and eval ops.

## Why This Course?

| What you learn | Microsoft (112k⭐) | awesome-llm-apps (115k⭐) | This course |
|----------------|-------------------|--------------------------|-------------|
| Context Engineering (U-curve, caching) | ❌ | ❌ | ✅ Module 12 |
| Agent Harness (crash-proof resume) | ❌ | ❌ | ✅ Module 13 |
| EvalOps (CI/CD for LLMs) | ❌ | ❌ | ✅ Module 9 |
| LLM Ops (tracing, drift detection) | Partial | ❌ | ✅ Module 8 |
| MCP Tool Design | Separate course | Templates only | ✅ Module 14 |
| Gateway & Guardrails | One lesson | ❌ | ✅ Module 10 |
| Multi-provider (6+ providers) | Azure-only | Varies | ✅ All modules |
| **Total coverage** | 21 lessons, ~12h | 100+ standalone demos | **15 modules, ~30h** |

## Quick Start

```bash
git clone https://github.com/dnakhoa/llm-engineering-playground.git
cd llm-engineering-playground
pip install -r requirements.txt
cp .env.example .env    # add ONE API key (OpenAI, Anthropic, DeepSeek, Ollama...)
python demo.py          # see the full pipeline in 60 seconds
```

The demo auto-detects your LLM provider and runs RAG + guardrails + caching + observability — no server needed.

## Run the Demo Locally

### Prerequisites

- Python 3.10+
- An API key from one of:
  | Provider | Env Variable | Cost |
  |----------|-------------|------|
  | OpenAI | `OPENAI_API_KEY` | Pay-as-you-go |
  | Anthropic | `ANTHROPIC_API_KEY` | Pay-as-you-go |
  | DeepSeek | `DEEPSEEK_API_KEY` | Very cheap |
  | xAI Grok | `GROK_API_KEY` | Pay-as-you-go |
  | Qwen | `QWEN_API_KEY` | Cheap |
  | Ollama | `OPENAI_BASE_URL=http://localhost:11434/v1` | Free (local) |

### Step 1: Clone and install

```bash
git clone https://github.com/dnakhoa/llm-engineering-playground.git
cd llm-engineering-playground
python3 -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install -r requirements.txt
```

### Step 2: Set your API key

```bash
cp .env.example .env
# Edit .env and uncomment + fill in ONE provider key, e.g.:
# OPENAI_API_KEY=sk-your-key-here
```

### Step 3: Run the 60-second demo

```bash
python demo.py
```

This runs a self-contained pipeline that shows:
1. **Provider auto-detection** — finds your key and selects the model
2. **RAG retrieval** — embeds 6 docs, retrieves relevant chunks for a query
3. **Input guardrails** — detects and blocks prompt injection attempts
4. **Semantic caching** — demonstrates cache hit on similar queries
5. **Output guardrails** — filters PII (SSNs, credit cards) from responses
6. **Cost tracking** — estimates token usage and dollar cost

No server, no database — just `python demo.py`.

### Step 4: Run the full capstone (optional)

```bash
cd capstone
pip install -r requirements.txt
python seed_knowledge.py          # seed ChromaDB with 10 LLM topics
python ui.py                      # Gradio web UI at http://localhost:7860
```

Or with Docker:
```bash
cd capstone
docker compose up                 # starts API on :8000 + UI on :7860
```

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'dotenv'` | Run `pip install python-dotenv` |
| `openai.AuthenticationError` | Check your `.env` has a valid API key |
| `chromadb` install fails on Mac | Use `pip install chromadb --no-cache-dir` |
| Ollama connection refused | Start Ollama first: `ollama serve` |

## Curriculum Structure

**15 Comprehensive Modules** covering the complete LLM engineering lifecycle:

| Module | Topic | Key Focus | Time | Status |
|--------|-------|-----------|------|--------|
| 00 | LLM Foundations | Tokens, embeddings, context, sampling, cost | ~1h | ✅ |
| 01 | Prompt Engineering | Foundation communication + extended thinking | ~1.5h | ✅ |
| 02 | RAG Systems | Knowledge augmentation | ~2h | ✅ |
| 03 | Fine-Tuning | Model adaptation | ~2h | ✅ |
| 04 | Evaluation | Quality assurance + LLM-as-judge | ~2h | ✅ |
| 05 | Deployment | Production serving | ~2h | ✅ |
| 06 | Optimization | Performance, efficiency & prompt caching | ~2h | ✅ |
| 07 | Agentic Workflows | Multi-agent systems + supervisor/swarm | ~3.5h | ✅ |
| 08 | LLM Ops | Observability & monitoring | ~2h | ✅ |
| 09 | EvalOps | Continuous evaluation | ~1.5h | ✅ |
| 10 | Gateway & Guardrails | Security & compliance | ~2h | ✅ |
| 11 | Memory & Context | Persistent intelligence | ~2h | ✅ |
| 12 | Context Engineering | Window anatomy, caching, compression | ~2h | ✅ |
| 13 | Agent Harness | Loop engineering, durable journals, repair | ~3h | ✅ |
| 14 | MCP & Tool Design | Model Context Protocol, tool interfaces | ~2h | ✅ |
| 15 | Multimodal | Vision, image generation, audio, CLIP | ~2h | ✅ |

### 🧱 Module 0: LLM Foundations
**Start Here** - The mental models every LLM engineer needs
- What tokens are (and why the model can't count characters)
- Embeddings as geometry — why semantic search works
- Context window anatomy: input tokens, output tokens, limits
- Sampling parameters: temperature, top-p, max_tokens
- API call anatomy: system prompt, user, assistant, parameters
- Model selection: when to use small/medium/large
- Cost estimation before you build

📁 Location: `00-llm-foundations/`

### 📚 Module 1: Prompt Engineering
**Foundation** - Learn to communicate effectively with LLMs
- Zero-shot and few-shot prompting
- Chain-of-thought reasoning
- Role prompting and personas
- Structured output generation
- Best practices and patterns

📁 Location: `01-prompt-engineering/`

### 🔍 Module 2: RAG Systems
**Knowledge Augmentation** - Connect LLMs to external data
- Vector databases and embeddings
- Document chunking strategies
- Retrieval techniques
- Building complete RAG pipelines
- Advanced patterns (HyDE, re-ranking)

📁 Location: `02-rag-systems/`

### 🎯 Module 3: Fine-Tuning
**Model Adaptation** - Customize LLMs for your needs
- When to fine-tune vs. other approaches
- LoRA and QLoRA techniques
- Data preparation
- Training frameworks
- Evaluation and deployment

📁 Location: `03-fine-tuning/`

### 📊 Module 4: Evaluation
**Quality Assurance** - Measure and improve performance
- Automated metrics (BLEU, ROUGE)
- Benchmark datasets
- Human evaluation methods
- Model-based evaluation
- A/B testing frameworks

📁 Location: `04-evaluation/`

### 🚀 Module 5: Deployment
**Production Ready** - Serve LLMs at scale
- Cloud APIs vs. self-hosting
- Latency optimization
- Cost management
- Security and privacy
- Monitoring and observability

📁 Location: `05-deployment/`

### ⚡ Module 6: Optimization
**Performance & Efficiency** - Make it faster and cheaper
- Quantization and pruning
- Caching strategies
- Model routing
- Prompt optimization
- Token budgeting

📁 Location: `06-optimization/`

### 🤖 Module 7: Agentic Workflows
**Autonomous Systems** - Build multi-agent collaborative systems
- LangChain agents and tools
- LangGraph state machines
- Specialist agent design
- Dynamic routing and orchestration
- Human-in-the-loop workflows
- Production patterns

📁 Location: `07-agentic-workflows/`

### 📈 Module 8: LLM Ops & Observability
**Production Monitoring** - Track, measure, and improve in production
- Distributed tracing for LLM calls
- Cost tracking and budget alerts
- Quality monitoring and drift detection
- Prompt versioning and A/B testing
- Feedback loop pipelines
- Dashboard metrics and alerting

📁 Location: `08-llmops-observability/`

### 🧪 Module 9: EvalOps (Evaluation Operations)
**Automated Quality Assurance** - Continuous evaluation at scale
- Automated eval pipelines
- Synthetic test data generation
- CI/CD integration for LLMs
- Regression testing frameworks
- Continuous production evaluation
- Adversarial testing

📁 Location: `09-eval-ops/`

### 🧠 Module 10: API Gateway & Guardrails
**Security & Compliance** - Production protection layer
- Authentication and authorization (JWT, OAuth)
- Rate limiting and quota management
- Input validation and prompt injection detection
- Content filtering and moderation
- PII detection and redaction
- Output validation and hallucination checks
- Compliance logging and audit trails
- Monitoring metrics and alerting

📁 Location: `10-gateway-guardrails/`

### 🧠 Module 12: Context Engineering
**Context as a Resource** - Design what enters the context window, not just fill it
- Context window anatomy and the U-shaped attention curve
- Context poisoning, confusion, and clash — failure modes and fixes
- Observation masking — compress tool outputs before they hit context
- Provider-level prefix caching (Anthropic cache_control, OpenAI auto-cache)
- Token budget management and per-slot allocation
- Context compression: sliding window, LLM-based summarization

📁 Location: `12-context-engineering/`

### 🔄 Module 13: Agent Harness & Loop Engineering
**Reliable Autonomy** - Build agents that don't get stuck, crash, or over-run
- Loop-until-dry with novelty gates (exhaustive without infinite loops)
- Budget-aware loops — token budget injected into each agent call
- Durable journals — crash-proof resume without replaying completed steps
- Self-repair loops — retry with error context fed back to the agent
- Human approval checkpoints for irreversible actions
- Deterministic orchestration vs model-driven autonomy
- Pipeline vs barrier synchronization, adversarial verification

📁 Location: `13-agent-harness/`

### 🔌 Module 14: MCP & Tool Design
**Tool Interfaces** - Build tools agents can actually use correctly
- Model Context Protocol (MCP spec 2025-06-18) — architecture and primitives
- Building MCP servers in Python with FastMCP (stdio + HTTP transport)
- Tool descriptions as routing signals — what makes an agent call the right tool
- Schema design to eliminate ambiguous arguments
- Error handling: protocol vs business-logic errors, actionable messages
- When to split vs consolidate tools
- Resources, prompts, and the MCP ecosystem

📁 Location: `14-mcp-tool-design/`

### 🖼️ Module 15: Multimodal LLMs
**Vision, Audio & Image Generation** - Handle more than just text
- Vision APIs (GPT-4V, Claude Vision) for image analysis and OCR
- Image generation with DALL-E 3
- Audio transcription with Whisper and text-to-speech
- CLIP embeddings for image retrieval
- Multimodal RAG combining text and images

📁 Location: `15-multimodal/`

### 🧠 Module 11: Memory & Context Management
**Persistent Intelligence** - Build memory-enabled applications
- Short-term conversation buffers
- Long-term vector-based memory
- Hierarchical memory systems (L1/L2/L3)
- Context assembly and token optimization
- User profiling and personalization
- Episodic vs semantic memory
- Memory compression and summarization
- Multi-session conversation support

📁 Location: `11-memory-context/`

## 🗺️ Learning Paths

### ⚡ Minimum Viable Path — "I want to ship my first AI product"
**~10 hours | 6 modules | Gets you to a working, deployed LLM application**

```
Module 00 → Module 01 → Module 02 → Module 05 → Module 07 → Capstone
Foundations  Prompting    RAG        Deployment  Agents     Full app
```

Do these six, build the capstone, and you'll have shipped something real. Then come back for the rest.

### 📈 Full Curriculum — "I want to be production-ready"
Work through all 15 modules in order. Each builds on the previous.

### 🔧 Practitioner Path — "I already build with LLMs, fill my gaps"
| Gap | Go to |
|-----|-------|
| Agent loops are flaky | Module 13 (Harness) |
| High API costs | Module 06 + Module 12 |
| No eval system | Module 04 + Module 09 |
| Tool/MCP integration | Module 14 |
| Production incidents | Module 08 |
| Fine-tuning questions | Module 03 |

---

## Getting Started

See [SETUP.md](SETUP.md) for the full environment guide — API keys, virtual env, troubleshooting.

### Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Set up your API keys
cp .env.example .env    # then edit .env with your keys

# 4. Start learning interactively
jupyter notebook
```

### Prerequisites
- Basic Python programming knowledge
- An API key from any supported provider: **OpenAI**, **Anthropic**, **DeepSeek**, **Grok (xAI)**, **Qwen (Alibaba)**, or **Ollama** (local)
- GPU optional (Module 03 fine-tuning only — use Google Colab if needed)

### Supported Providers

The course works with any LLM provider. Just set one key in `.env`:

| Provider | Env Variable | Default Model | Cost |
|----------|-------------|---------------|------|
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini | Pay-as-you-go |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 | Pay-as-you-go |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat | Very cheap |
| xAI Grok | `GROK_API_KEY` | grok-3-mini | Pay-as-you-go |
| Qwen | `QWEN_API_KEY` | qwen-plus | Cheap |
| Ollama | `OPENAI_BASE_URL=http://localhost:11434/v1` | llama3.2 | Free (local) |

### Learning Path

1. **Start with Module 1** - Even if you're experienced, review prompt engineering fundamentals
2. **Follow the sequence** - Each module builds on previous concepts
3. **Run the examples** - Execute the code files to see concepts in action
4. **Experiment** - Modify examples and observe the effects
5. **Build projects** - Combine concepts from multiple modules

## How to Use This Playground

### For Self-Study
1. Read the README.md in each module folder
2. Run the example Python files
3. Experiment with different parameters
4. Apply concepts to your own use cases

### For Teams
1. Use as training material for new team members
2. Reference architecture decisions
3. Share best practices across projects
4. Establish common vocabulary and patterns

## Project Structure

```
llm-engineering-playground/
├── README.md                          # This file
├── demo.py                            # ★ 60-second demo (auto-detects provider)
├── SETUP.md                           # Environment setup guide
├── requirements.txt                   # All dependencies
├── .env.example                       # API key template → copy to .env
│
├── shared/
│   └── provider.py                    # ★ Multi-provider LLM helper (OpenAI, Anthropic, DeepSeek, etc.)
│
├── 00-llm-foundations/                # Tokens, embeddings, context windows
├── 01-prompt-engineering/             # Zero-shot, few-shot, chain-of-thought
├── 02-rag-systems/                    # Vector DB, chunking, retrieval
├── 03-fine-tuning/                    # LoRA, QLoRA, data preparation
├── 04-evaluation/                     # Metrics, LLM-as-judge, benchmarks
├── 05-deployment/                     # APIs, latency, cost management
├── 06-optimization/                   # Caching, quantization, routing
├── 07-agentic-workflows/              # LangGraph, multi-agent, HITL
├── 08-llmops-observability/           # Tracing, monitoring, drift detection
├── 09-eval-ops/                       # CI/CD for LLMs, regression testing
├── 10-gateway-guardrails/             # Auth, rate limiting, injection detection
├── 11-memory-context/                 # Short/long-term memory, hierarchical
├── 12-context-engineering/            # U-curve, observation masking, prefix caching
├── 13-agent-harness/                  # Loop engineering, durable journals
├── 14-mcp-tool-design/                # Model Context Protocol, tool schemas
├── 15-multimodal/                     # Vision, image gen, audio, CLIP
│
├── capstone/
│   ├── app.py                         # FastAPI app (main entry point)
│   ├── ui.py                          # ★ Gradio web UI with streaming
│   ├── rag.py                         # RAG pipeline (multi-provider)
│   ├── cache.py                       # Semantic cache
│   ├── memory.py                      # Conversation memory
│   ├── guardrails.py                  # Input/output validation
│   ├── observability.py               # Cost + latency tracking
│   ├── evaluator.py                   # Async quality scoring
│   ├── seed_knowledge.py              # Populate the knowledge base
│   ├── chat_client.py                 # Terminal chat UI
│   ├── docker-compose.yml             # ★ Docker setup
│   ├── Dockerfile
│   └── requirements.txt
│
└── kaggle/                            # ★ Kaggle notebook series
    └── README.md
```

## Key Concepts Summary

| Concept | Description | When to Use |
|---------|-------------|-------------|
| **Prompt Engineering** | Craft effective inputs + extended thinking | Always - first line of defense |
| **RAG** | Add external knowledge | Domain-specific Q&A, current info |
| **Fine-Tuning** | Adapt model weights | Style, format, specialized tasks |
| **Evaluation** | Measure performance + LLM-as-judge | Before/after any change |
| **Deployment** | Production serving | When ready for users |
| **Optimization** | Performance, efficiency + prompt caching | Cost/performance issues |
| **Agentic Workflows** | Multi-agent + supervisor/swarm patterns | Complex multi-step tasks |
| **LLM Ops & Observability** | Monitor and trace production | After deployment, always |
| **EvalOps** | Continuous automated evaluation | CI/CD pipelines |
| **Gateway & Guardrails** | Security, rate limiting, compliance | Production protection layer |
| **Memory & Context** | Persistent memory across sessions | Conversational and personalized apps |
| **Context Engineering** | Design context quality, not just quantity | Every LLM call in production |
| **Agent Harness** | Loop engineering, durable execution | Any long-running autonomous agent |
| **MCP & Tool Design** | Standard tool interfaces for agents | When building agent tool ecosystems |
| **Multimodal** | Vision, image gen, audio, CLIP | When working with images or audio |

## Best Practices

1. **Start Simple**: Begin with prompt engineering before complex solutions
2. **Measure Everything**: Establish baselines before optimizing
3. **Iterate Quickly**: Small, frequent improvements beat big rewrites
4. **Consider Costs**: Factor in both development and operational costs
5. **Plan for Scale**: Design with production requirements in mind
6. **Stay Updated**: LLM field evolves rapidly - keep learning

## Common Pitfalls to Avoid

❌ Using GPT-4 for simple tasks (overkill)
❌ No caching strategy (wasting money)
❌ Skipping evaluation (flying blind)
❌ Ignoring latency (poor UX)
❌ Not monitoring in production (surprise failures)
❌ One-size-fits-all approach (suboptimal results)

## Capstone Project

After completing all modules, build the **Knowledge Assistant** — a full-stack LLM app that wires every module together:

```
RAG (02) + Caching (06) + Memory (11) + Guardrails (10) + Observability (08) + Evaluation (04)
```

📁 Location: [`capstone/`](capstone/)  
▶ Quick start: `cd capstone && python seed_knowledge.py && uvicorn app:app --port 8000`

## Next Steps

1. **Complete the capstone**: run the Knowledge Assistant end-to-end
2. **Contribute**: fix a broken example or add an exercise — see [CONTRIBUTING.md](CONTRIBUTING.md)
3. **Stay current**: [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent), Hugging Face blog, LangChain changelog
4. **Specialize**: dive deeper into whichever module is most relevant to your work

## Resources

### Documentation
- [LangChain Docs](https://python.langchain.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [OpenAI API Reference](https://platform.openai.com/docs)

### Communities
- r/LocalLLaMA (Reddit)
- Hugging Face Discord
- LangChain Discord

### Research
- [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent) (Computation and Language)
- [Papers With Code](https://paperswithcode.com/area/natural-language-processing)

## License

This educational resource is provided for learning purposes. Feel free to use, modify, and share.

---

**Happy Learning! 🚀**

*Remember: The field of LLM engineering is rapidly evolving. What's cutting-edge today may be standard tomorrow. Stay curious and keep building!*
