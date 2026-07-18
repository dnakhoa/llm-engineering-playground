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

**16 Comprehensive Modules** covering the complete LLM engineering lifecycle — all with interactive notebooks:

| Module | Topic | Key Focus | Time | Difficulty |
|--------|-------|-----------|------|------------|
| 00 | LLM Foundations | Tokens, embeddings, context, Responses API, reasoning | ~1h | ⭐ Beginner |
| 01 | Prompt Engineering | Zero/few-shot, CoT, structured output, reasoning models | ~1.5h | ⭐ Beginner |
| 02 | RAG Systems | Vector DB, chunking, HyDE, CRAG, evaluation | ~2h | ⭐⭐ Intermediate |
| 03 | Fine-Tuning | LoRA, QLoRA, model merging, interpretability | ~2h | ⭐⭐ Intermediate |
| 04 | Evaluation | LLM-as-judge, benchmarks, A/B testing | ~2h | ⭐⭐ Intermediate |
| 05 | Deployment | Cloud APIs, streaming, edge deployment, SLMs | ~2h | ⭐⭐ Intermediate |
| 06 | Optimization | Prompt caching, quantization, model routing | ~2h | ⭐⭐ Intermediate |
| 07 | Agentic Workflows | LangGraph, Agent SDKs, ACI, multi-agent | ~3.5h | ⭐⭐⭐ Advanced |
| 08 | LLM Ops | Tracing, drift detection, prompt versioning | ~2h | ⭐⭐ Intermediate |
| 09 | EvalOps | CI/CD for LLMs, synthetic data, regression testing | ~1.5h | ⭐⭐ Intermediate |
| 10 | Gateway & Guardrails | Auth, rate limiting, injection detection, PII | ~2h | ⭐⭐ Intermediate |
| 11 | Memory & Context | Short/long-term memory, hierarchical systems | ~2h | ⭐⭐ Intermediate |
| 12 | Context Engineering | U-curve, observation masking, prefix caching | ~2h | ⭐⭐⭐ Advanced |
| 13 | Agent Harness | Loop engineering, durable journals, budget loops | ~3h | ⭐⭐⭐ Advanced |
| 14 | MCP & Tool Design | MCP, A2A, ACI, Secure Tunnels, Computer Use | ~2h | ⭐⭐⭐ Advanced |
| 15 | Multimodal | Vision, image gen, video, audio, voice agents | ~2h | ⭐⭐ Intermediate |

### 🧱 Module 0: LLM Foundations
**Start Here** - The mental models every LLM engineer needs
- What tokens are (and why the model can't count characters)
- Embeddings as geometry — why semantic search works
- Context window anatomy: input tokens, output tokens, limits
- Sampling parameters: temperature, top-p, max_tokens
- Responses API (recommended) vs Chat Completions
- Reasoning effort tuning (none/low/medium/high/xhigh)
- Model selection: when to use small/medium/large
- Cost estimation before you build

📁 Location: `00-llm-foundations/`

### 📚 Module 1: Prompt Engineering
**Foundation** - Learn to communicate effectively with LLMs
- Zero-shot and few-shot prompting
- Chain-of-thought reasoning
- Role prompting and personas
- Structured output generation
- Reasoning models, effort tuning, adaptive thinking

📁 Location: `01-prompt-engineering/`

### 🔍 Module 2: RAG Systems
**Knowledge Augmentation** - Connect LLMs to external data
- Vector databases and embeddings
- Document chunking strategies (fixed, recursive, semantic)
- Dense, sparse, and hybrid retrieval
- Advanced patterns (HyDE, Corrective RAG, Graph RAG)
- RAG evaluation (Recall@k, faithfulness, relevancy)

📁 Location: `02-rag-systems/`

### 🎯 Module 3: Fine-Tuning
**Model Adaptation** - Customize LLMs for your needs
- When to fine-tune vs. other approaches (decision tree)
- LoRA and QLoRA techniques
- Data preparation and training frameworks
- Model merging (SLERP, TIES, DARE, FrankenMoE)
- Interpretability (SAEs, abliteration, feature analysis)
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
- Latency optimization and streaming
- Edge deployment (MLC LLM, llama.cpp, SLMs)
- Cost management
- Security and privacy

📁 Location: `05-deployment/`

### ⚡ Module 6: Optimization
**Performance & Efficiency** - Make it faster and cheaper
- Quantization and pruning
- Caching strategies (automatic + explicit breakpoints)
- Model routing
- Prompt optimization
- Token budgeting

📁 Location: `06-optimization/`

### 🤖 Module 7: Agentic Workflows
**Autonomous Systems** - Build multi-agent collaborative systems
- LangChain agents and tools
- Agent SDKs (OpenAI, Anthropic, Strands)
- LangGraph state machines
- Agent-Computer Interface (ACI) design
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

### 🧠 Module 12: Context Engineering
**Context as a Resource** - Design what enters the context window, not just fill it
- Context window anatomy and the U-shaped attention curve
- Context poisoning, confusion, and clash — failure modes and fixes
- Observation masking — compress tool outputs before they hit context
- Provider-level prefix caching (automatic + explicit breakpoints)
- Reasoning context management (reasoning.context parameter)
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
- Phase parameter for long-running interactions
- Background mode for async tasks
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
- Agent-Computer Interface (ACI) design principles
- Agent-to-Agent (A2A) protocol for multi-agent collaboration
- Secure MCP Tunnels for production deployment
- Computer Use as a tool type

📁 Location: `14-mcp-tool-design/`

### 🖼️ Module 15: Multimodal LLMs
**Vision, Audio, Video & Image Generation** - Handle more than just text
- Vision APIs (GPT-4o, Claude Vision) for image analysis and OCR
- Image generation with DALL-E 3
- Video generation (Sora, gpt-4o-video)
- Audio transcription with Whisper and text-to-speech
- Realtime Audio / Voice Agents
- CLIP embeddings for image retrieval
- Multimodal RAG combining text and images

📁 Location: `15-multimodal/`

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
- Basic Python or TypeScript knowledge
- An API key from any supported provider: **OpenAI**, **Anthropic**, **DeepSeek**, **Grok (xAI)**, **Qwen (Alibaba)**, or **Ollama** (local)
- GPU optional (Module 03 fine-tuning only — use Google Colab if needed)

### Supported Providers

The course works with any LLM provider. Just set one key in `.env`:

| Provider | Env Variable | Default Model | Cost |
|----------|-------------|---------------|------|
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini (or gpt-5.6 for reasoning) | Pay-as-you-go |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-5 (or claude-opus-4-8 for complex) | Pay-as-you-go |
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
├── typescript/                        # ★ TypeScript examples (chat, RAG, agents)
│   ├── README.md
│   ├── chat.ts
│   ├── rag.ts
│   ├── agent.ts
│   └── streaming.ts
│
└── kaggle/                            # ★ Kaggle notebook series
    └── README.md
```

## Key Concepts Summary

| Concept | Description | When to Use |
|---------|-------------|-------------|
| **Prompt Engineering** | Craft effective inputs + reasoning effort tuning | Always - first line of defense |
| **RAG** | Add external knowledge | Domain-specific Q&A, current info |
| **Fine-Tuning** | Adapt model weights + merging + interpretability | Style, format, specialized tasks |
| **Evaluation** | Measure performance + LLM-as-judge | Before/after any change |
| **Deployment** | Production serving + edge deployment | When ready for users |
| **Optimization** | Performance, efficiency + provider-level prompt caching | Cost/performance issues |
| **Agentic Workflows** | Multi-agent + supervisor/swarm patterns + Agent SDKs | Complex multi-step tasks |
| **LLM Ops & Observability** | Monitor and trace production | After deployment, always |
| **EvalOps** | Continuous automated evaluation | CI/CD pipelines |
| **Gateway & Guardrails** | Security, rate limiting, compliance | Production protection layer |
| **Memory & Context** | Persistent memory across sessions | Conversational and personalized apps |
| **Context Engineering** | Design context quality + reasoning context management | Every LLM call in production |
| **Agent Harness** | Loop engineering, durable execution, phase management | Any long-running autonomous agent |
| **MCP & Tool Design** | MCP + A2A + ACI + Secure Tunnels | When building agent tool ecosystems |
| **Multimodal** | Vision, image gen, audio, video generation | When working with images, audio, or video |

## Best Practices

1. **Start Simple**: Begin with prompt engineering before complex solutions
2. **Measure Everything**: Establish baselines before optimizing
3. **Iterate Quickly**: Small, frequent improvements beat big rewrites
4. **Consider Costs**: Factor in both development and operational costs
5. **Plan for Scale**: Design with production requirements in mind
6. **Stay Updated**: LLM field evolves rapidly - keep learning

## Common Pitfalls to Avoid

❌ Using frontier models for simple tasks (overkill — use small models)
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
- [OpenAI Responses API](https://platform.openai.com/docs/guides/responses)
- [OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents)
- [Anthropic Docs](https://docs.anthropic.com/)
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)

### Communities
- r/LocalLLaMA (Reddit)
- Hugging Face Discord
- LangChain Discord

### Research
- [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent) (Computation and Language)
- [Papers With Code](https://paperswithcode.com/area/natural-language-processing)


## 📖 Glossary

| Term | Definition |
|------|-----------|
| **LLM** | Large Language Model — neural network trained on text to generate/understand language |
| **Token** | Subword unit the model processes; ~1.3 tokens per English word |
| **Embedding** | Vector representation of text meaning; enables semantic search |
| **Context Window** | Maximum tokens the model can process in one call (input + output) |
| **RAG** | Retrieval-Augmented Generation — adding external knowledge to LLM prompts |
| **Fine-Tuning** | Updating model weights on domain-specific data |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning by updating small matrices |
| **QLoRA** | Quantized LoRA — fine-tuning with 4-bit quantization to save memory |
| **Agent** | LLM that can reason, plan, and use tools autonomously |
| **MCP** | Model Context Protocol — standard for connecting agents to tools |
| **A2A** | Agent-to-Agent — standard for agent-to-agent communication |
| **ACI** | Agent-Computer Interface — principles for designing tools agents use well |
| **Guardrails** | Safety filters that validate LLM inputs and outputs |
| **Prompt Injection** | Attack that tricks LLM into ignoring instructions |
| **PII** | Personally Identifiable Information — data that identifies a person |
| **Hallucination** | Model generating plausible-sounding but factually incorrect information |
| **Latency** | Time from request to first response token (TTFT) |
| **Streaming** | Sending tokens to client as they're generated, not all at once |
| **Temperature** | Controls randomness in output; 0.0=deterministic, 1.0=creative |
| **Top-p** | Nucleus sampling; restricts token selection to probability mass p |
| **Reasoning Tokens** | Hidden tokens the model uses to "think" before answering |
| **SLM** | Small Language Model — <10B parameter models for edge deployment |
| **GGUF** | Quantized model format for llama.cpp and CPU inference |

## License

This educational resource is provided for learning purposes. Feel free to use, modify, and share.

---

**Happy Learning! 🚀**

*Remember: The field of LLM engineering is rapidly evolving. What's cutting-edge today may be standard tomorrow. Stay curious and keep building!*
