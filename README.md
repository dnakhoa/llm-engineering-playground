# LLM Engineering Playground

A comprehensive, hands-on learning resource for mastering Large Language Model (LLM) engineering and development.

## Overview

This playground provides a structured, step-by-step curriculum covering all essential aspects of LLM engineering, from basic prompt engineering to advanced optimization techniques. Each module includes conceptual explanations and practical code examples.

## Curriculum Structure

**11 Comprehensive Modules** covering the complete LLM engineering lifecycle:

| Module | Topic | Key Focus | Time | Status |
|--------|-------|-----------|------|--------|
| 01 | Prompt Engineering | Foundation communication | ~1h | ✅ |
| 02 | RAG Systems | Knowledge augmentation | ~2h | ✅ |
| 03 | Fine-Tuning | Model adaptation | ~2h | ✅ |
| 04 | Evaluation | Quality assurance | ~1.5h | ✅ |
| 05 | Deployment | Production serving | ~2h | ✅ |
| 06 | Optimization | Performance & efficiency | ~1.5h | ✅ |
| 07 | Agentic Workflows | Multi-agent systems | ~3h | ✅ |
| 08 | LLM Ops | Observability & monitoring | ~2h | ✅ |
| 09 | EvalOps | Continuous evaluation | ~1.5h | ✅ |
| 10 | Gateway & Guardrails | Security & compliance | ~2h | ✅ |
| 11 | Memory & Context | Persistent intelligence | ~2h | ✅ |

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
├── SETUP.md                           # Environment setup guide
├── requirements.txt                   # All dependencies
├── .env.example                       # API key template → copy to .env
│
├── shared/
│   └── provider.py                    # ★ Multi-provider LLM helper (OpenAI, Anthropic, DeepSeek, etc.)
│
├── 01-prompt-engineering/
│   ├── README.md                      # Concepts and theory
│   ├── requirements.txt
│   ├── prompt_engineering.ipynb       # ★ Interactive notebook
│   └── prompt_examples.py             # Runnable script
│
├── 02-rag-systems/
│   ├── README.md
│   ├── requirements.txt
│   ├── rag_systems.ipynb              # ★ Interactive notebook
│   └── rag_example.py
│
├── 03-fine-tuning/
│   ├── README.md
│   ├── requirements.txt
│   └── finetune_example.py
│
├── 04-evaluation/
│   ├── README.md
│   ├── requirements.txt
│   └── evaluation_example.py
│
├── 05-deployment/
│   ├── README.md
│   ├── requirements.txt
│   └── deployment_example.py
│
├── 06-optimization/
│   ├── README.md
│   ├── requirements.txt
│   └── optimization_example.py
│
├── 07-agentic-workflows/
│   ├── README.md
│   ├── requirements.txt
│   ├── agents/
│   ├── examples/
│   │   ├── multi_agent_workflow.py
│   │   └── human_in_loop.py
│   ├── skills/
│   ├── router/
│   └── configs/
│
├── 08-llmops-observability/
│   ├── README.md
│   ├── requirements.txt
│   └── observability_example.py
│
├── 09-eval-ops/
│   ├── README.md
│   ├── requirements.txt
│   └── eval_ops_example.py
│
├── 10-gateway-guardrails/
│   ├── README.md
│   ├── requirements.txt
│   ├── gateway/
│   │   ├── auth.py
│   │   ├── rate_limiter.py
│   │   └── validator.py
│   └── guardrails/
│       ├── input_filters.py
│       ├── output_filters.py
│       └── compliance.py
│
└── 11-memory-context/
    ├── README.md
    ├── requirements.txt
    └── memory/
        ├── short_term.py
        ├── long_term.py
        ├── working.py
        └── hierarchical.py
```

## Key Concepts Summary

| Concept | Description | When to Use |
|---------|-------------|-------------|
| **Prompt Engineering** | Craft effective inputs | Always - first line of defense |
| **RAG** | Add external knowledge | Domain-specific Q&A, current info |
| **Fine-Tuning** | Adapt model weights | Style, format, specialized tasks |
| **Evaluation** | Measure performance | Before/after any change |
| **Deployment** | Production serving | When ready for users |
| **Optimization** | Improve efficiency | Cost/performance issues |
| **Agentic Workflows** | Multi-agent systems | Complex multi-step tasks |
| **LLM Ops & Observability** | Monitor and trace production | After deployment, always |
| **EvalOps** | Continuous automated evaluation | CI/CD pipelines |
| **Gateway & Guardrails** | Security, rate limiting, compliance | Production protection layer |
| **Memory & Context** | Persistent memory across sessions | Conversational and personalized apps |

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
