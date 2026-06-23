# Changelog

All notable changes to this project are documented here.

---

## [3.0.0] — 2026-06-23

### Added — 3 New Modules (12–14)

**Module 12: Context Engineering** (`12-context-engineering/`)
- `README.md` — U-shaped attention curve, context failure modes (poisoning, confusion, clash, bloat), observation masking, prefix caching, token budgets, context compression
- `context_engineering.py` — Runnable demos: `TokenBudget`, `ObservationMasker`, sliding window compression, Anthropic prefix caching (cache_control, 5-min/1-hr TTLs), extended thinking, context quality audit

**Module 13: Agent Harness & Loop Engineering** (`13-agent-harness/`)
- `README.md` — Loop-until-dry, budget-aware loops, durable journals, self-repair, human checkpoints, pipeline vs barrier, adversarial verification
- `harness/novelty_gate.py` — `NoveltyGate` class for deduplication and dry-round counting
- `harness/journal.py` — `Journal` class: append-only JSONL, idempotent step execution, crash-proof resume
- `loops/research_loop.py` — `ResearchLoop`: combines novelty gate + journal + budget tracking
- `loops/budget_loop.py` — `BudgetLoop` + `BudgetTracker`: context-hint injection, warn/stop thresholds
- `harness_example.py` — Full demos of all 5 harness patterns

**Module 14: MCP & Tool Design** (`14-mcp-tool-design/`)
- `README.md` — MCP spec 2025-06-18, FastMCP Python server, tool description principles, schema design, error handling, split vs consolidate, resources, prompts
- `servers/example_server.py` — Complete MCP server (notes knowledge base) with 4 tools, 2 resources, 1 prompt
- `tools/tool_design.py` — `ToolValidator` (scores description quality, naming, schema, return type), `describe_tool()` helper
- `mcp_example.py` — Standalone demos: tool audit, routing comparison, error handling, schema design

### Updated — 4 Existing Modules

**Module 01 — Prompt Engineering**: Added "Extended Thinking / Reasoning Models" section
- When to use vs standard mode, budget guidelines, adaptive vs enabled mode, inter-tool thinking on Claude 4+ models, prompt patterns that work well with thinking

**Module 04 — Evaluation**: Added "Advanced: LLM-as-Judge Systems" section
- Direct scoring with rubrics, pairwise comparison (position-bias-aware), tournament evaluation, evaluator bias table (position, length, verbosity, self-preference, anchoring), rubric calibration with agreement rate threshold

**Module 06 — Optimization**: Added "Provider-Level Prompt Caching" section
- Anthropic cache_control with 5-min (1.25×) and 1-hr (2×) TTLs, 90% discount on cache reads, OpenAI automatic caching with `prompt_cache_key`, cache-friendly design rules, cost calculation example (89% savings)

**Module 07 — Agentic Workflows**: Added "Advanced Multi-Agent Patterns" section
- Supervisor/worker with context isolation, pipeline vs barrier with decision rules, adversarial verification (N-voter + perspective-diverse), swarm coordination with asyncio queue

### Updated — README.md
- Curriculum table expanded from 11 to 14 modules
- Key concepts table updated with new modules
- Module descriptions for 12, 13, 14 added

---

## [2.0.0] — 2026-06-12

### Added — Interactive Notebooks (11 modules)
- `01-prompt-engineering/prompt_engineering.ipynb` — Zero-shot, few-shot, CoT, role prompting, structured output
- `02-rag-systems/rag_systems.ipynb` — ChromaDB setup, chunking strategies, LCEL RAG chain, LLM-as-judge eval
- `03-fine-tuning/fine_tuning.ipynb` — LoRA math, QLoRA pipeline, Alpaca data format, GPT-2 local demo
- `04-evaluation/evaluation.ipynb` — ROUGE metrics, LLM-as-judge, A/B testing, `EvalHarness` class
- `05-deployment/deployment.ipynb` — Semantic caching, streaming with TTFT, cost tracking, FastAPI server
- `06-optimization/optimization.ipynb` — Prompt compression, model routing, token budgeting, length optimization
- `07-agentic-workflows/agentic_workflows.ipynb` — ReAct agent, LangGraph state machine, multi-agent router, HITL
- `08-llmops-observability/llmops_observability.ipynb` — Structured tracing, metrics, drift detection, prompt versioning
- `09-eval-ops/eval_ops.ipynb` — Test suite design, synthetic generation, regression detection, CI quality gates
- `10-gateway-guardrails/gateway_guardrails.ipynb` — JWT auth, token-bucket rate limiter, input/output guardrails
- `11-memory-context/memory_context.ipynb` — Conversation buffer, vector LTM, working memory assembly, compression

### Added — Capstone Project (`capstone/`)
- `app.py` — FastAPI server wiring all modules end-to-end
- `rag.py` — RAG pipeline with ChromaDB + OpenAI embeddings
- `cache.py` — Semantic cache with cosine similarity
- `memory.py` — Sliding window conversation memory
- `guardrails.py` — Prompt injection + PII output scanning
- `observability.py` — JSONL trace logging with p95 latency and cost summaries
- `evaluator.py` — Async LLM-as-judge scorer
- `seed_knowledge.py` — Seeds ChromaDB with 10 LLM engineering topics
- `chat_client.py` — Terminal REPL client
- `requirements.txt` — Capstone-specific dependencies

### Added — Tooling & Developer Experience
- `requirements.txt` (root) — Pinned dependencies for all 11 modules
- `requirements.txt` per module (11 files) — Isolated per-module installs
- `.env.example` — API key template with cost guidance
- `SETUP.md` — End-to-end setup guide with troubleshooting for Apple Silicon and Windows
- `CONTRIBUTING.md` — PR process, notebook standards, code quality checklist
- `.github/workflows/ci.yml` — GitHub Actions: pyflakes lint, notebook smoke test, structure check
- `CHANGELOG.md` — This file

### Fixed
- `.gitignore` — Removed erroneous markdown code fences wrapping entire file; added `.ipynb_checkpoints/`, `chroma_db/`, `*.db`, `!.env.example`
- `02-rag-systems/rag_example.py` — Replaced deprecated `HuggingFaceEmbeddings`/`HuggingFaceHub` with `OpenAIEmbeddings`/`ChatOpenAI`
- `07-agentic-workflows/skills/skill_library.py` — Fixed deprecated `from langchain.tools import tool` → `from langchain_core.tools import tool`
- `07-agentic-workflows/examples/multi_agent_workflow.py` — Added documentation header explaining mock architecture
- `01-prompt-engineering/README.md` — Replaced stale exercise referencing a file that already existed with 4 concrete hands-on challenges

### Changed
- `README.md` — Updated project structure tree, added Quick Start section, linked SETUP.md and CONTRIBUTING.md

---

## [1.0.0] — 2026-06-11

### Added — Initial Release
- Module skeleton for 11 LLM engineering topics
- Source code examples: `rag_example.py`, `multi_agent_workflow.py`, `skill_library.py`
- Per-module `README.md` files with concept explanations
- Basic `.gitignore`
