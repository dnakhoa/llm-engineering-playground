# Changelog

All notable changes to this project are documented here.

---

## [5.6.0] — 2026-07-17

### Expanded — Module 02: RAG Systems (184 -> 330 lines)
- Added 4 chunking strategies with code: fixed, recursive, semantic, parent-child
- Added embedding model comparison (6 models) with best practices
- Added vector database comparison (7 databases) with ChromaDB example
- Added hybrid retrieval with reciprocal rank fusion code
- Added re-ranking with cross-encoder implementation
- Expanded HyDE, CRAG, Graph RAG, Self-RAG with code examples
- Added production RAG architecture diagram
- Added cost optimization table
- Added 7 hands-on exercises

### Expanded — Module 15: Multimodal (304 -> 486 lines)
- Added vision API comparison with detail parameter guidance
- Added image generation platforms comparison (DALL-E, SD, Flux, Midjourney)
- Added speaker diarization with pyannote.audio
- Added TTS voice comparison table (6 voices)
- Added voice agent architecture diagram
- Added document AI section (table extraction, PDF parsing pipeline)
- Added cost analysis for vision, audio, and generation
- Added cost optimization tips (up to 90% savings)
- Added 7 hands-on exercises

---

## [5.5.0] — 2026-07-17

### Added — Glossary
- Added 25-term glossary to root README
- Covers: LLM, tokens, embeddings, RAG, fine-tuning, LoRA, agents, MCP, A2A, ACI, guardrails, SLMs, GGUF, and more

### Added — Per-Module Troubleshooting
- Added troubleshooting table to every module README (4 common problems each)
- Covers: import errors, API issues, quality problems, configuration mistakes
- Total: 64 troubleshooting entries across 16 modules

### Added — TypeScript Examples
- `typescript/` directory with 5 runnable examples
- `chat.ts` — Basic LLM call (OpenAI + Anthropic)
- `structured-output.ts` — JSON schema extraction with Zod
- `rag.ts` — Simple RAG pipeline
- `agent.ts` — Tool-calling agent loop
- `streaming.ts` — SSE streaming response
- Run with `npx tsx <file>`

### Updated — README
- Added glossary section
- Added TypeScript directory to project structure
- Updated prerequisites to mention TypeScript

---

## [5.4.0] — 2026-07-17

### Added — Difficulty Ratings
- Added ⭐/⭐⭐/⭐⭐⭐ difficulty ratings to curriculum table in README
- Beginners: Modules 00, 01
- Intermediate: Modules 02-06, 08-11, 15
- Advanced: Modules 07, 12, 13, 14

### Added — "Why this matters" Intros
- Every module README now opens with a 2-3 sentence hook explaining real-world impact
- Helps students understand context before diving into technical content

### Added — Resource Links
- Every module README now includes a "📚 Resources" section with 3 curated links
- Links to official docs, tools, papers, and guides relevant to each module
- Total: 48 new resource links across 16 modules

### Updated — README
- Curriculum table now shows difficulty ratings instead of just ✅ status

---

## [5.3.0] — 2026-07-17

### Added — Module 03: Model Merging & Interpretability
- Model merging techniques: SLERP, TIES-Merging, DARE, FrankenMoE
- MergeKit config example and when to merge vs fine-tune
- Interpretability: Sparse Autoencoders (SAEs), ablation analysis
- Abliteration (uncensoring) workflow and use cases

### Added — Module 05: Edge Deployment
- Edge deployment frameworks: MLC LLM, mnn-llm, llama.cpp, Ollama, ONNX Runtime
- SLM comparison table: Phi-3 Mini, Gemma 2B, Llama 3.2 1B, SmolLM2, Qwen2-0.5B
- When to use edge vs cloud decision guide

### Added — Module 14: Agent-to-Agent (A2A) Protocol
- A2A vs MCP comparison (tool access vs agent collaboration)
- A2A architecture diagram
- When to use A2A: multi-agent teams, cross-org collaboration
- A2A + MCP integration pattern
- Links to A2A spec, Python SDK, Google blog post

### Updated — README
- Updated module descriptions for 03, 05, 14
- Updated curriculum table and Key Concepts table

---

## [5.2.0] — 2026-07-17

### Added — Test Suite
- `tests/test_all_modules.py` — 21 unit tests covering core logic from modules 00, 06, 12, 13, 14, 15, and capstone
- Tests validate: tokenization, cost estimation, caching savings, token budgets, novelty gates, budget trackers, journal idempotency, tool design principles, PII detection
- All tests run without API calls (no keys needed)
- Added `unit-tests` job to GitHub Actions CI

### Added — Exercise Cells to All 16 Notebooks
- Every notebook now ends with a "🧪 Exercises" section containing 3-5 hands-on challenges
- Exercises encourage experimentation: modify parameters, compare approaches, build variations
- Covers all modules: 00-15

### Updated — CI Pipeline
- Added pytest job to `.github/workflows/ci.yml`
- CI now validates: notebook structure, repo structure, AND unit tests

---

## [5.1.0] — 2026-07-17

### Added — Notebooks for 5 Modules
- `00-llm-foundations/llm_foundations.ipynb` — Tokens, embeddings, context windows, sampling, cost estimation
- `12-context-engineering/context_engineering.ipynb` — Token budgets, observation masking, sliding window compression
- `13-agent-harness/harness_example.ipynb` — Novelty gates, budget loops, durable journals
- `14-mcp-tool-design/mcp_example.ipynb` — Tool descriptions, schema design, error handling, ACI principles
- `15-multimodal/multimodal_example.ipynb` — Vision, OCR, DALL-E, Whisper TTS, multimodal pipelines

All 16 modules now have interactive notebooks (previously 11/16).

### Updated — Module 02: RAG Systems
- Expanded README from 94 to 163 lines
- Added chunking strategy guidelines with code examples
- Added embedding model comparison table
- Added advanced RAG patterns: HyDE, Corrective RAG, Graph RAG, Self-RAG
- Added evaluation metrics section (Recall@k, MRR, faithfulness, relevancy)
- Added 5 hands-on exercises

### Updated — Course Review
- Competitor analysis: mlabonne/llm-course, microsoft/generative-ai-for-beginners, awesome-llm-apps
- Identified unique differentiators: multi-provider, context engineering, agent harness, evalOps
- Gap analysis: missing notebooks, thin Module 02, no model merging/interpretability coverage

---

## [5.0.0] — 2026-07-17

### Updated — Course-wide Model References
- Updated Anthropic default from `claude-sonnet-4-20250514` to `claude-sonnet-5`
- Added Claude Fable 5, Mythos 5, Opus 4.8, Sonnet 5, Haiku 4.5 to model landscape
- Added GPT-5.6 family (gpt-5.6, gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna) references
- Updated cost ratios and model tier descriptions

### Updated — Module 00: LLM Foundations
- Added "The Responses API" section — OpenAI's recommended API replacing Chat Completions
- Added reasoning effort levels (none/low/medium/high/xhigh) and pro reasoning mode
- Added persisted reasoning (reasoning.context: current_turn/all_turns)
- Updated model landscape table with current provider lineups

### Updated — Module 01: Prompt Engineering
- Restructured "Extended Thinking" into "Reasoning Models & Effort Tuning"
- Added OpenAI GPT-5.x reasoning patterns (effort, pro mode, summaries)
- Added Anthropic adaptive thinking (always-on for Fable 5)
- Added reasoning context persistence for multi-turn conversations

### Updated — Module 06: Optimization
- Added Anthropic automatic caching (single cache_control field)
- Added explicit cache breakpoints (up to 4 per request)
- Added OpenAI explicit cache breakpoints (GPT-5.6+)
- Added `prompt_cache_key` parameter for cache routing
- Updated minimum token requirements per model

### Updated — Module 07: Agentic Workflows
- Added Agent SDKs section (OpenAI Agents SDK, Anthropic Agent SDK, Strands, LangGraph)
- Added Agent-Computer Interface (ACI) design principles from Anthropic research
- Added ACI checklist and practical examples

### Updated — Module 12: Context Engineering
- Added Anthropic automatic caching
- Added OpenAI explicit cache breakpoints (GPT-5.6+)
- Added reasoning context management (reasoning.context parameter)
- Updated cache minimum token requirements per model

### Updated — Module 13: Agent Harness
- Added phase parameter for long-running assistant interactions (commentary/final_answer)
- Added background mode for long-running tasks

### Updated — Module 14: MCP & Tool Design
- Added Secure MCP Tunnels section (OpenAI)
- Added Computer Use as a tool type (Anthropic, OpenAI)
- Updated MCP ecosystem references

### Updated — Module 15: Multimodal
- Added video generation section (OpenAI)
- Added Realtime Audio / Voice Agents section
- Updated Claude model references to Sonnet 5
- Expanded "When to Use" table

### Updated — shared/provider.py
- Updated Anthropic default model to `claude-sonnet-5`

### Updated — .env.example
- Added new model references (gpt-5.6, claude-opus-4-8, claude-fable-5)

---

## [4.0.0] — 2026-06-24

### Added — Module 15: Multimodal LLMs (`15-multimodal/`)
- `README.md` — Vision APIs (GPT-4V, Claude), image generation (DALL-E 3), Whisper transcription, CLIP embeddings, multimodal RAG pipeline
- `multimodal_example.py` — Runnable demos: image analysis, OCR, DALL-E generation, Whisper TTS→transcription round-trip, structured JSON from images
- `requirements.txt` — Module-specific dependencies

### Added — `demo.py` (repo root)
- 60-second standalone demo showing the full pipeline: provider auto-detection, RAG retrieval, input guardrails, semantic caching, output guardrails, cost tracking
- Works with OpenAI, Anthropic, DeepSeek, Ollama — auto-detects from env vars
- No server needed — runs as a single script

### Added — Capstone Web UI & Streaming
- `capstone/ui.py` — Gradio web UI with chat (streaming), document upload, observability stats tabs
- `capstone/app.py` — Added `/chat/stream` SSE endpoint for real-time token streaming
- `capstone/rag.py` — Added `stream_answer()` and `prepare_context()` methods

### Added — Multi-Provider Support
- `capstone/rag.py` — Provider-aware LLM building (OpenAI, Anthropic, DeepSeek, Grok, Qwen, Ollama)
- `capstone/cache.py` — Provider-aware embedding client
- `capstone/evaluator.py` — Provider-aware async evaluation
- `capstone/seed_knowledge.py` — Provider-aware embeddings for seeding

### Added — Docker Compose
- `capstone/docker-compose.yml` — Multi-service setup (API + UI + knowledge base)
- `capstone/Dockerfile` — Containerized capstone app
- `capstone/Spacefile` — Hugging Face Space deployment config

### Added — Kaggle Publishing
- `kaggle/README.md` — Publishing instructions and notebook series overview

### Updated — README.md
- Added "Why This Course" comparison table vs Microsoft and awesome-llm-apps
- Added Quick Start section with `python demo.py`
- Added Docker Compose instructions for capstone
- Updated project structure tree with new files
- Added Module 15 to curriculum table and module descriptions
- Added badges (Python, License, HF, Kaggle)

### Updated — `requirements.txt` (root)
- Added `gradio>=4.0.0` for web UI

### Updated — `capstone/requirements.txt`
- Added `gradio>=4.0.0`

---

### Added — Module 00: LLM Foundations (new entry point for career-changers)
- `README.md` — Tokens, embeddings, context window anatomy, sampling params, API anatomy, model tiers, cost estimation
- `llm_foundations.py` — Runnable demos: tiktoken token counting, embedding cosine similarity + semantic search, API anatomy with latency/cost, temperature effect, cost estimator across model tiers

### Added — Three Learning Paths in root README
- **Minimum Viable Path**: Modules 0→1→2→5→7→Capstone (~10h, gets to first shipped product)
- **Full Curriculum**: all 15 modules in order
- **Practitioner Path**: gap-targeting table for engineers already building with LLMs

### Updated — Module 01: Prompt Engineering
- **Structured Output**: Added native JSON Schema approach (OpenAI `response_format`, Pydantic `.parse()`), Anthropic tool-use extraction, `instructor` library — replacing "ask for JSON in the prompt"
- **Multimodal / Vision**: Added image inputs for both OpenAI (URL + base64) and Anthropic (content blocks), `detail` parameter, token cost guidance
- **Debugging LLM Failures**: Added four failure modes with diagnosis and fix: refusals, confident hallucinations, instruction ignoring, output truncation; quick debugging checklist

### Updated — Module 03: Fine-Tuning
- **Decision Tree first**: Added "Should You Fine-Tune?" decision tree as the opening section — walks through prompting → RAG → data requirements → eval before reaching fine-tuning. Added explicit "When NOT to fine-tune" list.

### Updated — Module 04: Evaluation
- **LLM-as-Judge leads (70%)**: Restructured into Part 1 (LLM-as-judge: direct scoring, pairwise comparison, bias mitigation table, calibration), Part 2 (task-specific eval harnesses), Part 3 (classical metrics — now clearly labeled as the minority case)

### Updated — Module 05: Deployment
- **Streaming as full section**: Added complete streaming treatment — `stream=True`, TTFT measurement, FastAPI SSE endpoint, JavaScript EventSource consumer
- **Error handling & retry**: Added exponential backoff with jitter, rate limit handling with `Retry-After`, timeout handling, fallback model pattern, graceful degradation hierarchy
- **Caching**: Removed duplicate caching content — redirected to Module 06 with an explicit note

### Updated — Module 08: LLM Ops & Observability
- **Prompt management workflow**: Added three practical approaches — (A) file-based Git workflow with pinned versions and diff-reviewable PRs, (B) LangSmith push/pull with tagged production pins, (C) Promptfoo for test-driven prompt development and CI integration; added checklist

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
