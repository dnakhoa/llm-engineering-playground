# Kaggle Notebook Series — LLM Engineering Playground

Each module is a standalone Kaggle notebook you can fork and run with free compute.

## Publishing Instructions

### 1. Create the shared dataset
```bash
# Upload shared code as a Kaggle dataset
cd kaggle
kaggle datasets create -p ./shared-code -r zip
```

### 2. Publish notebooks
For each module, create a Kaggle notebook:
1. Go to kaggle.com/code → New Notebook
2. Copy content from `../<module>/` into notebook cells
3. Add `!pip install` cell at the top
4. Name: `[LLM-E00] LLM Foundations`, `[LLM-E01] Prompt Engineering`, etc.
5. Set "Internet access" to ON (needed for API calls)
6. Save Version → Save and Run All

### 3. Series naming convention
All notebooks should start with `[LLM-EXX]` for easy discovery:
- `[LLM-E00] LLM Foundations — Tokens, Embeddings, Context`
- `[LLM-E01] Prompt Engineering — Zero-shot to Chain-of-Thought`
- `[LLM-E02] RAG Systems — Building Retrieval Pipelines`
- ... etc.

## Notebook List

| # | Notebook | Module |
|---|----------|--------|
| E00 | LLM Foundations | 00 |
| E01 | Prompt Engineering | 01 |
| E02 | RAG Systems | 02 |
| E03 | Fine-Tuning | 03 |
| E04 | Evaluation | 04 |
| E05 | Deployment | 05 |
| E06 | Optimization | 06 |
| E07 | Agentic Workflows | 07 |
| E08 | LLM Ops & Observability | 08 |
| E09 | EvalOps | 09 |
| E10 | Gateway & Guardrails | 10 |
| E11 | Memory & Context | 11 |
| E12 | Context Engineering | 12 |
| E13 | Agent Harness | 13 |
| E14 | MCP & Tool Design | 14 |
| E15 | Multimodal LLMs | 15 |
