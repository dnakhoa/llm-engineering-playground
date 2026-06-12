# Capstone: Knowledge Assistant — Full-Stack LLM App

This project combines every module into one production-ready application:

```
User question
     │
     ▼
[API Gateway]  ← Module 10: auth, rate limiting, input validation
     │
     ▼
[Semantic Cache] ← Module 06: skip LLM if near-duplicate query seen
     │ miss
     ▼
[RAG Pipeline]  ← Module 02: retrieve relevant docs from ChromaDB
     │
     ▼
[LLM + Memory]  ← Module 11: include conversation history
     │
     ▼
[Output Guard]  ← Module 10: validate, filter, check faithfulness
     │
     ▼
[Observability] ← Module 08: log cost, latency, quality score
     │
     ▼
[Evaluator]     ← Module 04: async LLM-as-judge score stored to DB
```

## Modules used

| Module | Component |
|--------|-----------|
| 01 | Prompt templates (system, RAG, conversation) |
| 02 | RAG pipeline (ChromaDB + OpenAI embeddings) |
| 04 | Async LLM-as-judge evaluation |
| 05 | FastAPI service with streaming |
| 06 | Semantic cache + model routing |
| 08 | Cost tracking + structured logging |
| 10 | Input validation + output guardrails |
| 11 | Short-term conversation memory |

## Running the app

```bash
cd capstone
pip install -r requirements.txt
cp ../.env.example ../.env   # add OPENAI_API_KEY

# Seed the knowledge base
python seed_knowledge.py

# Start the API
uvicorn app:app --reload --port 8000

# In another terminal — interactive chat
python chat_client.py
```

## File structure

```
capstone/
├── app.py               # FastAPI app (main entry point)
├── rag.py               # RAG pipeline
├── cache.py             # Semantic cache
├── memory.py            # Conversation memory
├── guardrails.py        # Input/output validation
├── observability.py     # Cost + latency tracking
├── evaluator.py         # Async quality scoring
├── seed_knowledge.py    # Populate the knowledge base
├── chat_client.py       # Terminal chat UI
└── requirements.txt
```
