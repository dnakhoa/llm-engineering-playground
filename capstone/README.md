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
# From the repo root
cd capstone
pip install -r requirements.txt
cp ../.env.example ../.env   # add OPENAI_API_KEY

# Seed the knowledge base with 10 LLM engineering topics
python seed_knowledge.py

# Start the API server
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

## Exercises & Stretch Goals

### Core Challenges
1. **Streaming Responses**: Modify `app.py` to stream tokens back to the client using Server-Sent Events. Update `chat_client.py` to display them in real-time.

2. **Add a New Guardrail**: Implement a "topic restriction" guardrail that only allows questions about LLM engineering. Test it with off-topic queries.

3. **Persistent Memory**: Replace the in-memory `ConversationMemory` with a SQLite-backed version that survives server restarts.

4. **Rate Limiting**: Add per-session rate limiting (max 20 requests/minute). Return a 429 status with a retry-after header when exceeded.

### Advanced Challenges
5. **Multi-Model Routing**: Route simple questions to a cheap model (gpt-4o-mini) and complex ones to a capable model (gpt-4o). Use the `ModelRouter` pattern from Module 06.

6. **Evaluation Dashboard**: Build a `/eval` endpoint that returns the last 50 evaluation scores, average quality, and a breakdown by category.

7. **Document Upload**: Add a `POST /upload` endpoint that accepts PDFs or text files, chunks them, and adds them to the ChromaDB knowledge base.

8. **Authentication**: Wire up the `AuthManager` from Module 10 so only authenticated users can chat. Issue tokens via a `/login` endpoint.
