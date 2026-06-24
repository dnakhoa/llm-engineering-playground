"""
Knowledge Assistant — Capstone FastAPI App
=========================================
Combines: RAG (02) + Caching (06) + Memory (11) + Guardrails (10) + Observability (08) + Evaluation (04)

Run:
    uvicorn app:app --reload --port 8000
"""
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv(dotenv_path="../.env")

from cache import SemanticCache
from evaluator import evaluate_response
from guardrails import check_input, check_output
from memory import ConversationMemory
from observability import ObservabilityTracker, RequestMetrics
from rag import RAGPipeline

# ── State ─────────────────────────────────────────────────────────────────────
rag: Optional[RAGPipeline] = None
cache = SemanticCache(threshold=0.93)
tracker = ObservabilityTracker()
MAX_SESSIONS = 1000
sessions: dict[str, ConversationMemory] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = RAGPipeline(k=4)
    print("✅ RAG pipeline ready")
    yield


app = FastAPI(title="Knowledge Assistant", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    cache_hit: bool
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", **tracker.summary()}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    request_id = str(uuid.uuid4())[:8]

    # 1. Input guard
    guard = check_input(req.question)
    if not guard.allowed:
        raise HTTPException(status_code=400, detail=guard.reason)

    # 2. Cache lookup — get() also returns the embedding so set() reuses it
    cached, query_emb = cache.get(req.question)
    if cached:
        metrics = RequestMetrics(
            request_id=request_id,
            question=req.question[:100],
            cache_hit=True,
            latency_ms=(time.time() - start) * 1000,
        )
        tracker.record(metrics)
        return ChatResponse(cache_hit=True, latency_ms=metrics.latency_ms, session_id=req.session_id, **cached)

    # 3. Get/create session memory (evict oldest when cap reached)
    if req.session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.pop(next(iter(sessions)))
        sessions[req.session_id] = ConversationMemory(max_turns=6)
    memory = sessions[req.session_id]
    history = memory.to_string()

    # 4. RAG answer
    result = rag.answer(req.question, history=history)
    answer = result["answer"]

    # 5. Output guard
    out_guard = check_output(answer)
    if not out_guard.allowed:
        answer = "I cannot provide that response."

    # 6. Update memory
    memory.add("user", req.question)
    memory.add("assistant", answer)

    # 7. Cache store — reuse the embedding computed in step 2 (no second API call)
    cache_value = {"answer": answer, "sources": result["sources"]}
    cache.set(cache_value, query_emb)

    # 8. Record metrics
    latency_ms = (time.time() - start) * 1000
    metrics = RequestMetrics(
        request_id=request_id,
        question=req.question[:100],
        model="gpt-4o-mini",
        latency_ms=latency_ms,
        cache_hit=False,
    )
    tracker.record(metrics)

    # 9. Async quality eval (fire-and-forget — doesn't block response)
    asyncio.create_task(evaluate_response(req.question, answer, result["context"]))

    return ChatResponse(
        answer=answer,
        sources=result["sources"],
        latency_ms=round(latency_ms),
        cache_hit=False,
        session_id=req.session_id,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming SSE endpoint — yields tokens as they arrive."""
    start = time.time()
    request_id = str(uuid.uuid4())[:8]

    guard = check_input(req.question)
    if not guard.allowed:
        raise HTTPException(status_code=400, detail=guard.reason)

    cached, query_emb = cache.get(req.question)
    if cached:
        metrics = RequestMetrics(
            request_id=request_id, question=req.question[:100],
            cache_hit=True, latency_ms=(time.time() - start) * 1000,
        )
        tracker.record(metrics)

        async def cached_stream():
            yield f"data: {json.dumps({'type': 'metadata', 'sources': cached['sources']})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': cached['answer']})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    if req.session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.pop(next(iter(sessions)))
        sessions[req.session_id] = ConversationMemory(max_turns=6)
    memory = sessions[req.session_id]
    history = memory.to_string()

    async def stream_response():
        answer_parts = []
        sources = []
        context = ""

        for event in rag.stream_answer(req.question, history=history):
            if event["type"] == "metadata":
                sources = event["sources"]
                yield f"data: {json.dumps(event)}\n\n"
            elif event["type"] == "token":
                answer_parts.append(event["content"])
                yield f"data: {json.dumps(event)}\n\n"
            elif event["type"] == "done":
                context = event.get("context", "")

        full_answer = "".join(answer_parts)

        out_guard = check_output(full_answer)
        if not out_guard.allowed:
            full_answer = "I cannot provide that response."

        memory.add("user", req.question)
        memory.add("assistant", full_answer)

        cache.set({"answer": full_answer, "sources": sources}, query_emb)

        latency_ms = (time.time() - start) * 1000
        metrics = RequestMetrics(
            request_id=request_id, question=req.question[:100],
            model="gpt-4o-mini", latency_ms=latency_ms, cache_hit=False,
        )
        tracker.record(metrics)

        asyncio.create_task(evaluate_response(req.question, full_answer, context))

        yield f"data: {json.dumps({'type': 'done', 'latency_ms': round(latency_ms)})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.get("/stats")
def stats():
    return {
        "cache": {"hit_rate": round(cache.hit_rate, 3), "size": len(cache._store)},
        "sessions": len(sessions),
        **tracker.summary(),
    }
