"""
Knowledge Assistant — Gradio Web UI
====================================
Launch:  python ui.py
Opens at http://localhost:7860

Features:
- Chat with streaming responses
- See sources and latency per turn
- Upload documents to the knowledge base
- View observability stats
"""

import os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import gradio as gr
from cache import SemanticCache
from guardrails import check_input, check_output
from memory import ConversationMemory
from observability import ObservabilityTracker, RequestMetrics
from rag import RAGPipeline

# ── State ─────────────────────────────────────────────────────────────────────
rag = RAGPipeline(k=4)
cache = SemanticCache(threshold=0.93)
tracker = ObservabilityTracker()
sessions: dict[str, ConversationMemory] = {}
MAX_SESSIONS = 1000


def chat_respond(message: str, history: list):
    """Process a chat message and yield streaming response."""
    session_id = "gradio-default"

    # Input guard
    guard = check_input(message)
    if not guard.allowed:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"**Blocked:** {guard.reason}"})
        yield history
        return

    # Cache lookup
    cached, query_emb = cache.get(message)
    if cached:
        latency_ms = 0
        stats_text = f"**[CACHED]** | Sources: {', '.join(cached.get('sources', []))}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": cached["answer"] + f"\n\n---\n{stats_text}"})
        yield history
        return

    # Session memory
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.pop(next(iter(sessions)))
        sessions[session_id] = ConversationMemory(max_turns=6)
    memory = sessions[session_id]
    history_str = memory.to_string()

    # RAG stream
    start = time.time()
    history.append({"role": "user", "content": message})
    partial_answer = ""
    sources = []

    for event in rag.stream_answer(message, history=history_str):
        if event["type"] == "metadata":
            sources = event["sources"]
            sources_text = f"**Sources:** {', '.join(sources)}"
        elif event["type"] == "token":
            partial_answer += event["content"]
            history[-1] = {"role": "assistant", "content": partial_answer + "\n\n---\n"}
            yield history
        elif event["type"] == "done":
            pass

    latency_ms = (time.time() - start) * 1000

    # Output guard
    full_answer = partial_answer
    out_guard = check_output(full_answer)
    if not out_guard.allowed:
        full_answer = "I cannot provide that response."

    # Update memory
    memory.add("user", message)
    memory.add("assistant", full_answer)

    # Cache store
    cache.set({"answer": full_answer, "sources": sources}, query_emb)

    # Metrics
    metrics = RequestMetrics(
        request_id=f"gradio-{int(time.time())}",
        question=message[:100],
        latency_ms=latency_ms,
        cache_hit=False,
    )
    tracker.record(metrics)

    # Final answer with stats
    stats_text = f"**Sources:** {', '.join(sources)} | **Latency:** {latency_ms:.0f}ms"
    history[-1] = {"role": "assistant", "content": f"{full_answer}\n\n---\n{stats_text}"}
    yield history


def get_stats():
    """Return observability stats as markdown."""
    s = tracker.summary()
    if not s:
        return "**No requests yet.** Start chatting to see stats."

    return f"""### Observability Stats

| Metric | Value |
|--------|-------|
| Total Requests | {s.get('total_requests', 0)} |
| Cache Hit Rate | {s.get('cache_hit_rate', 0):.1%} |
| Total Cost | ${s.get('total_cost_usd', 0):.6f} |
| Avg Cost/Request | ${s.get('avg_cost_usd', 0):.6f} |
| Avg Latency | {s.get('avg_latency_ms', 0):.0f}ms |
| P95 Latency | {s.get('p95_latency_ms', 0):.0f}ms |
| Cache Entries | {len(cache._store)} |
| Active Sessions | {len(sessions)} |
"""


def upload_document(file, collection_name: str):
    """Upload a document to the knowledge base."""
    if file is None:
        return "No file selected."

    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Read file
        with open(file.name, "r", errors="ignore") as f:
            content = f.read()

        # Chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        chunks = splitter.split_text(content)
        docs = [
            Document(page_content=chunk, metadata={"source": collection_name or file.name})
            for chunk in chunks
        ]

        # Add to vector store
        rag.add_documents(docs)
        return f"Added {len(docs)} chunks from `{file.name}` to knowledge base."
    except Exception as e:
        return f"Error: {e}"


# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="LLM Knowledge Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LLM Knowledge Assistant")
    gr.Markdown("Ask questions about LLM engineering. Powered by RAG + caching + guardrails.")

    with gr.Tabs():
        # Tab 1: Chat
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(type="messages", height=500, label="Conversation")
            msg = gr.Textbox(
                placeholder="Ask about RAG, fine-tuning, prompt engineering...",
                label="Your question",
                show_label=False,
            )
            msg.submit(chat_respond, [msg, chatbot], [chatbot]).then(
                lambda: "", outputs=msg
            )
            gr.Button("Send").click(chat_respond, [msg, chatbot], [chatbot]).then(
                lambda: "", outputs=msg
            )

        # Tab 2: Upload
        with gr.Tab("Upload Document"):
            gr.Markdown("Add documents to the knowledge base.")
            with gr.Row():
                file_input = gr.File(label="Upload .txt or .md file", file_types=[".txt", ".md"])
                collection_input = gr.Textbox(
                    label="Collection name",
                    value="uploaded_docs",
                    placeholder="e.g. company_policies",
                )
            upload_btn = gr.Button("Upload", variant="primary")
            upload_output = gr.Textbox(label="Result")
            upload_btn.click(upload_document, [file_input, collection_input], upload_output)

        # Tab 3: Stats
        with gr.Tab("Stats"):
            stats_display = gr.Markdown(value=get_stats())
            gr.Button("Refresh Stats").click(get_stats, outputs=stats_display)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
