"""Seed the ChromaDB knowledge base with LLM engineering docs."""
import os, sys, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use OpenAI-compatible embeddings (works for all providers)
from langchain_openai import OpenAIEmbeddings

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

DOCS = [
    ("prompt_engineering", "Prompt engineering is the practice of designing inputs to LLMs to get reliable, high-quality outputs. Key techniques include zero-shot prompting (direct instruction), few-shot prompting (examples in the prompt), chain-of-thought prompting (asking for step-by-step reasoning), and structured output prompting (requesting JSON). The most important principle: be specific. Vague instructions produce vague answers."),
    ("rag_systems", "RAG (Retrieval-Augmented Generation) connects LLMs to external knowledge by retrieving relevant documents and including them as context. The pipeline: embed query → search vector DB → retrieve top-k chunks → include in prompt → generate answer. Key metrics: context relevance (did we retrieve the right docs?), faithfulness (does the answer stay within the retrieved context?), and answer relevance (does it answer the question?)."),
    ("fine_tuning", "Fine-tuning adapts a pre-trained model on task-specific data. LoRA (Low-Rank Adaptation) freezes the original weights and trains small adapter matrices, reducing trainable parameters by 99%+. QLoRA adds 4-bit quantisation for memory efficiency. Use fine-tuning when: prompt engineering doesn't achieve required quality, you have 500+ high-quality examples, the task has a consistent format, and latency/cost requirements justify the compute investment."),
    ("evaluation", "LLM evaluation requires multiple layers: automated metrics (ROUGE, BLEU for reference-based tasks), LLM-as-judge (a stronger model scores outputs), and human evaluation. Key pitfalls: ROUGE misses paraphrases; LLM judges show positional bias (prefer first answer) and verbosity bias (longer = better). Best practice: fix your eval set before changing the model, use blind evaluation, and track metrics over time."),
    ("deployment", "LLM deployment options: cloud APIs (low ops burden, per-token cost), self-hosted (fixed infra cost, full privacy), hybrid (route by sensitivity/cost). Production essentials: semantic cache (40-60% cost reduction), rate limiting, cost tracking per request, health endpoints, graceful error handling with retry logic. Time-to-first-token (TTFT) matters more than total latency for interactive UIs — use streaming."),
    ("optimization", "LLM cost optimisation in priority order: (1) prompt compression (remove filler — 20-50% savings), (2) semantic caching (deduplicate similar queries), (3) model routing (cheap model for simple queries), (4) response length control, (5) batching. Always measure quality before and after each optimisation. A 60% cost cut is worthless if quality drops 30%."),
    ("agentic_workflows", "Agentic systems use LLMs as reasoning engines that select and call tools. LangGraph models agents as state machines with typed state and conditional edges. Key patterns: single-agent (one LLM + tools), multi-agent (specialist agents with a router), human-in-the-loop (interrupt at checkpoints for human approval). Design principle: make agents narrow and composable rather than general and monolithic."),
    ("llmops", "LLMOps monitors LLMs in production. The three pillars: tracing (record every LLM call with inputs, outputs, latency, cost), quality monitoring (track metrics over time, detect drift), feedback loops (collect user ratings, route low-scoring responses to improvement pipelines). Tools: Langfuse (open-source tracing), Prometheus (metrics), custom dashboards. Alert when: cost exceeds budget, quality score drops >5%, error rate spikes."),
    ("memory_context", "LLM memory types: short-term (conversation buffer, last N turns), long-term (vector store of past interactions, retrieved by similarity), working memory (assembled context for the current request). Hierarchical memory (L1 conversation, L2 session, L3 user profile) mirrors human memory. Token budget management: allocate tokens to system prompt, retrieved context, conversation history, and output in fixed ratios."),
    ("guardrails", "LLM guardrails protect against prompt injection (attempts to override system instructions), PII leakage, toxic content, and hallucinations. Input guardrails: detect injection patterns, validate schema. Output guardrails: scan for sensitive data, check factual grounding, verify format. A complete gateway also handles authentication (JWT), rate limiting (per user, not IP), and compliance logging for audit trails."),
]

def main():
    chroma_dir = Path(__file__).resolve().parent / "chroma_db"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        print("Cleared existing vector store")

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    documents = []
    for source, content in DOCS:
        chunks = splitter.split_text(content)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"source": source}))

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
        collection_name="knowledge_base",
    )
    print(f"Seeded {vectorstore._collection.count()} vectors from {len(DOCS)} topics")

if __name__ == "__main__":
    main()
