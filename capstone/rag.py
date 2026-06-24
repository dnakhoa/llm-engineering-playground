"""RAG pipeline — Module 02 applied, multi-provider aware."""
import os
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.provider import _detect_provider, _default_model, get_model_name

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR = "./chroma_db"

RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful knowledge assistant.
Answer using ONLY the context below. If the answer isn't there, say so.
Cite [Source: X] at the end.

Context:
{context}

Conversation history:
{history}

Question: {question}

Answer:""")


def _build_llm():
    """Build an LLM instance based on the detected provider."""
    provider = _detect_provider()
    model = os.getenv("LLM_MODEL") or _default_model(provider)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0, max_tokens=1024)
    else:
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model, "temperature": 0}
        base_url = os.getenv("OPENAI_BASE_URL")
        if provider == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com/v1"
        elif provider == "grok" and not base_url:
            base_url = "https://api.x.ai/v1"
        elif provider == "qwen" and not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)


def _build_embeddings():
    """Build embeddings instance. Uses OpenAI embeddings for all providers
    (Anthropic doesn't have its own embedding API, so we use OpenAI-compatible)."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBED_MODEL)


class RAGPipeline:
    def __init__(self, k: int = 4):
        self.k = k
        self.embeddings = _build_embeddings()
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name="knowledge_base",
        )
        self.llm = _build_llm()

    def add_documents(self, docs: list[Document]):
        self.vectorstore.add_documents(docs)

    def retrieve(self, query: str) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)

    def answer(self, question: str, history: str = "") -> dict:
        docs = self.retrieve(question)
        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'doc')}]\n{d.page_content}" for d in docs
        )
        chain = RAG_PROMPT | self.llm
        response = chain.invoke({"context": context, "history": history, "question": question})
        return {
            "answer": response.content,
            "sources": [d.metadata.get("source", "unknown") for d in docs],
            "context": context,
        }

    def prepare_context(self, question: str, history: str = "") -> tuple[str, list[str]]:
        """Retrieve docs and return (context, sources) without calling LLM."""
        docs = self.retrieve(question)
        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'doc')}]\n{d.page_content}" for d in docs
        )
        sources = [d.metadata.get("source", "unknown") for d in docs]
        return context, sources

    def stream_answer(self, question: str, history: str = ""):
        """Yield answer tokens one at a time. Also yields metadata as first event."""
        docs = self.retrieve(question)
        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'doc')}]\n{d.page_content}" for d in docs
        )
        sources = [d.metadata.get("source", "unknown") for d in docs]

        chain = RAG_PROMPT | self.llm
        inputs = {"context": context, "history": history, "question": question}

        # Yield metadata first
        yield {"type": "metadata", "sources": sources}

        # Stream tokens
        for chunk in chain.stream(inputs):
            if chunk.content:
                yield {"type": "token", "content": chunk.content}

        # Yield done signal
        yield {"type": "done", "context": context}
