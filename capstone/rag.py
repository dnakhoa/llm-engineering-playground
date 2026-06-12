"""RAG pipeline — Module 02 applied."""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import os

EMBED_MODEL = "text-embedding-3-small"
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


class RAGPipeline:
    def __init__(self, k: int = 4):
        self.k = k
        self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name="knowledge_base",
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
