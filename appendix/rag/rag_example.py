"""
RAG (Retrieval-Augmented Generation) Example
=============================================
This file demonstrates a complete RAG pipeline in plain Python: chunking, embeddings,
a ChromaDB vector store, and an answer generated through the course's provider layer
(llm/). No framework — each step is a function you can read and swap.

For the interactive step-by-step version, open rag_systems.ipynb instead.

Prerequisites:
    pip install -r requirements.txt
    cp ../../.env.example ../../.env   # add OPENAI_API_KEY (embeddings) and any chat key
"""

import os

from dotenv import load_dotenv

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
load_dotenv(os.path.join(ROOT, ".env"))

# ============================================================================
# STEP 1: Prepare Sample Documents
# ============================================================================

sample_documents = [
    """
    Artificial Intelligence (AI) is intelligence demonstrated by machines,
    as opposed to natural intelligence displayed by animals including humans.
    Leading AI textbooks define the field as the study of "intelligent agents":
    any device that perceives its environment and takes actions that maximize
    its chance of successfully achieving its goals.
    """,
    """
    Machine Learning (ML) is a subset of artificial intelligence that provides
    systems the ability to automatically learn and improve from experience
    without being explicitly programmed. Machine learning focuses on the
    development of computer programs that can access data and use it to learn for themselves.
    """,
    """
    Deep Learning is a subset of machine learning that uses neural networks
    with many layers (deep neural networks). It's particularly effective for
    tasks like image recognition, natural language processing, and speech recognition.
    Popular frameworks include TensorFlow, PyTorch, and Keras.
    """,
    """
    Natural Language Processing (NLP) is a branch of AI that helps computers
    understand, interpret, and manipulate human language. NLP draws from many
    disciplines including computer science and computational linguistics.
    Applications include translation, sentiment analysis, and chatbots.
    """,
    """
    Large Language Models (LLMs) are language models notable for their ability
    to achieve general-purpose language generation and understanding. They acquire
    these abilities by learning from massive amounts of text data.
    Examples include GPT, Claude, Gemini, and Llama.
    """
]

# ============================================================================
# STEP 2: Text Chunking
# ============================================================================

def split_text(text, chunk_size=200, chunk_overlap=20, separators=(". ", " ")):
    """
    Pack whole sentences (or, if a sentence is too long, whole words) into chunks
    of at most chunk_size characters.

    Key parameters:
    - chunk_size: Number of characters per chunk
    - chunk_overlap: Characters repeated from the end of one chunk at the start of
      the next, so a fact split across a boundary is still retrievable
    """
    for separator in separators:
        units = text.split(separator)
        if all(len(unit) <= chunk_size for unit in units):
            break
    else:
        separator, units = "", [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    chunks, current = [], ""
    for unit in units:
        candidate = f"{current}{separator}{unit}" if current else unit
        if len(candidate) <= chunk_size or not current:
            current = candidate
            continue
        chunks.append(current)
        carried = f"{current[-chunk_overlap:]}{separator}{unit}" if chunk_overlap else unit
        current = carried if len(carried) <= chunk_size else unit
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    chunks = [
        {"id": f"doc{d}-chunk{c}", "text": chunk, "source": f"document {d + 1}"}
        for d, document in enumerate(documents)
        for c, chunk in enumerate(split_text(" ".join(document.split())))
    ]
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

# ============================================================================
# STEP 3: Create Embeddings and Vector Store
# ============================================================================

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def embed(texts):
    """Embed a batch of texts. Embeddings are not part of the provider layer."""
    from openai import OpenAI

    response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=list(texts))
    return [item.embedding for item in response.data]


def create_vector_store(chunks):
    """
    Create embeddings and store them in a vector database.

    We're using:
    - OpenAI embeddings (text-embedding-3-small by default; set EMBEDDING_MODEL)
    - ChromaDB (lightweight vector store, saved to ./chroma_db)
    """
    import chromadb

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("rag_example", metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
        embeddings=embed(c["text"] for c in chunks),
    )
    print("Vector store created successfully!")
    return collection

# ============================================================================
# STEP 4: Set Up Retriever
# ============================================================================

def retrieve(collection, question, k=2):
    """Return the k chunks most similar to the question, with their sources."""
    result = collection.query(query_embeddings=embed([question]), n_results=k)
    return [
        {"text": text, "source": meta["source"]}
        for text, meta in zip(result["documents"][0], result["metadatas"][0])
    ]

# ============================================================================
# STEP 5: Create Custom Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an AI assistant specialized in explaining technology concepts.
Use the context you are given to answer the question.
If you don't know the answer based on the context, say so clearly.
Always cite which document section you're referencing."""

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Helpful Answer (with citations):"""


def build_prompt(question, sources):
    context = "\n\n".join(f"[{s['source']}] {s['text']}" for s in sources)
    return PROMPT_TEMPLATE.format(context=context, question=question)

# ============================================================================
# STEP 6: Generate the Answer
# ============================================================================

def answer(question, sources):
    """
    Generate the answer through the provider layer (llm/).

    Works with any model in llm/models.json — set LLM_MODEL, or just the API key
    for the provider you use, in the root .env.
    """
    import sys

    sys.path.insert(0, ROOT)
    from llm import ask

    return ask(build_prompt(question, sources), system=SYSTEM_PROMPT)

# ============================================================================
# STEP 7: Query the System
# ============================================================================

def query_system(collection, question):
    """
    Query the RAG system and display results.
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    sources = retrieve(collection, question)
    result = answer(question, sources)

    print(f"Answer: {result}")
    print("\nSource Documents:")
    for i, source in enumerate(sources, 1):
        print(f"\n[{i}] ({source['source']}) {source['text'][:150]}...")

    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete RAG pipeline."""
    print("🚀 RAG System Demo")
    print("="*60)

    # Step 1: Chunk documents
    chunks = chunk_documents(sample_documents)

    # Step 2: Create vector store
    collection = create_vector_store(chunks)

    # Step 3: Test queries (retrieve, then generate)
    test_questions = [
        "What is the difference between AI and Machine Learning?",
        "How does Deep Learning work?",
        "What are some applications of NLP?"
    ]

    for question in test_questions:
        query_system(collection, question)

    print("\n✅ Demo complete!")
    print("\nNext steps:")
    print("1. Load your own documents instead of sample text")
    print("2. Experiment with different chunk sizes and retrieval strategies")
    print("3. Add metadata filtering for more precise retrieval")
    print("4. Point LLM_MODEL at a different registry model and compare answers")

if __name__ == "__main__":
    main()
