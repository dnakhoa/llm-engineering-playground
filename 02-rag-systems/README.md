# Module 2: RAG (Retrieval-Augmented Generation) Systems


> **Why this matters:** RAG is how most production LLM apps add knowledge without fine-tuning. Getting retrieval right — chunking, embedding, re-ranking — is the difference between a chatbot that hallucinates and one that cites sources.


## 🎯 Learning Objectives
- Understand RAG architecture and when to use it
- Design effective chunking strategies
- Implement retrieval with vector databases
- Apply advanced RAG patterns (HyDE, re-ranking, corrective RAG)
- Evaluate RAG pipeline quality

---

## What is RAG?

RAG (Retrieval-Augmented Generation) combines LLMs with external knowledge bases. Instead of relying solely on training data, RAG retrieves relevant documents and feeds them as context.

```
User Query → Embed → Vector Search → Retrieve Docs → LLM + Context → Answer
```

## Why Use RAG?

| Benefit | Description |
|---------|-------------|
| **Up-to-date** | Access data beyond training cutoff |
| **Domain-specific** | Use proprietary organizational knowledge |
| **Reduced hallucination** | Ground responses in factual documents |
| **Cost-effective** | No fine-tuning for every knowledge domain |
| **Traceable** | Users can see which documents informed the answer |

---

## 1. Document Chunking Strategies

Chunking is the most impactful decision in RAG. Too small loses context; too large dilutes relevance.

### Fixed-Size Chunking
```python
def fixed_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

### Recursive Chunking (Recommended)
Split by paragraphs first, then sentences, then words — preserving natural boundaries.

### Semantic Chunking
Use embeddings to find natural break points where meaning shifts.

### Chunk Size Guidelines

| Content Type | Recommended Size | Overlap |
|-------------|-----------------|---------|
| Technical docs | 256–512 tokens | 10–20% |
| Legal/contracts | 128–256 tokens | 20–30% |
| Conversations | 512–1024 tokens | 5–10% |
| Code | 128–256 tokens | 10% |

---

## 2. Embedding Models

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| text-embedding-3-small | 1536 | Fast | Good |
| text-embedding-3-large | 3072 | Medium | Better |
| BGE-M3 | 1024 | Fast | Excellent (multilingual) |
| Cohere Embed v3 | 1024 | Fast | Excellent |

---

## 3. Vector Databases

| Database | Best For | Key Feature |
|----------|---------|-------------|
| ChromaDB | Prototyping | Simple, embedded |
| Pinecone | Production | Managed, scalable |
| Qdrant | Performance | Rust-based, fast |
| FAISS | Research | Facebook's library |
| Weaviate | Hybrid search | Supports BM25 + vectors |

---

## 4. Retrieval Strategies

### Dense Retrieval (Semantic)
Embed query + documents, find nearest neighbors. Great for meaning-based matching.

### Sparse Retrieval (BM25)
Keyword-based. Catches exact matches that embeddings miss.

### Hybrid Retrieval
Combine dense + sparse with reciprocal rank fusion. Best of both worlds.

### Re-ranking
After initial retrieval, use a cross-encoder to re-score results for relevance. Often the single biggest quality improvement.

---

## 5. Advanced RAG Patterns

### HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer first, then use its embedding to search for real documents.

```python
# 1. Ask LLM to generate a hypothetical answer
hypothetical = llm("Answer this question: " + query)
# 2. Embed the hypothetical answer
hyde_embedding = embed(hypothetical)
# 3. Search for similar real documents
results = vector_db.search(hyde_embedding, top_k=5)
```

### Corrective RAG (CRAG)
Grade retrieval quality and retry if insufficient.

```python
retrieved_docs = retrieve(query)
grade = grade_retrieval(query, retrieved_docs)

if grade == "insufficient":
    # Rewrite query and retry
    improved_query = rewrite_query(query, retrieved_docs)
    retrieved_docs = retrieve(improved_query)
```

### Graph RAG
Build a knowledge graph from documents, traverse relationships for multi-hop answers.

### Self-RAG
Model learns when to retrieve vs answer from memory. Adds "reflection tokens" to decide.

---

## 6. Evaluation

| Metric | What It Measures |
|--------|-----------------|
| **Recall@k** | How many relevant docs are in top-k results |
| **MRR** | Rank of first relevant result |
| **Faithfulness** | Does answer match retrieved context? |
| **Answer relevancy** | Does answer address the question? |

Use [RAGAS](https://github.com/explodinggradients/ragas) for automated RAG evaluation.

---


## 📚 Resources

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) — step-by-step RAG pipeline
- [RAGAS](https://github.com/explodinggradients/ragas) — automated RAG evaluation
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) — concepts and patterns

## 🧪 Hands-On Exercises

1. **Chunk Size Experiment**: Index the same document at 128, 256, 512, 1024 tokens. Query each. Which size gives best answers?
2. **HyDE vs Standard**: Compare standard retrieval vs HyDE on 10 queries. When does HyDE help?
3. **Re-ranking Impact**: Add a cross-encoder re-ranker. Measure precision@3 before and after.
4. **Hybrid Search**: Combine BM25 + semantic search. Does it beat semantic-only?
5. **Failure Analysis**: Find 5 queries where RAG fails. Diagnose: bad chunking? bad retrieval? bad generation?

---

## Next Steps

**Next:** Module 03 — Fine-Tuning, where you'll learn how to adapt LLM weights for specific tasks.
