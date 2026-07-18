# Module 2: RAG (Retrieval-Augmented Generation) Systems

> **Why this matters:** RAG is how most production LLM apps add knowledge without fine-tuning. Getting retrieval right — chunking, embedding, re-ranking — is the difference between a chatbot that hallucinates and one that cites sources.

## Learning Objectives
- Understand RAG architecture and when to use it
- Design effective chunking strategies with code
- Implement retrieval with vector databases
- Apply advanced RAG patterns (HyDE, CRAG, Graph RAG, Self-RAG)
- Build production-grade RAG pipelines
- Evaluate RAG quality with automated metrics

---

## What is RAG?

RAG combines LLMs with external knowledge bases. Instead of relying on training data, RAG retrieves relevant documents and feeds them as context.

```
User Query -> Embed -> Vector Search -> Retrieve Docs -> Re-rank -> LLM + Context -> Answer
```

## Why Use RAG?

| Benefit | Description |
|---------|-------------|
| **Up-to-date** | Access data beyond training cutoff |
| **Domain-specific** | Use proprietary organizational knowledge |
| **Reduced hallucination** | Ground responses in factual documents |
| **Cost-effective** | No fine-tuning for every knowledge domain |
| **Traceable** | Users can see which documents informed the answer |
| **Testable** | Retrieval quality is measurable (recall, precision) |

---

## 1. Document Chunking Strategies

Chunking is the most impactful decision in RAG. Too small loses context; too large dilutes relevance.

### Fixed-Size Chunking
```python
def fixed_chunk(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_text = " ".join(words[i:i + chunk_size])
        chunks.append({"text": chunk_text, "start": i, "end": min(i + chunk_size, len(words))})
    return chunks
```

### Recursive Chunking (Recommended)
Split by paragraphs first, then sentences, then words — preserving natural boundaries.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(document)
```

### Semantic Chunking
Use embeddings to find natural break points where meaning shifts.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_chunk(text, threshold=0.5):
    sentences = text.split(". ")
    embeddings = model.encode(sentences)
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
        )
        if sim < threshold:
            chunks.append(". ".join(current) + ".")
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(". ".join(current) + ".")
    return chunks
```

### Parent-Child Chunking
Retrieve small chunks for precision, return larger parent chunks for context.

```python
def parent_child_chunk(text, parent_size=1024, child_size=256):
    parents = fixed_chunk(text, chunk_size=parent_size, overlap=100)
    children = []
    for parent in parents:
        for child in fixed_chunk(parent["text"], chunk_size=child_size, overlap=30):
            child["parent_text"] = parent["text"]
            children.append(child)
    return parents, children
```

### Chunk Size Guidelines

| Content Type | Recommended Size | Overlap | Why |
|-------------|-----------------|---------|-----|
| Technical docs | 256-512 tokens | 10-20% | Preserve code blocks and formulas |
| Legal/contracts | 128-256 tokens | 20-30% | Each clause is independent |
| Conversations | 512-1024 tokens | 5-10% | Keep full turns together |
| Code | 128-256 tokens | 10% | Function/class boundaries matter |
| Books/articles | 512-1024 tokens | 15% | Paragraphs contain full ideas |
| FAQs | 128 tokens | 5% | Each Q&A is self-contained |

---

## 2. Embedding Models

### Model Comparison

| Model | Dimensions | Speed | Quality | Best For |
|-------|-----------|-------|---------|----------|
| text-embedding-3-small | 1536 | Fast | Good | Quick prototyping |
| text-embedding-3-large | 3072 | Medium | Better | Production quality |
| BGE-M3 | 1024 | Fast | Excellent | Multilingual, open-source |
| Cohere Embed v3 | 1024 | Fast | Excellent | Enterprise, multilingual |
| nomic-embed-text | 768 | Fast | Good | Local, open-source |
| GTE-Qwen2 | 1024 | Medium | Excellent | Chinese + English |

### Embedding Best Practices

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# 1. Use consistent model for indexing and querying
def embed(text):
    return client.embeddings.create(model="text-embedding-3-large", input=text).data[0].embedding

# 2. Batch embedding for efficiency
def embed_batch(texts):
    return [item.embedding for item in client.embeddings.create(model="text-embedding-3-large", input=texts).data]

# 3. Normalize for cosine similarity
def normalize(embedding):
    arr = np.array(embedding)
    return (arr / np.linalg.norm(arr)).tolist()
```

---

## 3. Vector Databases

| Database | Best For | Key Feature | Scalability |
|----------|---------|-------------|-------------|
| ChromaDB | Prototyping | Simple, embedded | Limited |
| Pinecone | Production | Managed, scalable | High |
| Qdrant | Performance | Rust-based, fast | High |
| FAISS | Research | Facebook's library | Medium |
| Weaviate | Hybrid search | BM25 + vectors | High |
| pgvector | PostgreSQL users | Extension to Postgres | High |
| Milvus | Enterprise | Distributed, GPU-accelerated | Very High |

### ChromaDB Example

```python
import chromadb

chroma = chromadb.Client()
collection = chroma.create_collection("docs")

def store(docs, ids):
    collection.add(documents=docs, embeddings=[embed(d) for d in docs], ids=ids)

def search(query, top_k=5):
    return collection.query(query_embeddings=[embed(query)], n_results=top_k)["documents"][0]
```

---

## 4. Retrieval Strategies

### Dense Retrieval (Semantic)
Embed query + documents, find nearest neighbors. Great for meaning-based matching.

### Sparse Retrieval (BM25)
Keyword-based. Catches exact matches that embeddings miss.

### Hybrid Retrieval
Combine dense + sparse with reciprocal rank fusion. Best of both worlds.

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
```

### Re-ranking
After initial retrieval, use a cross-encoder to re-score for relevance. Often the single biggest quality improvement.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs, top_k=3):
    scores = reranker.predict([(query, doc) for doc in docs])
    return [doc for doc, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]]
```

---

## 5. Advanced RAG Patterns

### HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer first, then use its embedding to search for real documents.

```python
def hyde_retrieve(query, docs, llm, top_k=5):
    hypothetical = llm("Answer this question: " + query)
    hyde_emb = embed(hypothetical)
    doc_embs = [embed(doc) for doc in docs]
    scores = cosine_similarity_batch(hyde_emb, doc_embs)
    return [doc for doc, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]]
```

**When HyDE helps:** Vague queries, different languages, technical jargon.
**When HyDE hurts:** Specific queries, latency-sensitive, inaccurate hypotheticals.

### Corrective RAG (CRAG)
Grade retrieval quality and retry if insufficient.

```python
def crag_retrieve(query, docs, llm, max_retries=2):
    for attempt in range(max_retries):
        retrieved = retrieve(query, docs)
        grade = grade_retrieval(query, retrieved, llm)
        if grade == "sufficient":
            return retrieved
        query = rewrite_query(query, retrieved, llm)
    return retrieved
```

### Graph RAG
Build a knowledge graph from documents, traverse relationships for multi-hop answers.

**When to use:** Multi-hop reasoning, relationship-heavy domains (medical, legal).

### Self-RAG
Model learns when to retrieve vs answer from memory. Adds "reflection tokens" to decide.

**When to use:** Reduces unnecessary retrieval, improves faithfulness.

---

## 6. Production RAG Architecture

```
API Gateway -> Query Processing -> Retrieval Layer -> Post-Processing -> Generation
                  |                    |                    |                |
            Query Parser         Dense+Sparse+Graph    Re-rank+Dedup    Context Assembly
            Rewrite (HyDE)       Reciprocal Rank Fusion  Filter         Citation Extract
            Classify (intent)
```

### Cost Optimization

| Technique | Savings | Tradeoff |
|-----------|---------|----------|
| Caching | 50-90% | Stale results if docs update |
| Smaller embeddings | 30-50% | Slight quality loss |
| Batch queries | 20-40% | Added latency |
| Hybrid (dense+sparse) | Better recall | More infrastructure |
| Re-ranking | +15-25% precision | Extra model call |

---

## 7. Evaluation

| Metric | What It Measures | Range |
|--------|-----------------|-------|
| **Recall@k** | How many relevant docs in top-k | 0-1 |
| **MRR** | Rank of first relevant result | 0-1 |
| **Faithfulness** | Does answer match retrieved context? | 0-1 |
| **Answer relevancy** | Does answer address the question? | 0-1 |
| **Context precision** | Are retrieved docs relevant? | 0-1 |

Use [RAGAS](https://github.com/explodinggradients/ragas) for automated RAG evaluation.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Retrieved docs are irrelevant | Try smaller chunks, add metadata filtering, use re-ranking |
| Answer doesn't use retrieved context | Strengthen system prompt: "Answer ONLY from provided context" |
| ChromaDB import fails | pip install chromadb --no-cache-dir |
| Embedding dimension mismatch | Ensure same embedding model for indexing and querying |
| Too many results, slow response | Reduce top_k, add diversity filter |
| Answer is generic | Increase chunk size, add more domain-specific documents |
| Retrieval misses obvious matches | Try hybrid search (dense + BM25) |

## Resources

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) — step-by-step RAG pipeline
- [RAGAS](https://github.com/explodinggradients/ragas) — automated RAG evaluation
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) — concepts and patterns
- [LlamaIndex](https://docs.llamaindex.ai/) — data framework for RAG

## Exercises

1. **Chunk Size**: Index the same document at 128, 256, 512 tokens. Which gives best retrieval?
2. **HyDE vs Standard**: Compare standard retrieval vs HyDE on 10 queries.
3. **Re-ranking**: Add cross-encoder re-ranker. Measure precision@3 before/after.
4. **Hybrid Search**: Combine BM25 + semantic search. Does it beat semantic-only?
5. **Failure Analysis**: Find 5 queries where RAG fails. Diagnose the cause.
6. **Parent-Child**: Implement parent-child chunking. Compare with simple chunking.
7. **CRAG**: Build a corrective RAG pipeline that rewrites queries when retrieval is poor.

---

**Next:** Module 03 — Fine-Tuning, where you'll learn how to adapt LLM weights for specific tasks.
