# Module 2: RAG (Retrieval-Augmented Generation) Systems

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that combines LLMs with external knowledge bases. Instead of relying solely on the model's training data, RAG retrieves relevant information from your own documents and feeds it to the LLM as context.

## Why Use RAG?

1. **Up-to-date Information**: Access current data beyond the model's training cutoff
2. **Domain-Specific Knowledge**: Use your organization's proprietary information
3. **Reduced Hallucinations**: Ground responses in factual, retrieved documents
4. **Cost-Effective**: No need to fine-tune for every new knowledge domain
5. **Traceability**: Users can see which documents informed the answer

## RAG Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   User      │────▶│   Retriever  │────▶│  Vector DB  │
│   Query     │     │   (Embedding)│     │  (Documents)│
└─────────────┘     └──────────────┘     └─────────────┘
       │                                       ▲
       │                                       │
       ▼                                       │
┌─────────────┐     ┌──────────────┐           │
│   Answer    │◀────│     LLM      │◀──────────┘
│             │     │ (Generation) │  Retrieved
└─────────────┘     └──────────────┘  Context
```

## Key Components

### 1. Document Processing
- **Chunking**: Breaking documents into manageable pieces
- **Cleaning**: Removing noise, formatting issues
- **Metadata**: Adding source, date, author information

### 2. Embedding Models
- Convert text into vector representations
- Popular choices: OpenAI embeddings, Hugging Face models, Cohere
- Store vectors in a vector database

### 3. Vector Databases
- **ChromaDB**: Simple, lightweight, great for prototyping
- **Pinecone**: Managed service, production-ready
- **Weaviate**: Feature-rich, supports hybrid search
- **FAISS**: Facebook's library for similarity search
- **Qdrant**: Rust-based, high performance

### 4. Retrieval Strategies
- **Dense Retrieval**: Semantic similarity using embeddings
- **Sparse Retrieval**: Keyword-based (BM25)
- **Hybrid Retrieval**: Combine both approaches
- **Re-ranking**: Re-order results for better relevance

### 5. Generation
- Feed retrieved context + user query to LLM
- Prompt engineering crucial for quality responses
- Include citations and source attribution

## Hands-On Implementation

See `rag_example.py` for a complete working example using LangChain and ChromaDB.

## Best Practices

1. **Chunk Size Matters**: Too small loses context, too large dilutes relevance (typical: 256-512 tokens)
2. **Overlap Chunks**: Add 10-20% overlap to maintain context across boundaries
3. **Quality Over Quantity**: Retrieve fewer, more relevant documents
4. **Metadata Filtering**: Filter by date, source, type before semantic search
5. **Evaluate Retrieval**: Measure recall and precision of your retrieval system
6. **Handle Edge Cases**: What if no relevant documents are found?

## Common Challenges

| Challenge | Solution |
|-----------|----------|
| Irrelevant retrievals | Improve chunking strategy, add re-ranking |
| Lost in the middle | Place key info at beginning/end of context |
| Multi-hop reasoning | Implement iterative retrieval |
| Large context windows | Use summarization or map-reduce |
| Stale data | Implement document update pipelines |

## Advanced RAG Patterns

1. **Query Transformation**: Rewrite/expand queries for better retrieval
2. **HyDE (Hypothetical Document Embeddings)**: Generate hypothetical answer, then search
3. **Parent-Child Chunking**: Retrieve small chunks, feed larger parent context
4. **RAG Fusion**: Multiple queries + reciprocal rank fusion
5. **Self-RAG**: Model learns to retrieve when needed

## Next Steps

After understanding RAG, move to Module 3: Fine-Tuning, where you'll learn how to adapt LLM weights for specific tasks.
