"""Long-term memory — vector-based persistent storage (in-memory demo)."""

import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class MemoryDoc:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


class VectorMemory:
    """Persistent long-term memory using vector similarity (in-memory for demo)."""

    def __init__(self, persist_directory: str = "./memory_store"):
        self.persist_directory = persist_directory
        self._store: list[MemoryDoc] = []

    def store_memory(self, content: str, metadata: Optional[Dict] = None, memory_type: str = "fact") -> str:
        doc_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()[:12]
        doc = MemoryDoc(
            id=doc_id,
            content=content,
            metadata={**(metadata or {}), "type": memory_type, "timestamp": datetime.now().isoformat()},
        )
        self._store.append(doc)
        return doc_id

    def retrieve_memories(self, query: str, k: int = 5, filter_type: Optional[str] = None) -> List[Dict]:
        candidates = self._store
        if filter_type:
            candidates = [d for d in candidates if d.metadata.get("type") == filter_type]

        query_words = set(query.lower().split())
        scored = []
        for doc in candidates:
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words | doc_words), 1)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"content": d.content, "metadata": d.metadata, "relevance_score": s}
            for s, d in scored[:k]
        ]

    def forget_memories(self, older_than_days: int = 90, min_access_count: int = 0) -> int:
        cutoff = datetime.now().timestamp() - (older_than_days * 86400)
        before = len(self._store)
        self._store = [
            d for d in self._store
            if datetime.fromisoformat(d.metadata.get("timestamp", datetime.now().isoformat())).timestamp() >= cutoff
        ]
        return before - len(self._store)

    def __len__(self):
        return len(self._store)
