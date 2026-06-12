"""Semantic cache — Module 06 applied."""
from openai import OpenAI
import numpy as np
from typing import Optional

client = OpenAI()


class SemanticCache:
    def __init__(self, threshold: float = 0.93, max_size: int = 500):
        self.threshold = threshold
        self.max_size = max_size
        self._store: list[dict] = []
        self.hits = 0
        self.misses = 0

    def _embed(self, text: str) -> list[float]:
        return client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding

    def _sim(self, a, b) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def get(self, query: str) -> Optional[dict]:
        emb = self._embed(query)
        for entry in self._store:
            if self._sim(emb, entry["emb"]) >= self.threshold:
                self.hits += 1
                return entry["value"]
        self.misses += 1
        return None

    def set(self, query: str, value: dict):
        if len(self._store) >= self.max_size:
            self._store.pop(0)
        self._store.append({"emb": self._embed(query), "value": value})

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
