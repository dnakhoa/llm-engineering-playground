"""Semantic cache — Module 06 applied, multi-provider aware."""
import os, sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.provider import _detect_provider


def _get_embed_client():
    """Get an OpenAI-compatible client for embeddings.
    All providers use OpenAI embeddings API format."""
    from openai import OpenAI
    provider = _detect_provider()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv(f"{provider.upper()}_API_KEY", "no-key")
    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


class SemanticCache:
    def __init__(self, threshold: float = 0.93, max_size: int = 500):
        self.threshold = threshold
        self.max_size = max_size
        self._store: list[dict] = []
        self.hits = 0
        self.misses = 0

    def _embed(self, text: str) -> list[float]:
        client = _get_embed_client()
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return client.embeddings.create(model=model, input=text).data[0].embedding

    def _sim(self, a, b) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def get(self, query: str) -> tuple[Optional[dict], Optional[list]]:
        """Returns (cached_value, query_embedding). Embedding is returned so callers
        can pass it to set() and avoid a second API call on a miss."""
        emb = self._embed(query)
        for entry in self._store:
            if self._sim(emb, entry["emb"]) >= self.threshold:
                self.hits += 1
                return entry["value"], emb
        self.misses += 1
        return None, emb

    def set(self, value: dict, emb: list):
        """Store a value using a pre-computed embedding (returned by get())."""
        if len(self._store) >= self.max_size:
            self._store.pop(0)
        self._store.append({"emb": emb, "value": value})

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
