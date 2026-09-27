"""Acme Notes' Knowledge Base, and retrieval over it.

The Knowledge Base is the help-centre and policy articles in
``company/knowledge_base/``, one markdown file each. A retriever answers one
question: which articles are most relevant to this text?

    retriever = LexicalRetriever(load_articles())
    retriever.search("Can I get a refund?", k=2)   # [Hit(article=..., score=...), ...]

Two retrievers share that interface:

* ``LexicalRetriever`` (the default) scores articles by the words they share
  with the query, BM25-style, in pure Python. No embeddings API, no key, and
  the same answer every time, which is what Offline Checks need.
* ``EmbeddingsRetriever`` is the upgrade: it ranks by cosine similarity
  between embedding vectors, so "money back" can find "refund". You bring the
  ``embed`` function (your provider's embeddings endpoint, or a local model).

A Case that declares ``"knowledge_base": true`` gives the agent a
``search_knowledge_base`` tool backed by a retriever (``company.runner``).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from llm.types import ToolCall, ToolResult, ToolSpec

try:  # pragma: no cover - typing only
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"


@dataclass(frozen=True)
class Article:
    """One Knowledge Base article. Its id is its file name without ``.md``."""

    id: str
    title: str
    body: str


@dataclass(frozen=True)
class Hit:
    """An article a retriever found, and how well it matched."""

    article: Article
    score: float


class Retriever(Protocol):  # pragma: no cover - structural type
    """Anything that can find the ``k`` most relevant articles for a text."""

    def search(self, query: str, k: int = 2) -> List[Hit]:
        ...


def load_articles(folder: Optional[Path] = None) -> Tuple[Article, ...]:
    """Every article in the Knowledge Base, in file-name order."""
    articles = []
    for path in sorted((folder or KNOWLEDGE_BASE_DIR).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        first, _, rest = text.partition("\n")
        if not first.startswith("# "):
            raise ValueError("{} must start with a '# Title' line.".format(path))
        articles.append(Article(id=path.stem, title=first[2:].strip(), body=rest.strip()))
    return tuple(articles)


# ── Lexical retrieval ─────────────────────────────────────────────────────────

_STOPWORDS = frozenset(
    "a an and are as at be but by can could do does for from get got had has have "
    "how i if in is it its me my of on or our so that the their them then there "
    "this to up us was we what when where which will with would you your".split()
)


_SUFFIXES = ("ing", "ed", "ly", "s")


def _stem(word: str) -> str:
    """A crude stemmer: "refunded", "refunds" and "refund" all become "refund"."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4 and not word.endswith("ss"):
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> List[str]:
    """Lowercase words, stopwords dropped, common endings taken off."""
    return [
        _stem(word)
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _STOPWORDS
    ]


class LexicalRetriever:
    """BM25 over the articles' words. Pure Python, deterministic, free.

    An article scores for each query word it contains: more for a word that is
    rare across the Knowledge Base, more for a word it repeats, and a little
    less if it is long. The title counts twice, because a title is a summary.
    """

    def __init__(self, articles: Sequence[Article], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.articles = tuple(articles)
        self._k1, self._b = k1, b
        self._counts: List[Counter] = []
        for article in self.articles:
            self._counts.append(Counter(_terms(article.title) * 2 + _terms(article.body)))
        self._lengths = [sum(counts.values()) for counts in self._counts]
        self._average = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        document_frequency: Counter = Counter()
        for counts in self._counts:
            document_frequency.update(counts.keys())
        n = len(self.articles)
        self._idf: Dict[str, float] = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def search(self, query: str, k: int = 2) -> List[Hit]:
        terms = set(_terms(query))
        hits = []
        for article, counts, length in zip(self.articles, self._counts, self._lengths):
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                norm = self._k1 * (1 - self._b + self._b * length / self._average)
                score += self._idf[term] * frequency * (self._k1 + 1) / (frequency + norm)
            if score > 0:
                hits.append(Hit(article, round(score, 6)))
        hits.sort(key=lambda hit: (-hit.score, hit.article.id))
        return hits[:k]


# ── Embeddings retrieval: the upgrade ─────────────────────────────────────────

Embed = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class EmbeddingsRetriever:
    """Ranks articles by cosine similarity of embeddings. Same interface.

    ``embed`` turns a list of texts into a list of vectors. The articles are
    embedded once, when the retriever is built; each search embeds its query.
    Embeddings come from a model, so a live embeddings call makes retrieval
    cost money and vary by model: the reason it is not the Offline default.
    """

    def __init__(self, articles: Sequence[Article], *, embed: Embed) -> None:
        self.articles = tuple(articles)
        self._embed = embed
        self._vectors = list(embed(["{}\n\n{}".format(a.title, a.body) for a in self.articles]))

    def search(self, query: str, k: int = 2) -> List[Hit]:
        query_vector = list(self._embed([query]))[0]
        hits = [
            Hit(article, round(_cosine(query_vector, vector), 6))
            for article, vector in zip(self.articles, self._vectors)
        ]
        hits = [hit for hit in hits if hit.score > 0]
        hits.sort(key=lambda hit: (-hit.score, hit.article.id))
        return hits[:k]


# ── The tool the agent sees ───────────────────────────────────────────────────

SEARCH_TOOL = ToolSpec(
    name="search_knowledge_base",
    description=(
        "Search Acme Notes' help-centre and policy articles, for example the "
        "refund policy, plans and pricing, or how plan changes work. Returns the "
        "most relevant articles in full. Answer and act from what they say."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up, in plain words."}
        },
        "required": ["query"],
    },
)

#: How many articles one search returns.
SEARCH_K = 2


def run_search(retriever: Retriever, call: ToolCall) -> ToolResult:
    """Answer one ``search_knowledge_base`` call with the articles found."""
    query = call.arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return ToolResult(call.id, call.name, "search_knowledge_base needs a query.", is_error=True)
    hits = retriever.search(query, k=SEARCH_K)
    if not hits:
        return ToolResult(call.id, call.name, "No article matches {!r}.".format(query))
    content = "\n\n".join(
        "# {} (article {})\n\n{}".format(hit.article.title, hit.article.id, hit.article.body)
        for hit in hits
    )
    return ToolResult(call.id, call.name, content)
