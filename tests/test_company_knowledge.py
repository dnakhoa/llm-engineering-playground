"""Acme Notes' Knowledge Base and the two retrievers behind one interface.

Retrieval quality is graded through Cases (tests/test_spine_2_knowledge.py),
not by asserting on scores. What is tested here is the contract the Cases rely
on: the policy the agent reads is the policy the Action enforces, and either
retriever can stand in for the other.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from company import backend  # noqa: E402
from company.knowledge import (  # noqa: E402
    EmbeddingsRetriever,
    LexicalRetriever,
    load_articles,
)


def _article(article_id):
    return next(a for a in load_articles() if a.id == article_id)


def test_the_refund_policy_article_states_the_limits_the_action_enforces():
    policy = _article("refund-policy").body

    assert "{} days".format(backend.REFUND_WINDOW_DAYS) in policy
    assert "${:.2f}".format(backend.MAX_REFUND_CENTS / 100) in policy
    assert "rounded down to the cent" in policy


def test_the_plans_article_states_the_prices_on_the_invoices():
    plans = _article("plans-and-pricing").body
    seed = backend.Backend.seeded().export_state()["accounts"]

    pro = seed["acct_1002"]["invoices"]["inv_2002"]
    team = seed["acct_1003"]["invoices"]["inv_3001"]
    assert "${:.2f} a month".format(pro["amount_cents"] / 100) in plans
    per_seat_year = team["amount_cents"] / seed["acct_1003"]["seats"] / 100
    assert "${:.2f} per seat a year".format(per_seat_year) in plans


def test_every_article_has_an_id_a_title_and_a_body():
    articles = load_articles()

    assert len(articles) >= 4
    assert len({a.id for a in articles}) == len(articles)
    for article in articles:
        assert article.title and not article.title.startswith("#")
        assert article.body.strip()


def test_the_lexical_retriever_finds_the_refund_policy_for_a_refund_question():
    hits = LexicalRetriever(load_articles()).search("Can I get a refund for this month?", k=2)

    assert 1 <= len(hits) <= 2
    assert hits[0].article.id == "refund-policy"


def test_the_lexical_retriever_returns_nothing_for_a_question_it_has_no_words_for():
    assert LexicalRetriever(load_articles()).search("zzzz qqqq", k=3) == []


def _bag_of_words(texts):
    """A toy embedding, so the upgrade can be tested with no embeddings API."""
    vocabulary = ["refund", "plan", "notebook", "invoice", "export", "seat"]
    vectors = []
    for text in texts:
        words = re.findall(r"[a-z]+", text.lower())
        vector = [float(sum(w.startswith(v) for w in words)) for v in vocabulary]
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        vectors.append([x / norm for x in vector])
    return vectors


def test_the_embeddings_retriever_answers_through_the_same_interface():
    articles = load_articles()
    lexical = LexicalRetriever(articles)
    embedded = EmbeddingsRetriever(articles, embed=_bag_of_words)

    for retriever in (lexical, embedded):
        hits = retriever.search("How do refunds work?", k=2)
        assert 1 <= len(hits) <= 2
        assert all(hit.article in articles for hit in hits)
        assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_the_embeddings_retriever_embeds_the_articles_once():
    calls = []

    def counting_embed(texts):
        calls.append(len(texts))
        return _bag_of_words(texts)

    articles = load_articles()
    retriever = EmbeddingsRetriever(articles, embed=counting_embed)
    retriever.search("refund", k=1)
    retriever.search("plan", k=1)

    assert calls == [len(articles), 1, 1]
