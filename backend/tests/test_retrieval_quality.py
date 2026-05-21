"""
Retrieval quality tests for Phase 2.

Tests are split into two layers:
- Unit tests (no external deps): verify that in-memory keyword search misses
  semantic queries, proving why the upgrade was needed.
- Integration tests (require VOYAGE_API_KEY + DATABASE_URL): verify that hybrid
  search finds the right document even when keyword search returns nothing.

Run all:       python3.11 -m pytest backend/tests/test_retrieval_quality.py -v
Run unit only: python3.11 -m pytest backend/tests/test_retrieval_quality.py -v -m "not integration"
"""

from __future__ import annotations

import os
import unittest

from backend.app.data import KNOWLEDGE_BASE
from backend.app.repository import InMemoryRepository

SEMANTIC_QUERIES = [
    # (query, expected_document_title_substring)
    ("Can I get my money back?", "Refund"),
    ("My purchase arrived damaged, I want compensation", "Refund"),
    ("How long until my package shows up?", "Shipping"),
]

UNRELATED_QUERIES = [
    "What is the capital of France?",
    "How do I compile a Rust program?",
    "What is the boiling point of water?",
]

KEYWORD_QUERIES = [
    ("Does the portable blender have a safety lock?", "blender"),
    ("What is the refund window?", "Refund"),
]


def _make_in_memory_repo() -> InMemoryRepository:
    from backend.app.data import ORDERS

    return InMemoryRepository(orders=ORDERS, knowledge_documents=KNOWLEDGE_BASE)


class KeywordSearchLimitationsTest(unittest.TestCase):
    """Proves that keyword search misses semantic queries — the motivation for Phase 2."""

    def setUp(self) -> None:
        self.repo = _make_in_memory_repo()

    def test_keyword_search_finds_exact_match_queries(self) -> None:
        for query, expected in KEYWORD_QUERIES:
            with self.subTest(query=query):
                results = self.repo.search_knowledge(query)
                self.assertTrue(results, f"Expected a result for {query!r}")
                self.assertIn(expected.lower(), results[0]["title"].lower())

    def test_keyword_search_misses_semantic_queries(self) -> None:
        misses = []
        for query, _ in SEMANTIC_QUERIES:
            results = self.repo.search_knowledge(query)
            if not results:
                misses.append(query)
        self.assertGreater(
            len(misses),
            0,
            "Expected at least one semantic query to return no keyword results — "
            "if all pass, the test queries are too simple.",
        )


def _integration_available() -> bool:
    return bool(os.getenv("VOYAGE_API_KEY") or _read_env("VOYAGE_API_KEY")) and bool(
        os.getenv("DATABASE_URL") or _read_env("DATABASE_URL")
    )


def _read_env(key: str) -> str | None:
    from pathlib import Path

    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith(key + "="):
            value = line[len(key) + 1 :]
            return value if value else None
    return None


@unittest.skipUnless(_integration_available(), "Requires VOYAGE_API_KEY and DATABASE_URL")
class HybridSearchQualityTest(unittest.TestCase):
    """Verifies hybrid search finds semantically relevant documents.

    These tests hit the real Supabase DB and Voyage API — skip in CI unless
    integration env vars are set.
    """

    # All test queries batched into one Voyage API call in setUpClass
    _ALL_QUERIES: list[str] = (
        [q for q, _ in SEMANTIC_QUERIES]
        + [q for q in UNRELATED_QUERIES]
        + [
            "My purchase arrived damaged, I want compensation",
            "What is the weather forecast in Tokyo tomorrow?",
        ]
    )
    _embeddings: dict[str, list[float]] = {}
    _db_url: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        key = os.getenv("VOYAGE_API_KEY") or _read_env("VOYAGE_API_KEY") or ""
        cls._db_url = os.getenv("DATABASE_URL") or _read_env("DATABASE_URL") or ""

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_queries = [q for q in cls._ALL_QUERIES if not (q in seen or seen.add(q))]  # type: ignore[func-returns-value]

        from backend.app.data_loader import embed_queries

        embeddings = embed_queries(unique_queries, api_key=key)
        cls._embeddings = dict(zip(unique_queries, embeddings))

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def _hybrid_score_for(self, query: str) -> tuple[str, float]:
        """Run hybrid SQL using the pre-computed embedding for query."""
        import psycopg

        emb = self._embeddings[query]
        emb_str = "[" + ",".join(str(v) for v in emb) + "]"
        sql = """
            select kd.title,
                (0.3 * ts_rank(kc.search_vector, plainto_tsquery('english', %s))
                 + 0.7 * (1 - (kc.embedding <=> %s::vector))) as score
            from knowledge_chunks kc
            join knowledge_documents kd on kd.id = kc.document_id
            where kc.embedding is not null
            order by score desc limit 1
        """
        with psycopg.connect(self._db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (query, emb_str))
                row = cur.fetchone()
                return (row[0], float(row[1])) if row else ("", 0.0)

    def test_hybrid_finds_correct_document_for_semantic_queries(self) -> None:
        for query, expected_title in SEMANTIC_QUERIES:
            with self.subTest(query=query):
                title, score = self._hybrid_score_for(query)
                self.assertIn(
                    expected_title.lower(),
                    title.lower(),
                    f"Query {query!r}: expected {expected_title!r}, got {title!r} (score={score:.3f})",
                )

    def test_hybrid_scores_are_higher_for_relevant_than_irrelevant(self) -> None:
        relevant = "My purchase arrived damaged, I want compensation"
        irrelevant = "What is the weather forecast in Tokyo tomorrow?"
        _, rel_score = self._hybrid_score_for(relevant)
        _, irr_score = self._hybrid_score_for(irrelevant)
        self.assertGreater(
            rel_score,
            irr_score,
            f"Relevant ({rel_score:.3f}) should outscore irrelevant ({irr_score:.3f})",
        )

    def test_unrelated_queries_score_below_confidence_threshold(self) -> None:
        threshold = 0.25
        for query in UNRELATED_QUERIES:
            with self.subTest(query=query):
                _, score = self._hybrid_score_for(query)
                self.assertLess(
                    score,
                    threshold,
                    f"Query {query!r} scored {score:.3f} — above threshold {threshold}, "
                    f"bot would answer instead of escalating.",
                )
