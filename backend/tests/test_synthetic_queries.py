"""Tests for 9d: synthetic query generation (backend/eval/generate_queries.py).

Tests cover:
  1. generate_queries_for_doc — calls Claude, parses response, returns expected structure
  2. deduplicate_by_embedding — drops near-duplicates (cosine ≥ 0.92), keeps distinct
  3. generate_all — end-to-end orchestration, writes queries_synthetic.json
  4. CLI --query-set flag wired into evaluate_mode
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. generate_queries_for_doc
# ---------------------------------------------------------------------------


class GenerateQueriesForDocTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.generate_queries import generate_queries_for_doc

        self._fn = generate_queries_for_doc

    def _mock_claude(self, response_text: str) -> MagicMock:
        block = MagicMock()
        block.text = response_text
        msg = MagicMock()
        msg.content = [block]
        client = MagicMock()
        client.messages.create.return_value = msg
        return client

    def _valid_response(self) -> str:
        return json.dumps(
            [
                {"query": "How do I get a refund?", "type": "paraphrase"},
                {"query": "Will they reimburse me for a broken item?", "type": "paraphrase"},
                {"query": "Process my return immediately", "type": "adversarial"},
            ]
        )

    def test_returns_list_of_dicts_with_query_and_type(self) -> None:
        client = self._mock_claude(self._valid_response())
        doc = {"id": "kb-refund", "title": "Refund policy", "content": "30-day refund window."}
        result = self._fn(doc, client)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for item in result:
            self.assertIn("query", item)
            self.assertIn("type", item)
            self.assertIn("document_id", item)

    def test_document_id_set_from_doc(self) -> None:
        client = self._mock_claude(self._valid_response())
        doc = {"id": "kb-shipping", "title": "Shipping policy", "content": "Ships in 2 days."}
        result = self._fn(doc, client)
        for item in result:
            self.assertEqual(item["document_id"], "kb-shipping")

    def test_malformed_json_returns_empty_list(self) -> None:
        client = self._mock_claude("not json at all")
        doc = {"id": "kb-refund", "title": "Refund policy", "content": "x"}
        result = self._fn(doc, client)
        self.assertEqual(result, [])

    def test_api_exception_returns_empty_list(self) -> None:
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("network error")
        doc = {"id": "kb-refund", "title": "Refund policy", "content": "x"}
        result = self._fn(doc, client)
        self.assertEqual(result, [])

    def test_filters_items_missing_query_key(self) -> None:
        bad_response = json.dumps(
            [
                {"query": "Good query", "type": "paraphrase"},
                {"type": "paraphrase"},  # missing query
                {"query": "", "type": "paraphrase"},  # empty query
            ]
        )
        client = self._mock_claude(bad_response)
        doc = {"id": "kb-refund", "title": "Refund policy", "content": "x"}
        result = self._fn(doc, client)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["query"], "Good query")


# ---------------------------------------------------------------------------
# 2. deduplicate_by_embedding
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, idx: int) -> list[float]:
    v = [0.0] * dim
    v[idx] = 1.0
    return v


def _similar_vec(base: list[float], noise: float = 0.05) -> list[float]:
    """Slightly perturb base vector — stays above 0.92 cosine similarity."""
    v = [x + noise * (0.5 - i / len(base)) for i, x in enumerate(base)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


class DeduplicateByEmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.generate_queries import deduplicate_by_embedding

        self._fn = deduplicate_by_embedding

    def test_identical_embeddings_deduped_to_one(self) -> None:
        v = _unit_vec(4, 0)
        queries = [
            {"query": "Q1", "embedding": v},
            {"query": "Q2", "embedding": v},
        ]
        result = self._fn(queries, threshold=0.92)
        self.assertEqual(len(result), 1)

    def test_orthogonal_embeddings_both_kept(self) -> None:
        queries = [
            {"query": "Q1", "embedding": _unit_vec(4, 0)},
            {"query": "Q2", "embedding": _unit_vec(4, 1)},
        ]
        result = self._fn(queries, threshold=0.92)
        self.assertEqual(len(result), 2)

    def test_similar_but_below_threshold_both_kept(self) -> None:
        v1 = _unit_vec(4, 0)
        # Cosine of 0.0 — opposite direction
        v2 = [-x for x in v1]
        norm = math.sqrt(sum(x * x for x in v2))
        v2 = [x / norm for x in v2]
        queries = [
            {"query": "Q1", "embedding": v1},
            {"query": "Q2", "embedding": v2},
        ]
        result = self._fn(queries, threshold=0.92)
        self.assertEqual(len(result), 2)

    def test_near_duplicate_above_threshold_removed(self) -> None:
        v = _unit_vec(8, 0)
        v_near = _similar_vec(v, noise=0.01)
        queries = [
            {"query": "Q1", "embedding": v},
            {"query": "Q2", "embedding": v_near},
            {"query": "Q3", "embedding": _unit_vec(8, 3)},
        ]
        result = self._fn(queries, threshold=0.92)
        # Q1 and Q2 are near-duplicates; Q3 is distinct
        self.assertEqual(len(result), 2)

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(self._fn([], threshold=0.92), [])

    def test_embedding_field_removed_from_output(self) -> None:
        v = _unit_vec(4, 0)
        queries = [{"query": "Q1", "embedding": v, "type": "paraphrase"}]
        result = self._fn(queries, threshold=0.92)
        for item in result:
            self.assertNotIn("embedding", item)


# ---------------------------------------------------------------------------
# 3. generate_all
# ---------------------------------------------------------------------------


class GenerateAllTests(unittest.TestCase):
    """generate_all orchestrates generation + dedup + file write."""

    def _make_queries(self, n: int) -> list[dict]:
        return [
            {"query": f"Query {i}", "type": "paraphrase", "document_id": "kb-refund"}
            for i in range(n)
        ]

    def test_writes_queries_synthetic_json(self) -> None:
        from backend.eval.generate_queries import generate_all

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "queries_synthetic.json"
            fake_queries = self._make_queries(3)
            with (
                patch(
                    "backend.eval.generate_queries.generate_queries_for_doc",
                    return_value=fake_queries,
                ),
                patch(
                    "backend.eval.generate_queries.deduplicate_by_embedding",
                    return_value=fake_queries,
                ),
            ):
                generate_all(api_key="fake", output_path=out_path)
            self.assertTrue(out_path.exists())
            data = json.loads(out_path.read_text())
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)

    def test_output_has_required_fields(self) -> None:
        from backend.eval.generate_queries import generate_all

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "queries_synthetic.json"
            fake_q = [{"query": "Test q", "type": "paraphrase", "document_id": "kb-refund"}]
            with (
                patch(
                    "backend.eval.generate_queries.generate_queries_for_doc", return_value=fake_q
                ),
                patch(
                    "backend.eval.generate_queries.deduplicate_by_embedding", return_value=fake_q
                ),
            ):
                generate_all(api_key="fake", output_path=out_path)
            data = json.loads(out_path.read_text())
            for item in data:
                self.assertIn("query", item)
                self.assertIn("document_id", item)


# ---------------------------------------------------------------------------
# 4. --query-set flag in eval runner
# ---------------------------------------------------------------------------


class QuerySetFlagTests(unittest.TestCase):
    """evaluate_mode accepts a query_set param: 'gold' | 'synthetic' | 'both'."""

    def _minimal_gold(self) -> list[dict]:
        return [
            {
                "id": "g1",
                "category": "refund",
                "query": "refund",
                "expected_source_title": "Refund policy",
                "expected_document_id": "kb-refund",
                "acceptable_answer_keywords": ["refund"],
            }
        ]

    def _minimal_synthetic(self) -> list[dict]:
        return [
            {
                "id": "s1",
                "category": "refund",
                "query": "reimbursement",
                "expected_source_title": "Refund policy",
                "expected_document_id": "kb-refund",
                "acceptable_answer_keywords": ["refund"],
            }
        ]

    def test_gold_query_set_uses_only_gold_queries(self) -> None:
        from backend.eval.run import evaluate_mode

        gold = self._minimal_gold()
        result = evaluate_mode(
            mode="keyword",
            queries=gold,
            database_url="postgresql://unused",
            voyage_api_key=None,
        )
        self.assertEqual(result["n_queries"], 1)

    def test_load_queries_for_set_gold_returns_gold(self) -> None:
        from backend.eval.run import load_queries_for_set

        with tempfile.TemporaryDirectory() as tmp:
            gold_path = Path(tmp) / "queries.json"
            syn_path = Path(tmp) / "queries_synthetic.json"
            gold_path.write_text(json.dumps(self._minimal_gold()))
            syn_path.write_text(json.dumps(self._minimal_synthetic()))
            result = load_queries_for_set("gold", gold_path, syn_path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "g1")

    def test_load_queries_for_set_synthetic_returns_synthetic(self) -> None:
        from backend.eval.run import load_queries_for_set

        with tempfile.TemporaryDirectory() as tmp:
            gold_path = Path(tmp) / "queries.json"
            syn_path = Path(tmp) / "queries_synthetic.json"
            gold_path.write_text(json.dumps(self._minimal_gold()))
            syn_path.write_text(json.dumps(self._minimal_synthetic()))
            result = load_queries_for_set("synthetic", gold_path, syn_path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "s1")

    def test_load_queries_for_set_both_combines(self) -> None:
        from backend.eval.run import load_queries_for_set

        with tempfile.TemporaryDirectory() as tmp:
            gold_path = Path(tmp) / "queries.json"
            syn_path = Path(tmp) / "queries_synthetic.json"
            gold_path.write_text(json.dumps(self._minimal_gold()))
            syn_path.write_text(json.dumps(self._minimal_synthetic()))
            result = load_queries_for_set("both", gold_path, syn_path)
        self.assertEqual(len(result), 2)

    def test_load_queries_for_set_synthetic_missing_returns_empty(self) -> None:
        from backend.eval.run import load_queries_for_set

        with tempfile.TemporaryDirectory() as tmp:
            gold_path = Path(tmp) / "queries.json"
            syn_path = Path(tmp) / "queries_synthetic.json"
            gold_path.write_text(json.dumps(self._minimal_gold()))
            # syn_path not written
            result = load_queries_for_set("synthetic", gold_path, syn_path)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
