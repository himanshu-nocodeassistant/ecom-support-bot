"""Unit tests for phase 6/9 eval infrastructure.

Covers six areas — all without external deps (no DB, no Voyage, no Claude API):
  1. MetricHelpers     — _precision_at_k, _recall_at_k, _context_relevance,
                         _answer_correctness, _estimate_cost
  2. DocIdMetrics      — _precision_at_k_doc, _recall_at_k_doc: document-id-based
                         versions that fix the structural bias against chunked modes
  3. LlmJudge         — _llm_judge_context_relevance: valid response, score clamping,
                         malformed JSON, API exception
  4. BenchmarkMd      — _generate_benchmark_md: file structure with and without
                         LLM column
  5. RegressionGate   — check_regression logic: pass, fail, missing baseline,
                         missing results file, agent metrics
  6. AgentEvalScoring — run_agent_eval metric formulas isolated from the
                         Claude API via mocked handle_message
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. Metric helper unit tests
# ---------------------------------------------------------------------------


class PrecisionAtKTests(unittest.TestCase):
    from backend.eval.run import _precision_at_k

    def setUp(self) -> None:
        from backend.eval.run import _precision_at_k

        self._fn = _precision_at_k

    def _chunk(self, title: str) -> dict:
        return {"title": title, "content": "x", "score": 0.5}

    def test_all_match(self) -> None:
        results = [self._chunk("Refund policy")] * 3
        self.assertAlmostEqual(self._fn(results, "Refund policy"), 1.0)

    def test_no_match(self) -> None:
        results = [self._chunk("Shipping policy")] * 3
        self.assertAlmostEqual(self._fn(results, "Refund policy"), 0.0)

    def test_one_of_three_matches(self) -> None:
        results = [
            self._chunk("Refund policy"),
            self._chunk("Shipping policy"),
            self._chunk("Shipping policy"),
        ]
        self.assertAlmostEqual(self._fn(results, "Refund policy"), 1 / 3)

    def test_case_insensitive(self) -> None:
        results = [self._chunk("refund policy")] * 3
        self.assertAlmostEqual(self._fn(results, "Refund Policy"), 1.0)

    def test_no_expected_title(self) -> None:
        results = [self._chunk("Refund policy")]
        self.assertAlmostEqual(self._fn(results, None), 0.0)

    def test_empty_results(self) -> None:
        self.assertAlmostEqual(self._fn([], "Refund policy"), 0.0)

    def test_only_top_k_considered(self) -> None:
        # 4th result matches but k=3 so it should not count
        results = [
            self._chunk("Shipping policy"),
            self._chunk("Shipping policy"),
            self._chunk("Shipping policy"),
            self._chunk("Refund policy"),
        ]
        self.assertAlmostEqual(self._fn(results, "Refund policy", k=3), 0.0)


class RecallAtKTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.run import _recall_at_k

        self._fn = _recall_at_k

    def _chunk(self, title: str) -> dict:
        return {"title": title, "content": "x", "score": 0.5}

    def test_found_in_top3(self) -> None:
        results = [self._chunk("Shipping policy"), self._chunk("Refund policy"), self._chunk("x")]
        self.assertAlmostEqual(self._fn(results, "Refund policy"), 1.0)

    def test_not_found(self) -> None:
        results = [self._chunk("Shipping policy")] * 3
        self.assertAlmostEqual(self._fn(results, "Refund policy"), 0.0)

    def test_binary_not_fractional(self) -> None:
        # All three match — still 1.0, not 3.0
        results = [self._chunk("Refund policy")] * 3
        self.assertAlmostEqual(self._fn(results, "Refund policy"), 1.0)

    def test_none_expected_title(self) -> None:
        self.assertAlmostEqual(self._fn([self._chunk("Refund policy")], None), 0.0)

    def test_empty_results(self) -> None:
        self.assertAlmostEqual(self._fn([], "Refund policy"), 0.0)


# ---------------------------------------------------------------------------
# 2. Document-id-based metric tests (9a — fixes structural bias)
# ---------------------------------------------------------------------------


class PrecisionAtKDocTests(unittest.TestCase):
    """_precision_at_k_doc compares by document_id, not title string.

    This eliminates the structural bias where a chunked mode returning 3 chunks
    from the same correct document scored only 1/3 (one title match out of three
    distinct title strings) versus keyword mode returning the whole doc once.
    """

    def setUp(self) -> None:
        from backend.eval.run import _precision_at_k_doc

        self._fn = _precision_at_k_doc

    def _chunk(self, doc_id: str, title: str = "Any") -> dict:
        return {"id": doc_id, "title": title, "content": "x", "score": 0.5}

    def test_all_three_chunks_same_correct_doc_scores_one(self) -> None:
        # Three chunks from the same correct document — should score 1.0, not 1/3
        results = [self._chunk("kb-refund")] * 3
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 1.0)

    def test_chunked_mode_bias_is_fixed(self) -> None:
        # Demonstrates the old bug: title-based P@3 would score 1/3 here
        # because all three returned the same title once. Doc-id P@3 should be 1.0.
        results = [
            {"id": "kb-refund", "title": "Refund policy", "content": "chunk 1", "score": 0.9},
            {"id": "kb-refund", "title": "Refund policy", "content": "chunk 2", "score": 0.8},
            {"id": "kb-refund", "title": "Refund policy", "content": "chunk 3", "score": 0.7},
        ]
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 1.0)

    def test_mixed_docs_partial_score(self) -> None:
        results = [
            self._chunk("kb-refund"),
            self._chunk("kb-shipping"),
            self._chunk("kb-shipping"),
        ]
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 1 / 3)

    def test_no_match_returns_zero(self) -> None:
        results = [self._chunk("kb-shipping")] * 3
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 0.0)

    def test_none_expected_returns_zero(self) -> None:
        self.assertAlmostEqual(self._fn([self._chunk("kb-refund")], None), 0.0)

    def test_empty_results_returns_zero(self) -> None:
        self.assertAlmostEqual(self._fn([], "kb-refund"), 0.0)

    def test_only_top_k_considered(self) -> None:
        results = [
            self._chunk("kb-shipping"),
            self._chunk("kb-shipping"),
            self._chunk("kb-shipping"),
            self._chunk("kb-refund"),  # 4th slot, outside k=3
        ]
        self.assertAlmostEqual(self._fn(results, "kb-refund", k=3), 0.0)


class RecallAtKDocTests(unittest.TestCase):
    """_recall_at_k_doc: binary hit whether any of the top-k chunks is from the expected doc."""

    def setUp(self) -> None:
        from backend.eval.run import _recall_at_k_doc

        self._fn = _recall_at_k_doc

    def _chunk(self, doc_id: str) -> dict:
        return {"id": doc_id, "title": "Any", "content": "x", "score": 0.5}

    def test_found_in_top3_scores_one(self) -> None:
        results = [self._chunk("kb-shipping"), self._chunk("kb-refund"), self._chunk("kb-blender")]
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 1.0)

    def test_not_found_scores_zero(self) -> None:
        results = [self._chunk("kb-shipping")] * 3
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 0.0)

    def test_binary_not_fractional(self) -> None:
        results = [self._chunk("kb-refund")] * 3
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 1.0)

    def test_none_expected_returns_zero(self) -> None:
        self.assertAlmostEqual(self._fn([self._chunk("kb-refund")], None), 0.0)

    def test_empty_results_returns_zero(self) -> None:
        self.assertAlmostEqual(self._fn([], "kb-refund"), 0.0)


class ContextRelevanceTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.run import _context_relevance

        self._fn = _context_relevance

    def _unit_vec(self, dim: int, nonzero_idx: int) -> list[float]:
        v = [0.0] * dim
        v[nonzero_idx] = 1.0
        return v

    def test_identical_vectors_score_one(self) -> None:
        v = self._unit_vec(4, 0)
        chunk = {"content": "x" * 80, "score": 0.5}
        chunk_embs = {("x" * 80)[:80]: v}
        score = self._fn(v, [chunk], chunk_embs)
        self.assertAlmostEqual(score, 1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        q_vec = self._unit_vec(4, 0)
        c_vec = self._unit_vec(4, 1)
        chunk = {"content": "x" * 80, "score": 0.5}
        chunk_embs = {("x" * 80)[:80]: c_vec}
        score = self._fn(q_vec, [chunk], chunk_embs)
        self.assertAlmostEqual(score, 0.0)

    def test_no_query_embedding_returns_zero(self) -> None:
        chunk = {"content": "x", "score": 0.5}
        self.assertAlmostEqual(self._fn(None, [chunk], {}), 0.0)

    def test_empty_results_returns_zero(self) -> None:
        v = self._unit_vec(4, 0)
        self.assertAlmostEqual(self._fn(v, [], {}), 0.0)

    def test_missing_chunk_embedding_skipped(self) -> None:
        v = self._unit_vec(4, 0)
        chunk = {"content": "no match key", "score": 0.5}
        # chunk_embs keyed differently — no overlap
        self.assertAlmostEqual(self._fn(v, [chunk], {}), 0.0)


class AnswerCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.run import _answer_correctness

        self._fn = _answer_correctness

    def test_all_keywords_present(self) -> None:
        self.assertAlmostEqual(
            self._fn("refund within 30 days of delivery", ["refund", "30 days", "delivery"]), 1.0
        )

    def test_no_keywords_present(self) -> None:
        self.assertAlmostEqual(self._fn("we cannot help you", ["refund", "30 days"]), 0.0)

    def test_partial_match(self) -> None:
        self.assertAlmostEqual(self._fn("refund available", ["refund", "30 days"]), 0.5)

    def test_empty_keywords_returns_one(self) -> None:
        self.assertAlmostEqual(self._fn("any text", []), 1.0)

    def test_case_insensitive(self) -> None:
        self.assertAlmostEqual(self._fn("REFUND within 30 DAYS", ["refund", "30 days"]), 1.0)


class EstimateCostTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.run import _estimate_cost

        self._fn = _estimate_cost

    def test_no_rerank_has_zero_rerank_cost(self) -> None:
        result = self._fn(10, uses_rerank=False)
        self.assertEqual(result["rerank_cost_usd"], 0.0)

    def test_rerank_adds_cost(self) -> None:
        no_rerank = self._fn(10, uses_rerank=False)
        with_rerank = self._fn(10, uses_rerank=True)
        self.assertGreater(with_rerank["total_cost_usd"], no_rerank["total_cost_usd"])

    def test_total_equals_embed_plus_rerank(self) -> None:
        result = self._fn(10, uses_rerank=True)
        self.assertAlmostEqual(
            result["total_cost_usd"],
            result["embed_cost_usd"] + result["rerank_cost_usd"],
            places=6,
        )

    def test_zero_queries_zero_cost(self) -> None:
        result = self._fn(0, uses_rerank=False)
        self.assertEqual(result["total_cost_usd"], 0.0)

    def test_cost_scales_linearly(self) -> None:
        c1 = self._fn(1, uses_rerank=False)["embed_cost_usd"]
        c10 = self._fn(10, uses_rerank=False)["embed_cost_usd"]
        self.assertAlmostEqual(c10, c1 * 10, places=8)


# ---------------------------------------------------------------------------
# 2. LLM-judge tests
# ---------------------------------------------------------------------------


class LlmJudgeContextRelevanceTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.run import _llm_judge_context_relevance

        self._fn = _llm_judge_context_relevance

    def _mock_response(self, text: str) -> MagicMock:
        block = MagicMock()
        block.text = text
        response = MagicMock()
        response.content = [block]
        return response

    def _patch_client(self, response_text: str):
        client = MagicMock()
        client.messages.create.return_value = self._mock_response(response_text)
        return patch("anthropic.Anthropic", return_value=client)

    def test_valid_response_parses_score(self) -> None:
        with self._patch_client('{"score": 0.9, "reason": "correct"}'):
            score, cost = self._fn("q", "ctx", "answer", "key")
        self.assertAlmostEqual(score, 0.9)
        self.assertGreater(cost, 0.0)

    def test_score_clamped_above_one(self) -> None:
        with self._patch_client('{"score": 1.5, "reason": "too high"}'):
            score, _ = self._fn("q", "ctx", "answer", "key")
        self.assertAlmostEqual(score, 1.0)

    def test_score_clamped_below_zero(self) -> None:
        with self._patch_client('{"score": -0.2, "reason": "negative"}'):
            score, _ = self._fn("q", "ctx", "answer", "key")
        self.assertAlmostEqual(score, 0.0)

    def test_malformed_json_returns_zero(self) -> None:
        with self._patch_client("not json at all"):
            score, cost = self._fn("q", "ctx", "answer", "key")
        self.assertAlmostEqual(score, 0.0)
        self.assertAlmostEqual(cost, 0.0)

    def test_api_exception_returns_zero(self) -> None:
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("network error")
        with patch("anthropic.Anthropic", return_value=client):
            score, cost = self._fn("q", "ctx", "answer", "key")
        self.assertAlmostEqual(score, 0.0)
        self.assertAlmostEqual(cost, 0.0)

    def test_missing_score_key_returns_zero(self) -> None:
        with self._patch_client('{"reason": "no score key"}'):
            score, _ = self._fn("q", "ctx", "answer", "key")
        self.assertAlmostEqual(score, 0.0)


# ---------------------------------------------------------------------------
# 3. Benchmark markdown generator
# ---------------------------------------------------------------------------


def _make_mode_result(mode: str, llm_score: float | None = None) -> dict:
    r = {
        "mode": mode,
        "avg_precision_at_3": 0.75,
        "avg_recall_at_3": 0.80,
        "avg_precision_at_3_doc": 0.85,
        "avg_recall_at_3_doc": 0.90,
        "avg_hit_rate_at_1": 0.70,
        "avg_hit_rate_at_3": 0.88,
        "avg_hit_rate_at_5": 0.92,
        "avg_hit_rate_at_10": 0.95,
        "avg_ndcg_at_5": 0.78,
        "avg_mrr": 0.82,
        "avg_context_relevance": 0.60,
        "avg_answer_correctness_kw": 0.65,
        "p50_latency_s": 0.12,
        "p95_latency_s": 0.30,
        "estimated_cost": {"total_cost_usd": 0.000123},
        "n_queries": 30,
        "n_answerable": 25,
    }
    if llm_score is not None:
        r["avg_context_relevance_llm"] = llm_score
    return r


class BenchmarkMdTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.run import _generate_benchmark_md

        self._fn = _generate_benchmark_md
        self._tmpdir = tempfile.TemporaryDirectory()
        self._out = Path(self._tmpdir.name) / "benchmark.md"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_creates_file(self) -> None:
        self._fn([_make_mode_result("keyword")], self._out)
        self.assertTrue(self._out.exists())

    def test_contains_mode_name(self) -> None:
        self._fn([_make_mode_result("hybrid+rerank")], self._out)
        content = self._out.read_text()
        self.assertIn("hybrid+rerank", content)

    def test_contains_header_columns_without_llm(self) -> None:
        self._fn([_make_mode_result("keyword")], self._out)
        lines = self._out.read_text().splitlines()
        # CtxRelLLM column should not appear in any table header row (| ... | lines)
        table_header_lines = [line for line in lines if line.startswith("| Mode")]
        self.assertTrue(table_header_lines, "expected at least one table header row")
        self.assertIn("P@3", self._out.read_text())
        self.assertIn("KwCorr", self._out.read_text())
        for header in table_header_lines:
            self.assertNotIn("CtxRelLLM", header)

    def test_contains_doc_id_columns(self) -> None:
        self._fn([_make_mode_result("hybrid")], self._out)
        content = self._out.read_text()
        self.assertIn("P@3 (doc)", content)
        self.assertIn("R@3 (doc)", content)

    def test_contains_ranking_metrics(self) -> None:
        self._fn([_make_mode_result("hybrid")], self._out)
        content = self._out.read_text()
        self.assertIn("NDCG@5", content)
        self.assertIn("MRR", content)
        self.assertIn("H@1", content)
        self.assertIn("H@10", content)

    def test_deprecated_title_columns_still_present(self) -> None:
        self._fn([_make_mode_result("hybrid")], self._out)
        content = self._out.read_text()
        self.assertIn("P@3 (title)", content)
        self.assertIn("R@3 (title)", content)

    def test_contains_llm_column_when_present(self) -> None:
        self._fn([_make_mode_result("hybrid", llm_score=0.88)], self._out)
        content = self._out.read_text()
        self.assertIn("CtxRelLLM", content)

    def test_multiple_modes_all_appear(self) -> None:
        modes = ["keyword", "fulltext", "hybrid"]
        self._fn([_make_mode_result(m) for m in modes], self._out)
        content = self._out.read_text()
        for m in modes:
            self.assertIn(m, content)

    def test_contains_column_definitions_section(self) -> None:
        self._fn([_make_mode_result("keyword")], self._out)
        content = self._out.read_text()
        self.assertIn("## Column definitions", content)

    def test_contains_regenerating_section(self) -> None:
        self._fn([_make_mode_result("keyword")], self._out)
        content = self._out.read_text()
        self.assertIn("## Regenerating", content)
        self.assertIn("--all-modes --benchmark", content)

    def test_valid_markdown_table_structure(self) -> None:
        # Two tables: ranking (header + sep + 2 data) + quality (header + sep + 2 data) = 8 rows
        self._fn([_make_mode_result("keyword"), _make_mode_result("hybrid")], self._out)
        lines = self._out.read_text().splitlines()
        table_lines = [line for line in lines if line.startswith("|")]
        self.assertEqual(len(table_lines), 8)


# ---------------------------------------------------------------------------
# 4. evaluate_mode output shape (doc-id metrics wired in, no DB required)
# ---------------------------------------------------------------------------


class EvaluateModeDocIdTests(unittest.TestCase):
    """Verify evaluate_mode emits the new doc-id metrics using keyword (InMemory) mode."""

    def _minimal_queries(self) -> list[dict]:
        return [
            {
                "id": "t1",
                "category": "refund",
                "query": "refund",
                "expected_source_title": "Refund policy",
                "expected_document_id": "refund-policy",
                "acceptable_answer_keywords": ["refund"],
            },
            {
                "id": "t2",
                "category": "off-topic",
                "query": "something completely off-topic xyz123",
                "expected_source_title": None,
                "expected_document_id": None,
                "acceptable_answer_keywords": [],
            },
        ]

    def test_result_contains_doc_id_averages(self) -> None:
        from backend.eval.run import evaluate_mode

        result = evaluate_mode(
            mode="keyword",
            queries=self._minimal_queries(),
            database_url="postgresql://unused",
            voyage_api_key=None,
        )
        self.assertIn("avg_precision_at_3_doc", result)
        self.assertIn("avg_recall_at_3_doc", result)
        self.assertIsInstance(result["avg_precision_at_3_doc"], float)
        self.assertIsInstance(result["avg_recall_at_3_doc"], float)

    def test_per_query_row_contains_doc_id_fields(self) -> None:
        from backend.eval.run import evaluate_mode

        result = evaluate_mode(
            mode="keyword",
            queries=self._minimal_queries(),
            database_url="postgresql://unused",
            voyage_api_key=None,
        )
        for row in result["per_query"]:
            self.assertIn("precision_at_3_doc", row)
            self.assertIn("recall_at_3_doc", row)
            self.assertIn("retrieved_doc_ids", row)
            self.assertIn("expected_doc_id", row)

    def test_correct_retrieval_scores_doc_precision_one(self) -> None:
        from backend.eval.run import evaluate_mode

        # "refund" query should retrieve refund-policy from InMemoryRepository
        result = evaluate_mode(
            mode="keyword",
            queries=[self._minimal_queries()[0]],
            database_url="postgresql://unused",
            voyage_api_key=None,
        )
        row = result["per_query"][0]
        # refund-policy should appear in retrieved doc IDs
        self.assertIn("refund-policy", row["retrieved_doc_ids"])
        self.assertGreater(row["precision_at_3_doc"], 0.0)
        self.assertEqual(row["recall_at_3_doc"], 1.0)


# ---------------------------------------------------------------------------
# 5. Regression gate (check_regression)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5b. 9c: hit_rate@k, NDCG@5, MRR
# ---------------------------------------------------------------------------


class HitRateAtKTests(unittest.TestCase):
    """hit_rate_at_k: binary 1.0 if expected doc appears in top-k, else 0."""

    def setUp(self) -> None:
        from backend.eval.run import hit_rate_at_k

        self._fn = hit_rate_at_k

    def _chunk(self, doc_id: str) -> dict:
        return {"id": doc_id, "title": "T", "content": "x", "score": 0.5}

    def test_found_at_rank_1(self) -> None:
        self.assertEqual(self._fn([self._chunk("kb-refund")] * 3, "kb-refund", k=1), 1.0)

    def test_found_beyond_k_scores_zero(self) -> None:
        results = [self._chunk("kb-shipping")] * 3 + [self._chunk("kb-refund")]
        self.assertEqual(self._fn(results, "kb-refund", k=3), 0.0)

    def test_found_within_k_scores_one(self) -> None:
        results = [self._chunk("kb-shipping"), self._chunk("kb-refund"), self._chunk("kb-blender")]
        self.assertEqual(self._fn(results, "kb-refund", k=3), 1.0)

    def test_none_expected_returns_zero(self) -> None:
        self.assertEqual(self._fn([self._chunk("kb-refund")], None, k=5), 0.0)

    def test_empty_results_returns_zero(self) -> None:
        self.assertEqual(self._fn([], "kb-refund", k=5), 0.0)

    def test_k_equals_10_finds_result_at_rank_8(self) -> None:
        results = (
            [self._chunk("kb-shipping")] * 7
            + [self._chunk("kb-refund")]
            + [self._chunk("kb-blender")] * 2
        )
        self.assertEqual(self._fn(results, "kb-refund", k=10), 1.0)


class NdcgAtKTests(unittest.TestCase):
    """ndcg_at_k: single-relevant-doc NDCG. IDCG=1 (ideal rank is 1)."""

    def setUp(self) -> None:
        from backend.eval.run import ndcg_at_k

        self._fn = ndcg_at_k

    def _chunk(self, doc_id: str) -> dict:
        return {"id": doc_id, "title": "T", "content": "x", "score": 0.5}

    def test_found_at_rank_1_scores_one(self) -> None:
        results = [self._chunk("kb-refund"), self._chunk("kb-shipping"), self._chunk("kb-blender")]
        self.assertAlmostEqual(self._fn(results, "kb-refund", k=5), 1.0)

    def test_found_at_rank_2_discounted(self) -> None:
        import math

        results = [self._chunk("kb-shipping"), self._chunk("kb-refund"), self._chunk("kb-blender")]
        expected = 1.0 / math.log2(3)  # rank 2 → DCG = 1/log2(2+1), IDCG = 1/log2(2) = 1
        self.assertAlmostEqual(self._fn(results, "kb-refund", k=5), expected, places=5)

    def test_not_found_scores_zero(self) -> None:
        results = [self._chunk("kb-shipping")] * 5
        self.assertAlmostEqual(self._fn(results, "kb-refund", k=5), 0.0)

    def test_found_beyond_k_scores_zero(self) -> None:
        results = [self._chunk("kb-shipping")] * 3 + [self._chunk("kb-refund")]
        self.assertAlmostEqual(self._fn(results, "kb-refund", k=3), 0.0)

    def test_none_expected_returns_zero(self) -> None:
        self.assertAlmostEqual(self._fn([self._chunk("kb-refund")], None, k=5), 0.0)


class MrrTests(unittest.TestCase):
    """mrr: 1/rank of first relevant chunk found; 0 if none."""

    def setUp(self) -> None:
        from backend.eval.run import mrr

        self._fn = mrr

    def _chunk(self, doc_id: str) -> dict:
        return {"id": doc_id, "title": "T", "content": "x", "score": 0.5}

    def test_found_at_rank_1_scores_one(self) -> None:
        results = [self._chunk("kb-refund"), self._chunk("kb-shipping")]
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 1.0)

    def test_found_at_rank_2_scores_half(self) -> None:
        results = [self._chunk("kb-shipping"), self._chunk("kb-refund")]
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 0.5)

    def test_found_at_rank_3_scores_third(self) -> None:
        results = [self._chunk("kb-shipping"), self._chunk("kb-blender"), self._chunk("kb-refund")]
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 1 / 3)

    def test_not_found_scores_zero(self) -> None:
        results = [self._chunk("kb-shipping")] * 3
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 0.0)

    def test_first_match_counts_not_all(self) -> None:
        # Two matching chunks: rank 1 and rank 3 — should return 1/1 = 1.0
        results = [self._chunk("kb-refund"), self._chunk("kb-shipping"), self._chunk("kb-refund")]
        self.assertAlmostEqual(self._fn(results, "kb-refund"), 1.0)

    def test_none_expected_returns_zero(self) -> None:
        self.assertAlmostEqual(self._fn([self._chunk("kb-refund")], None), 0.0)

    def test_empty_results_returns_zero(self) -> None:
        self.assertAlmostEqual(self._fn([], "kb-refund"), 0.0)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class RegressionGateTests(unittest.TestCase):
    """Tests check_regression.main() and save_baseline() in isolation using temp dirs."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._results_dir = self._root / "results"
        self._results_dir.mkdir()
        self._thresholds_path = self._root / "thresholds.json"
        self._baseline_path = self._results_dir / "baseline.json"

        self._thresholds = {
            "regression_max_drop": 0.10,
            "metrics_to_gate": ["avg_precision_at_3", "avg_recall_at_3"],
            "best_mode": "hybrid",
            "agent_metrics_to_gate": ["avg_tool_accuracy"],
            "agent_regression_max_drop": 0.10,
        }
        _write_json(self._thresholds_path, self._thresholds)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _patch_paths(self):
        """Patch THRESHOLDS_PATH, RESULTS_DIR, and BASELINE_PATH in check_regression."""
        import backend.eval.check_regression as cr

        return (
            patch.object(cr, "THRESHOLDS_PATH", self._thresholds_path),
            patch.object(cr, "RESULTS_DIR", self._results_dir),
            patch.object(cr, "BASELINE_PATH", self._baseline_path),
        )

    def _run_main(self) -> int:
        import backend.eval.check_regression as cr

        p1, p2, p3 = self._patch_paths()
        with p1, p2, p3:
            try:
                cr.main()
                return 0
            except SystemExit as e:
                return int(e.code)

    def _write_current(self, precision: float, recall: float) -> None:
        _write_json(
            self._results_dir / "hybrid.json",
            {"avg_precision_at_3": precision, "avg_recall_at_3": recall},
        )

    def _write_baseline(
        self, precision: float, recall: float, tool_acc: float | None = None
    ) -> None:
        data: dict = {"retrieval": {"avg_precision_at_3": precision, "avg_recall_at_3": recall}}
        if tool_acc is not None:
            data["agent"] = {"avg_tool_accuracy": tool_acc}
        _write_json(self._baseline_path, data)

    def test_no_regression_exits_zero(self) -> None:
        self._write_baseline(0.75, 0.80)
        self._write_current(0.75, 0.80)
        self.assertEqual(self._run_main(), 0)

    def test_small_improvement_exits_zero(self) -> None:
        self._write_baseline(0.70, 0.75)
        self._write_current(0.80, 0.85)
        self.assertEqual(self._run_main(), 0)

    def test_precision_drop_within_threshold_passes(self) -> None:
        # 0.09 drop is just under the 0.10 threshold
        self._write_baseline(0.80, 0.80)
        self._write_current(0.71, 0.80)
        self.assertEqual(self._run_main(), 0)

    def test_precision_drop_beyond_threshold_fails(self) -> None:
        # 0.11 drop exceeds threshold
        self._write_baseline(0.80, 0.80)
        self._write_current(0.69, 0.80)
        self.assertEqual(self._run_main(), 1)

    def test_recall_drop_beyond_threshold_fails(self) -> None:
        self._write_baseline(0.80, 0.80)
        self._write_current(0.80, 0.60)
        self.assertEqual(self._run_main(), 1)

    def test_missing_baseline_exits_zero(self) -> None:
        # No baseline.json — should skip check, not fail
        self._write_current(0.75, 0.80)
        self.assertEqual(self._run_main(), 0)

    def test_missing_results_file_warns_but_passes(self) -> None:
        # Baseline exists, but hybrid.json does not
        self._write_baseline(0.75, 0.80)
        # No current results file written — check_regression warns but exits 0
        self.assertEqual(self._run_main(), 0)

    def test_agent_metric_regression_fails(self) -> None:
        self._write_baseline(0.75, 0.80, tool_acc=0.90)
        self._write_current(0.75, 0.80)
        # Write agent eval with a large drop
        _write_json(
            self._results_dir / "agent_eval.json",
            {"avg_tool_accuracy": 0.70},
        )
        self.assertEqual(self._run_main(), 1)

    def test_agent_metric_no_regression_passes(self) -> None:
        self._write_baseline(0.75, 0.80, tool_acc=0.90)
        self._write_current(0.75, 0.80)
        _write_json(
            self._results_dir / "agent_eval.json",
            {"avg_tool_accuracy": 0.92},
        )
        self.assertEqual(self._run_main(), 0)

    def test_save_baseline_writes_file(self) -> None:
        import backend.eval.check_regression as cr

        self._write_current(0.75, 0.80)
        _write_json(
            self._results_dir / "agent_eval.json",
            {"avg_tool_accuracy": 0.88},
        )
        p1, p2, p3 = self._patch_paths()
        with p1, p2, p3:
            cr.save_baseline()

        self.assertTrue(self._baseline_path.exists())
        saved = json.loads(self._baseline_path.read_text())
        self.assertIn("retrieval", saved)
        self.assertAlmostEqual(saved["retrieval"]["avg_precision_at_3"], 0.75)
        self.assertIn("agent", saved)
        self.assertAlmostEqual(saved["agent"]["avg_tool_accuracy"], 0.88)


# ---------------------------------------------------------------------------
# 6. Agent eval scoring formulas
# ---------------------------------------------------------------------------


def _make_handle_message_result(tool_names: list[str], reply: str) -> dict:
    return {
        "reply": reply,
        "tool_events": [{"name": t, "input": {}, "output": {}} for t in tool_names],
    }


class AgentEvalScoringTests(unittest.TestCase):
    """Tests for run_agent_eval metric formulas.

    handle_message is mocked so no Claude API or DB calls occur.
    Each test controls exactly which tools were called and what reply was returned,
    then asserts the scoring logic produces the correct values.
    """

    def _run_with_fixtures(
        self,
        fixtures: list[dict],
        tool_sequences: list[list[str]],
        replies: list[str],
    ) -> dict:
        """Run run_agent_eval with patched fixtures file and handle_message."""
        from backend.eval.run import run_agent_eval

        call_iter = iter(
            _make_handle_message_result(tools, reply)
            for tools, reply in zip(tool_sequences, replies)
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(fixtures, f)
            fixtures_path = Path(f.name)

        # run_agent_eval does `from backend.app.agent import SESSION_MEMORY, handle_message`
        # inside the function body, so the patch target must be the source module.
        with (
            patch("backend.eval.run.FIXTURES_PATH", fixtures_path),
            patch("backend.app.agent.handle_message", side_effect=lambda *a, **kw: next(call_iter)),
            patch("backend.app.agent.SESSION_MEMORY", {}),
        ):
            result = run_agent_eval("fake-key", None, None)

        fixtures_path.unlink(missing_ok=True)
        return result

    def test_all_expected_tools_called_scores_one(self) -> None:
        fixtures = [
            {
                "id": "t1",
                "description": "d",
                "turns": [{"role": "user", "content": "hi"}],
                "expected_tools": ["lookup_order"],
                "should_refuse": False,
            }
        ]
        result = self._run_with_fixtures(fixtures, [["lookup_order"]], ["Order found."])
        self.assertAlmostEqual(result["per_fixture"][0]["tool_accuracy"], 1.0)

    def test_no_expected_tools_no_calls_scores_one(self) -> None:
        fixtures = [
            {
                "id": "t2",
                "description": "d",
                "turns": [{"role": "user", "content": "hi"}],
                "expected_tools": [],
                "should_refuse": False,
            }
        ]
        result = self._run_with_fixtures(fixtures, [[]], ["Hello!"])
        self.assertAlmostEqual(result["per_fixture"][0]["tool_accuracy"], 1.0)

    def test_no_expected_tools_but_called_scores_zero(self) -> None:
        fixtures = [
            {
                "id": "t3",
                "description": "d",
                "turns": [{"role": "user", "content": "hi"}],
                "expected_tools": [],
                "should_refuse": False,
            }
        ]
        result = self._run_with_fixtures(fixtures, [["lookup_order"]], ["Found."])
        self.assertAlmostEqual(result["per_fixture"][0]["tool_accuracy"], 0.0)

    def test_partial_tool_hit_scores_half(self) -> None:
        fixtures = [
            {
                "id": "t4",
                "description": "d",
                "turns": [{"role": "user", "content": "hi"}],
                "expected_tools": ["lookup_order", "request_refund"],
                "should_refuse": False,
            }
        ]
        result = self._run_with_fixtures(fixtures, [["lookup_order"]], ["Done."])
        self.assertAlmostEqual(result["per_fixture"][0]["tool_accuracy"], 0.5)

    def test_extra_tool_calls_counted(self) -> None:
        fixtures = [
            {
                "id": "t5",
                "description": "d",
                "turns": [{"role": "user", "content": "hi"}],
                "expected_tools": ["lookup_order"],
                "should_refuse": False,
            }
        ]
        # called lookup_order (expected) + create_ticket (unexpected)
        result = self._run_with_fixtures(
            fixtures, [["lookup_order", "create_ticket"]], ["Ticket created."]
        )
        self.assertEqual(result["per_fixture"][0]["extra_tool_calls"], 1)
        self.assertIn("create_ticket", result["per_fixture"][0]["extra_tool_names"])

    def test_refusal_correct_when_refuse_keyword_in_reply(self) -> None:
        fixtures = [
            {
                "id": "t6",
                "description": "d",
                "turns": [{"role": "user", "content": "hi"}],
                "expected_tools": ["lookup_order"],
                "should_refuse": True,
            }
        ]
        result = self._run_with_fixtures(
            fixtures, [["lookup_order"]], ["I cannot process this request."]
        )
        self.assertTrue(result["per_fixture"][0]["refusal_correct"])

    def test_refusal_incorrect_when_approved_in_reply(self) -> None:
        fixtures = [
            {
                "id": "t7",
                "description": "d",
                "turns": [{"role": "user", "content": "hi"}],
                "expected_tools": ["lookup_order"],
                "should_refuse": True,
            }
        ]
        # Reply contains "approved" and no refusal keyword — refusal_correct should be False
        result = self._run_with_fixtures(
            fixtures,
            [["lookup_order", "request_refund"]],
            ["Your refund has been approved."],
        )
        self.assertFalse(result["per_fixture"][0]["refusal_correct"])

    def test_averages_computed_correctly(self) -> None:
        fixtures = [
            {
                "id": "a1",
                "description": "d",
                "turns": [{"role": "user", "content": "hi"}],
                "expected_tools": ["lookup_order"],
                "should_refuse": False,
            },
            {
                "id": "a2",
                "description": "d",
                "turns": [{"role": "user", "content": "hi"}],
                "expected_tools": ["lookup_order"],
                "should_refuse": False,
            },
        ]
        # First fixture: hits. Second: misses.
        result = self._run_with_fixtures(
            fixtures,
            [["lookup_order"], []],
            ["Found.", "Sorry."],
        )
        self.assertAlmostEqual(result["avg_tool_accuracy"], 0.5)
        self.assertEqual(result["n_fixtures"], 2)

    def test_multi_turn_fixture_aggregates_all_tool_calls(self) -> None:
        fixtures = [
            {
                "id": "mt1",
                "description": "d",
                "turns": [
                    {"role": "user", "content": "turn 1"},
                    {"role": "user", "content": "turn 2"},
                ],
                "expected_tools": ["lookup_order", "request_refund"],
                "should_refuse": False,
            }
        ]
        # Turn 1 calls lookup_order, turn 2 calls request_refund
        result = self._run_with_fixtures(
            fixtures,
            [["lookup_order"], ["request_refund"]],
            ["Shipped.", "Refund created."],
        )
        called = result["per_fixture"][0]["called_tools"]
        self.assertIn("lookup_order", called)
        self.assertIn("request_refund", called)
        self.assertAlmostEqual(result["per_fixture"][0]["tool_accuracy"], 1.0)


# ---------------------------------------------------------------------------
# Gap 1 — compute_context_relevance
# ---------------------------------------------------------------------------


def _voyage_key() -> str | None:
    import os
    from pathlib import Path

    val = os.getenv("VOYAGE_API_KEY")
    if val:
        return val
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("VOYAGE_API_KEY="):
                v = line[len("VOYAGE_API_KEY=") :]
                return v if v else None
    return None


def _anthropic_key() -> str | None:
    import os
    from pathlib import Path

    val = os.getenv("ANTHROPIC_API_KEY")
    if val:
        return val
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                v = line[len("ANTHROPIC_API_KEY=") :]
                return v if v else None
    return None


@unittest.skipUnless(_voyage_key(), "Requires VOYAGE_API_KEY")
class ComputeContextRelevanceTests(unittest.TestCase):
    """Gap 1: compute_context_relevance must return real cosine similarity, not 0.0."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._key = _voyage_key()

    def test_relevant_query_chunk_pair_is_nonzero(self) -> None:
        from backend.eval.run import compute_context_relevance

        query = "How long does delivery take?"
        chunk = (
            "Orders usually ship within 2 business days. "
            "Standard delivery takes 3 to 7 business days."
        )
        score = compute_context_relevance(query, chunk, api_key=self._key)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_irrelevant_chunk_scores_below_half(self) -> None:
        from backend.eval.run import compute_context_relevance

        query = "How long does delivery take?"
        chunk = "The portable blender has a 400ml jar and a safety lock."
        score = compute_context_relevance(query, chunk, api_key=self._key)
        self.assertIsNotNone(score)
        self.assertLess(score, 0.5)

    def test_none_api_key_returns_none(self) -> None:
        from backend.eval.run import compute_context_relevance

        result = compute_context_relevance("any query", "any chunk", api_key="")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Gap 2 unit — evaluate_mode result has backend field
# ---------------------------------------------------------------------------


class EvaluateModeBackendFieldTests(unittest.TestCase):
    """Gap 2: evaluate_mode result must include a 'backend' field."""

    def _make_queries(self) -> list[dict]:
        return [
            {
                "id": "q1",
                "query": "Does the blender have a safety lock?",
                "category": "product",
                "expected_source_title": "Portable blender guide",
                "expected_document_id": "portable-blender-guide",
                "acceptable_answer_keywords": ["safety lock"],
            }
        ]

    def test_keyword_mode_reports_memory_backend(self) -> None:
        from unittest.mock import patch

        from backend.eval.run import evaluate_mode

        dummy_db = "postgresql://localhost/test"
        with patch("backend.app.repository.InMemoryRepository.search_knowledge", return_value=[]):
            result = evaluate_mode(
                mode="keyword",
                queries=self._make_queries(),
                database_url=dummy_db,
                voyage_api_key=None,
            )
        self.assertIn("backend", result)
        self.assertEqual(result["backend"], "memory")

    def test_evaluate_mode_always_includes_backend_key(self) -> None:
        from backend.eval.run import evaluate_mode

        dummy_db = "postgresql://localhost/test"
        result = evaluate_mode(
            mode="keyword",
            queries=self._make_queries(),
            database_url=dummy_db,
            voyage_api_key=None,
        )
        self.assertIn("backend", result)


if __name__ == "__main__":
    unittest.main()
