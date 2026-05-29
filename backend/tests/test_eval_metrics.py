"""Unit tests for phase 6 eval infrastructure.

Covers five areas — all without external deps (no DB, no Voyage, no Claude API):
  1. MetricHelpers     — _precision_at_k, _recall_at_k, _context_relevance,
                         _answer_correctness, _estimate_cost
  2. LlmJudge         — _llm_judge_correctness: valid response, score clamping,
                         malformed JSON, API exception
  3. BenchmarkMd      — _generate_benchmark_md: file structure with and without
                         LLM column
  4. RegressionGate   — check_regression logic: pass, fail, missing baseline,
                         missing results file, agent metrics
  5. AgentEvalScoring — run_agent_eval metric formulas isolated from the
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


class LlmJudgeCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.run import _llm_judge_correctness

        self._fn = _llm_judge_correctness

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
        "avg_context_relevance": 0.60,
        "avg_answer_correctness_kw": 0.65,
        "p50_latency_s": 0.12,
        "p95_latency_s": 0.30,
        "estimated_cost": {"total_cost_usd": 0.000123},
        "n_queries": 30,
        "n_answerable": 25,
    }
    if llm_score is not None:
        r["avg_answer_correctness_llm"] = llm_score
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
        # The markdown table header row is the first line starting with "| Mode"
        header_row = next(line for line in lines if line.startswith("| Mode"))
        self.assertIn("P@3", header_row)
        self.assertIn("KwCorr", header_row)
        self.assertNotIn("LLMCorr", header_row)

    def test_contains_llm_column_when_present(self) -> None:
        self._fn([_make_mode_result("hybrid", llm_score=0.88)], self._out)
        content = self._out.read_text()
        self.assertIn("LLMCorr", content)

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
        self._fn([_make_mode_result("keyword"), _make_mode_result("hybrid")], self._out)
        lines = self._out.read_text().splitlines()
        table_lines = [line for line in lines if line.startswith("|")]
        # header row + separator row + 2 data rows
        self.assertEqual(len(table_lines), 4)


# ---------------------------------------------------------------------------
# 4. Regression gate (check_regression)
# ---------------------------------------------------------------------------


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
# 5. Agent eval scoring formulas
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


if __name__ == "__main__":
    unittest.main()
