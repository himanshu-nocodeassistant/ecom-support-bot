"""Tests for 9g: benchmark trend history (docs/benchmark-history.jsonl + sparklines)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def _make_result(mode: str = "hybrid", ndcg: float = 0.6, cost: float = 0.001) -> dict:
    return {
        "mode": mode,
        "avg_ndcg_at_5": ndcg,
        "avg_precision_at_3_doc": 0.5,
        "avg_recall_at_3_doc": 0.8,
        "avg_hit_rate_at_3": 0.7,
        "avg_mrr": 0.6,
        "avg_precision_at_3": 0.3,
        "avg_recall_at_3": 0.7,
        "avg_hit_rate_at_1": 0.5,
        "avg_hit_rate_at_5": 0.75,
        "avg_hit_rate_at_10": 0.85,
        "avg_context_relevance": 0.5,
        "avg_answer_correctness_kw": 0.6,
        "p50_latency_s": 0.05,
        "p95_latency_s": 0.10,
        "estimated_cost": {"total_cost_usd": cost},
        "n_queries": 30,
        "n_answerable": 25,
    }


class AppendBenchmarkHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.run import append_benchmark_history

        self._fn = append_benchmark_history

    def test_creates_file_if_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "benchmark-history.jsonl"
            self._fn([_make_result()], history_path=history_path)
            self.assertTrue(history_path.exists())

    def test_appends_one_line_per_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "benchmark-history.jsonl"
            self._fn([_make_result("hybrid", ndcg=0.5)], history_path=history_path)
            self._fn([_make_result("hybrid", ndcg=0.6)], history_path=history_path)
            lines = history_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)

    def test_each_line_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "benchmark-history.jsonl"
            self._fn([_make_result()], history_path=history_path)
            for line in history_path.read_text().strip().splitlines():
                data = json.loads(line)
                self.assertIsInstance(data, dict)

    def test_entry_contains_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "benchmark-history.jsonl"
            self._fn([_make_result()], history_path=history_path)
            entry = json.loads(history_path.read_text().strip())
            self.assertIn("timestamp", entry)

    def test_entry_contains_best_mode_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "benchmark-history.jsonl"
            self._fn([_make_result("hybrid", ndcg=0.72)], history_path=history_path)
            entry = json.loads(history_path.read_text().strip())
            self.assertIn("avg_ndcg_at_5", entry)
            self.assertAlmostEqual(entry["avg_ndcg_at_5"], 0.72)

    def test_multiple_modes_uses_best_ndcg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "benchmark-history.jsonl"
            results = [_make_result("keyword", ndcg=0.4), _make_result("hybrid", ndcg=0.75)]
            self._fn(results, history_path=history_path)
            entry = json.loads(history_path.read_text().strip())
            self.assertAlmostEqual(entry["avg_ndcg_at_5"], 0.75)
            self.assertEqual(entry["best_mode"], "hybrid")


class SparklineTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.run import _generate_sparkline

        self._fn = _generate_sparkline

    def test_returns_string(self) -> None:
        result = self._fn([0.3, 0.5, 0.6, 0.7])
        self.assertIsInstance(result, str)

    def test_empty_list_returns_empty_string(self) -> None:
        result = self._fn([])
        self.assertEqual(result, "")

    def test_single_value_does_not_crash(self) -> None:
        result = self._fn([0.5])
        self.assertIsInstance(result, str)

    def test_contains_svg_element(self) -> None:
        result = self._fn([0.1, 0.5, 0.9])
        self.assertIn("<svg", result)


class BenchmarkMdTrendSectionTests(unittest.TestCase):
    """_generate_benchmark_md includes a Trend section when history exists."""

    def setUp(self) -> None:
        import tempfile

        from backend.eval.run import _generate_benchmark_md

        self._fn = _generate_benchmark_md
        self._tmpdir = tempfile.TemporaryDirectory()
        self._out = Path(self._tmpdir.name) / "benchmark.md"
        self._history = Path(self._tmpdir.name) / "benchmark-history.jsonl"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_history(self, entries: list[dict]) -> None:
        lines = [json.dumps(e) for e in entries]
        self._history.write_text("\n".join(lines) + "\n")

    def test_no_history_no_trend_section(self) -> None:
        # Pass a path that doesn't exist so history lookup returns []
        no_history = Path(self._tmpdir.name) / "nonexistent-history.jsonl"
        self._fn([_make_result()], self._out, history_path=no_history)
        content = self._out.read_text()
        self.assertNotIn("## Trend", content)

    def test_with_history_contains_trend_section(self) -> None:
        self._write_history(
            [
                {"timestamp": "2026-01-01T00:00:00", "avg_ndcg_at_5": 0.5, "best_mode": "hybrid"},
                {"timestamp": "2026-01-02T00:00:00", "avg_ndcg_at_5": 0.6, "best_mode": "hybrid"},
            ]
        )
        self._fn([_make_result()], self._out, history_path=self._history)
        content = self._out.read_text()
        self.assertIn("## Trend", content)

    def test_trend_section_contains_sparkline(self) -> None:
        self._write_history(
            [
                {
                    "timestamp": f"2026-01-0{i}T00:00:00",
                    "avg_ndcg_at_5": 0.4 + i * 0.05,
                    "best_mode": "hybrid",
                }
                for i in range(1, 6)
            ]
        )
        self._fn([_make_result()], self._out, history_path=self._history)
        content = self._out.read_text()
        self.assertIn("<svg", content)


# ---------------------------------------------------------------------------
# Gap 6 — chunking decision doc must exist and name a winner
# ---------------------------------------------------------------------------


class ChunkingDecisionDocTests(unittest.TestCase):
    """Gap 6: plans/decisions/chunking.md must exist and state a concluded winner."""

    def test_chunking_decision_doc_exists(self) -> None:
        from pathlib import Path

        path = Path("plans/decisions/chunking.md")
        self.assertTrue(path.exists(), "plans/decisions/chunking.md must exist")

    def test_chunking_decision_names_a_winner(self) -> None:
        from pathlib import Path

        content = Path("plans/decisions/chunking.md").read_text().lower()
        has_conclusion = "winner" in content or "decision" in content
        self.assertTrue(has_conclusion, "chunking.md must state a conclusion")
        names_strategy = "fixed" in content or "semantic" in content
        self.assertTrue(names_strategy, "chunking.md must name the chosen strategy")


# ---------------------------------------------------------------------------
# Gap 2 — benchmark.md template includes backend label
# ---------------------------------------------------------------------------


class BenchmarkMdBackendLabelTests(unittest.TestCase):
    """Gap 2: _generate_benchmark_md must document which backend produced the results."""

    def setUp(self) -> None:
        import tempfile

        from backend.eval.run import _generate_benchmark_md

        self._fn = _generate_benchmark_md
        self._tmpdir = tempfile.TemporaryDirectory()
        self._out = Path(self._tmpdir.name) / "benchmark.md"
        self._no_history = Path(self._tmpdir.name) / "no-history.jsonl"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _result(self, backend: str = "postgres") -> dict:
        r = _make_result()
        r["backend"] = backend
        return r

    def test_postgres_backend_appears_in_benchmark_md(self) -> None:
        self._fn([self._result("postgres")], self._out, history_path=self._no_history)
        content = self._out.read_text().lower()
        self.assertIn("postgres", content)

    def test_memory_backend_appears_in_benchmark_md(self) -> None:
        self._fn([self._result("memory")], self._out, history_path=self._no_history)
        content = self._out.read_text().lower()
        self.assertIn("in-memory", content)

    def test_results_from_line_present(self) -> None:
        self._fn([self._result("postgres")], self._out, history_path=self._no_history)
        content = self._out.read_text()
        self.assertIn("Results from:", content)


if __name__ == "__main__":
    unittest.main()
