"""Tests for 9f: Pareto visualisation (cost vs NDCG@5, latency vs NDCG@5)."""

from __future__ import annotations

import unittest
from pathlib import Path


def _make_results(n: int = 3) -> list[dict]:
    modes = ["keyword", "fulltext", "hybrid", "hybrid+rerank"][:n]
    return [
        {
            "mode": m,
            "avg_ndcg_at_5": 0.3 * (i + 1) / n,
            "avg_precision_at_3_doc": 0.2,
            "avg_recall_at_3_doc": 0.5,
            "avg_hit_rate_at_1": 0.4,
            "avg_hit_rate_at_3": 0.6,
            "avg_hit_rate_at_5": 0.7,
            "avg_hit_rate_at_10": 0.8,
            "avg_mrr": 0.5,
            "avg_precision_at_3": 0.3,
            "avg_recall_at_3": 0.6,
            "avg_context_relevance": 0.5,
            "avg_answer_correctness_kw": 0.6,
            "p50_latency_s": 0.05 * (i + 1),
            "p95_latency_s": 0.1 * (i + 1),
            "estimated_cost": {"total_cost_usd": 0.001 * (i + 1)},
            "n_queries": 30,
            "n_answerable": 25,
        }
        for i, m in enumerate(modes)
    ]


class ParetoSvgTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.eval.run import _generate_pareto_svgs

        self._fn = _generate_pareto_svgs

    def test_returns_two_svgs(self) -> None:
        cost_svg, latency_svg = self._fn(_make_results())
        self.assertIsInstance(cost_svg, str)
        self.assertIsInstance(latency_svg, str)

    def test_svgs_are_valid_xml(self) -> None:
        import xml.etree.ElementTree as ET

        cost_svg, latency_svg = self._fn(_make_results())
        # Should parse without exception
        ET.fromstring(cost_svg)
        ET.fromstring(latency_svg)

    def test_cost_svg_contains_mode_labels(self) -> None:
        results = _make_results(3)
        cost_svg, _ = self._fn(results)
        for r in results:
            self.assertIn(r["mode"], cost_svg)

    def test_latency_svg_contains_mode_labels(self) -> None:
        results = _make_results(3)
        _, latency_svg = self._fn(results)
        for r in results:
            self.assertIn(r["mode"], latency_svg)

    def test_svgs_start_with_svg_tag(self) -> None:
        cost_svg, latency_svg = self._fn(_make_results())
        self.assertTrue(cost_svg.strip().startswith("<svg"))
        self.assertTrue(latency_svg.strip().startswith("<svg"))

    def test_single_mode_does_not_crash(self) -> None:
        cost_svg, latency_svg = self._fn(_make_results(1))
        self.assertIn("<svg", cost_svg)
        self.assertIn("<svg", latency_svg)


class BenchmarkMdIncludesParetoTests(unittest.TestCase):
    """_generate_benchmark_md embeds the Pareto SVGs when results have NDCG data."""

    def setUp(self) -> None:
        import tempfile

        from backend.eval.run import _generate_benchmark_md

        self._fn = _generate_benchmark_md
        self._tmpdir = tempfile.TemporaryDirectory()
        self._out = Path(self._tmpdir.name) / "benchmark.md"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_benchmark_md_contains_pareto_section(self) -> None:
        self._fn(_make_results(), self._out)
        content = self._out.read_text()
        self.assertIn("## Pareto", content)

    def test_benchmark_md_embeds_svg(self) -> None:
        self._fn(_make_results(), self._out)
        content = self._out.read_text()
        self.assertIn("<svg", content)

    def test_benchmark_md_has_two_charts(self) -> None:
        self._fn(_make_results(), self._out)
        content = self._out.read_text()
        self.assertIn("Cost", content)
        self.assertIn("Latency", content)


if __name__ == "__main__":
    unittest.main()
