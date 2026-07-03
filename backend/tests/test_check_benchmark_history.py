"""Tests for the pre-commit guard on docs/benchmark-history.jsonl (6c)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.eval.check_benchmark_history import check


def _entry(n_docs=15, metric_version="doc-id-v1", **overrides):
    e = {
        "timestamp": "2026-07-03T10:00:00",
        "best_mode": "hybrid",
        "avg_ndcg_at_5": 0.3,
        "n_docs": n_docs,
        "metric_version": metric_version,
    }
    e.update(overrides)
    return e


class CheckBenchmarkHistoryTests(unittest.TestCase):
    def test_missing_file_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nonexistent.jsonl"
            self.assertEqual(check(missing), [])

    def test_empty_file_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            path.write_text("")
            self.assertEqual(check(path), [])

    def test_consistent_fingerprint_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            path.write_text("\n".join(json.dumps(_entry()) for _ in range(3)) + "\n")
            self.assertEqual(check(path), [])

    def test_entry_missing_fingerprint_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            entry = _entry()
            del entry["n_docs"]
            path.write_text(json.dumps(entry) + "\n")
            errors = check(path)
            self.assertTrue(errors)
            self.assertIn("n_docs", errors[0])

    def test_mixed_fingerprints_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            lines = [
                json.dumps(_entry(n_docs=62, metric_version="title-match-v1")),
                json.dumps(_entry(n_docs=15, metric_version="doc-id-v1")),
            ]
            path.write_text("\n".join(lines) + "\n")
            errors = check(path)
            self.assertTrue(any("mixes" in e for e in errors))

    def test_invalid_json_line_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            path.write_text("not json\n")
            errors = check(path)
            self.assertTrue(any("invalid JSON" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
