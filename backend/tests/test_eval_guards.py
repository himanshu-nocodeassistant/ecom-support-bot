"""Guards against eval-infrastructure rot: stale baselines and fail-open
regression gates. See plans/decisions/eval-audit.md."""

import json
from pathlib import Path

import pytest

from backend.eval import check_regression

EVAL = Path(__file__).parents[1] / "eval"


def _load(name):
    return json.loads((EVAL / name).read_text())


# The regression gate must never silently go fail-open: every metric named
# in thresholds.json's gate lists has to actually be present in baseline.json,
# otherwise check_regression's per-metric `continue` on None skips it and CI
# passes regardless of retrieval quality.
def test_baseline_covers_every_gated_metric():
    thresholds = _load("thresholds.json")
    baseline = _load("results/baseline.json")
    missing = [m for m in thresholds["metrics_to_gate"] if m not in baseline.get("retrieval", {})]
    assert not missing, (
        f"baseline.json lacks gated metrics {missing}; "
        "check_regression skips missing metrics, so the gate is fail-open. "
        "Re-run eval with --save-baseline."
    )
    missing_agent = [
        m for m in thresholds.get("agent_metrics_to_gate", []) if m not in baseline.get("agent", {})
    ]
    assert not missing_agent, f"baseline.json lacks agent metrics {missing_agent}"


def test_baseline_mode_matches_best_mode():
    thresholds = _load("thresholds.json")
    baseline = _load("results/baseline.json")
    assert baseline["retrieval"]["mode"] == thresholds["best_mode"], (
        "Baseline was saved from a different mode than the one the gate guards"
    )


# The in-memory KB (used by keyword/fallback mode) must use the same doc-id
# scheme as queries.json's expected_document_id (the canonical ids the
# Postgres-backed loader derives from knowledge/*.md filenames). If the ids
# don't match, doc-id metrics read 0.0 for in-memory modes regardless of
# whether retrieval actually found the right document.
def test_expected_doc_ids_exist_in_inmemory_kb():
    from backend.app.data import KNOWLEDGE_BASE

    kb_ids = {doc["id"] for doc in KNOWLEDGE_BASE}
    expected = {
        q["expected_document_id"] for q in _load("queries.json") if q.get("expected_document_id")
    }
    unknown = expected - kb_ids
    assert not unknown, (
        f"queries.json expects doc ids absent from the in-memory KB: {sorted(unknown)[:5]}. "
        "Doc-id metrics will read 0.0 for in-memory modes regardless of retrieval quality."
    )


# check_regression.py --strict must exit non-zero when a gated metric is
# missing from the baseline, instead of silently passing CI.
def test_strict_flag_fails_on_missing_gated_metric(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (tmp_path / "thresholds.json").write_text(
        json.dumps(
            {
                "regression_max_drop": 0.10,
                "metrics_to_gate": ["avg_precision_at_3_doc"],
                "best_mode": "hybrid",
                "agent_metrics_to_gate": [],
            }
        )
    )
    (results_dir / "baseline.json").write_text(
        json.dumps(
            {
                "retrieval": {"avg_precision_at_3": 0.1, "mode": "hybrid"},
                "agent": {},
            }
        )
    )
    (results_dir / "hybrid.json").write_text(
        json.dumps(
            {
                "avg_precision_at_3_doc": 0.2,
                "mode": "hybrid",
            }
        )
    )

    monkeypatch.setattr(check_regression, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(check_regression, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(check_regression, "THRESHOLDS_PATH", tmp_path / "thresholds.json")
    monkeypatch.setattr(check_regression, "BASELINE_PATH", results_dir / "baseline.json")

    with pytest.raises(SystemExit) as exc_info:
        check_regression.main(strict=True)
    assert exc_info.value.code == 1
