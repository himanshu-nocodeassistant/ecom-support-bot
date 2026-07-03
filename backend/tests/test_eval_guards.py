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


# H@5/H@10/NDCG@5 are only honest if the eval actually asked retrieval for that
# many candidates, instead of re-slicing the production top-3 list and calling
# it a deeper metric. Rather than asserting every query hit >=5 or >=10 results
# (full-text search legitimately returns fewer rows for sparse queries — an
# honest empty result, not truncation), this guard checks the *configuration*:
# EVAL_DEPTH must be >= 5 (the deepest published metric, NDCG@5/H@5) and
# repository.search_knowledge must honor a k beyond the production default of 3.
# See plans/decisions/eval-audit.md, finding 6f.
def test_eval_depth_covers_published_metrics():
    from backend.eval.run import EVAL_DEPTH

    assert EVAL_DEPTH >= 5, (
        f"EVAL_DEPTH={EVAL_DEPTH} is shallower than the deepest published metric "
        "(NDCG@5/H@5); benchmark.md would publish depth it never retrieved."
    )


def test_repository_search_knowledge_honors_deeper_k():
    from backend.app.data import KNOWLEDGE_BASE, ORDERS
    from backend.app.repository import InMemoryRepository

    repo = InMemoryRepository(orders=ORDERS, knowledge_documents=KNOWLEDGE_BASE)
    shallow = repo.search_knowledge("refund policy return")
    deep = repo.search_knowledge("refund policy return", k=10)
    assert len(shallow) <= 3
    assert len(deep) >= len(shallow), (
        "search_knowledge(k=10) returned fewer results than the default k=3 call; "
        "eval's deep-retrieval pass depends on k actually widening the result set."
    )
