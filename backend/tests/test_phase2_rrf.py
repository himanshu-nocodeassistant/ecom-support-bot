"""Phase 2 RRF retrieval contracts (red/green TDD)."""

from unittest.mock import patch

from backend.app.repository import InMemoryRepository, PostgresRepository


def _repo(**kwargs):
    return PostgresRepository("dsn", InMemoryRepository({}, []), voyage_api_key="v", **kwargs)


def test_rrf_fusion_is_deterministic_duplicate_safe_and_keeps_stage_metadata():
    repo = _repo(retrieval_mode="rrf")
    sparse = [
        {"id": "b", "title": "B", "content": "b", "score": 0.8},
        {"id": "a", "title": "A", "content": "a", "score": 0.7},
    ]
    dense = [
        {"id": "a", "title": "A", "content": "a", "score": 0.9},
        {"id": "c", "title": "C", "content": "c", "score": 0.8},
    ]
    first = repo._rrf_fuse(sparse, dense, k=60)
    second = repo._rrf_fuse(sparse, dense, k=60)
    assert [item["id"] for item in first] == ["a", "b", "c"]
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert len(first) == 3
    assert first[0]["sparse_rank"] == 2
    assert first[0]["dense_rank"] == 1
    assert first[0]["fused_rank"] == 1
    assert first[0]["sparse_score"] == 0.7
    assert first[0]["dense_score"] == 0.9


def test_rrf_fusion_handles_empty_stage_and_ties_by_id():
    repo = _repo(retrieval_mode="rrf")
    fused = repo._rrf_fuse([], [{"id": "z", "title": "Z", "content": "z", "score": 1}], k=60)
    assert [r["id"] for r in fused] == ["z"]
    tied = repo._rrf_fuse([{"id": "b", "content": "b"}], [{"id": "a", "content": "a"}], k=0)
    assert [r["id"] for r in tied] == ["a", "b"]


def test_rrf_search_runs_independent_sparse_and_dense_lists():
    repo = _repo(retrieval_mode="rrf", candidate_depth=5, final_depth=2)
    sparse = [{"id": "s", "title": "S", "content": "s", "score": 0.2}]
    dense = [{"id": "d", "title": "D", "content": "d", "score": 0.9}]
    with (
        patch.object(repo, "_sparse_search", return_value=sparse) as sparse_search,
        patch.object(repo, "_dense_search", return_value=dense) as dense_search,
    ):
        result = repo.search_knowledge("q")
    sparse_search.assert_called_once_with("q", k=5)
    dense_search.assert_called_once_with("q", k=5)
    assert len(result) == 2
    assert {r["id"] for r in result} == {"s", "d"}


def test_rrf_search_reranks_fused_candidates_and_preserves_stage_metadata():
    repo = _repo(retrieval_mode="rrf", enable_reranking=True, candidate_depth=5, final_depth=2)
    fused = [{"id": "a", "content": "a", "fused_rank": 1, "sparse_rank": 1}]
    with (
        patch.object(repo, "_sparse_search", return_value=fused),
        patch.object(repo, "_dense_search", return_value=[]),
        patch.object(repo, "_rrf_fuse", return_value=fused),
        patch.object(repo, "_rerank", return_value=[{**fused[0], "reranked": True}]) as rerank,
    ):
        result = repo.search_knowledge("q")
    rerank.assert_called_once()
    assert rerank.call_args.args[1][0]["fused_rank"] == 1
    assert result[0]["reranked"] is True


def test_rrf_and_weighted_use_same_comparison_settings_and_metrics_recommendation():
    from backend.eval.run import recommend_retrieval_mode

    baseline = {
        "mode": "hybrid",
        "avg_ndcg_at_5": 0.70,
        "p95_latency_s": 1.0,
        "estimated_cost": {"total_cost_usd": 0.01},
        "n_degraded": 0,
        "regression_gates_pass": True,
        "candidate_depth": 20,
        "final_depth": 3,
        "dataset_sha256": "x",
    }
    rrf = {**baseline, "mode": "rrf", "avg_ndcg_at_5": 0.75, "p95_latency_s": 1.05}
    assert recommend_retrieval_mode(baseline, rrf) == "rrf"
    for key, value in (("candidate_depth", 10), ("final_depth", 2), ("dataset_sha256", "y")):
        incompatible = {**rrf, key: value, "p95_latency_s": 1.0}
        assert recommend_retrieval_mode(baseline, incompatible) == "hybrid"
    rrf["p95_latency_s"] = 1.3
    assert recommend_retrieval_mode(baseline, rrf) == "hybrid"


def test_recommendation_accepts_comparable_rrf_rerank_mode_and_requires_gate_status():
    from backend.eval.run import recommend_retrieval_mode

    baseline = {
        "mode": "hybrid+rerank-deep",
        "avg_ndcg_at_5": 0.7,
        "p95_latency_s": 1,
        "estimated_cost": {"total_cost_usd": 0.01},
        "candidate_depth": 20,
        "final_depth": 3,
        "dataset_sha256": "x",
        "reranker": "voyage",
        "reranker_model": "rerank-2-lite",
        "regression_gates_pass": True,
    }
    experiment = {**baseline, "mode": "rrf+rerank", "avg_ndcg_at_5": 0.8}
    assert recommend_retrieval_mode(baseline, experiment) == "rrf+rerank"
    del experiment["regression_gates_pass"]
    assert recommend_retrieval_mode(baseline, experiment) == "hybrid+rerank-deep"


def test_evaluate_mode_reports_dataset_reranker_and_gate_metadata(monkeypatch):
    from backend.eval.run import evaluate_mode

    class FakeRepo:
        def search_knowledge(self, query, k=None):
            return [{"id": "doc", "title": "Doc", "content": "answer", "score": 1}]

    monkeypatch.setattr("backend.app.repository.PostgresRepository", lambda **kwargs: FakeRepo())
    monkeypatch.setattr(
        "backend.app.data_loader.embed_queries", lambda texts, api_key: [[1.0] for _ in texts]
    )
    monkeypatch.setattr(
        "backend.app.data_loader.embed_texts", lambda texts, api_key: [[1.0] for _ in texts]
    )
    monkeypatch.setattr("backend.eval.run.time.sleep", lambda _: None)
    result = evaluate_mode(
        "rrf+rerank",
        [
            {
                "id": "q",
                "category": "x",
                "query": "q",
                "expected_source_title": "Doc",
                "expected_document_id": "doc",
            }
        ],
        "dsn",
        "key",
    )
    assert result["dataset_sha256"]
    assert result["reranker"] == "voyage"
    assert result["reranker_model"] == "rerank-2-lite"
    assert result["regression_gates_pass"] is False


def test_benchmark_output_states_recommended_mode():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from backend.eval.run import _generate_benchmark_md

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "benchmark.md"
        baseline = {
            "mode": "hybrid",
            "avg_ndcg_at_5": 0.7,
            "p95_latency_s": 1,
            "estimated_cost": {"total_cost_usd": 0.01},
            "n_degraded": 0,
            "regression_gates_pass": True,
            "candidate_depth": 20,
            "final_depth": 3,
            "dataset_sha256": "x",
            "n_queries": 1,
            "n_answerable": 1,
        }
        rrf = {**baseline, "mode": "rrf", "avg_ndcg_at_5": 0.8}
        _generate_benchmark_md([baseline, rrf], path)
        assert "Recommended mode: `rrf`" in path.read_text()
