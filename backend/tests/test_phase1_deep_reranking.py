"""Phase 1 deep-candidate retrieval contract (red/green TDD)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.app.repository import InMemoryRepository, PostgresRepository


def _fallback() -> InMemoryRepository:
    return InMemoryRepository(
        {},
        [
            {"id": "a", "title": "A", "category": "x", "content": "alpha"},
        ],
    )


def test_reranking_receives_candidate_depth_and_returns_final_depth() -> None:
    repo = PostgresRepository(
        "dsn",
        _fallback(),
        voyage_api_key="key",
        enable_reranking=True,
        candidate_depth=5,
        final_depth=2,
    )
    candidates = [
        {"id": str(i), "content": f"doc {i}", "title": str(i), "score": i} for i in range(5)
    ]
    response = MagicMock()
    response.results = [
        MagicMock(index=4, relevance_score=0.9),
        MagicMock(index=1, relevance_score=0.8),
    ]
    with (
        patch.object(repo, "_hybrid_search", return_value=candidates) as search,
        patch("voyageai.Client") as client,
    ):
        client.return_value.rerank.return_value = response
        result = repo.search_knowledge("query")
    search.assert_called_once_with("query", k=5)
    assert [r["id"] for r in result] == ["4", "1"]
    assert len(result) == 2
    assert client.return_value.rerank.call_args.kwargs["top_k"] == 2
    assert len(client.return_value.rerank.call_args.args[1]) == 5


def test_shallow_corpus_returns_available_results_without_padding() -> None:
    repo = PostgresRepository(
        "dsn",
        _fallback(),
        voyage_api_key="key",
        enable_reranking=True,
        candidate_depth=20,
        final_depth=3,
    )
    candidates = [{"id": "only", "content": "doc", "title": "only", "score": 1}]
    with (
        patch.object(repo, "_hybrid_search", return_value=candidates) as search,
        patch("voyageai.Client") as client,
    ):
        response = MagicMock()
        response.results = [MagicMock(index=0, relevance_score=0.9)]
        client.return_value.rerank.return_value = response
        result = repo.search_knowledge("query")
    search.assert_called_once_with("query", k=20)
    assert len(result) == 1


def test_rerank_failure_is_marked_degraded() -> None:
    repo = PostgresRepository("dsn", _fallback(), voyage_api_key="key", enable_reranking=True)
    candidates = [{"id": "a", "content": "doc", "title": "a", "score": 1}]
    with (
        patch.object(repo, "_hybrid_search", return_value=candidates),
        patch("voyageai.Client") as client,
    ):
        client.return_value.rerank.side_effect = RuntimeError("down")
        result = repo.search_knowledge("query")
    assert result[0]["degraded"] == "rerank_failed"


def test_production_repository_wiring_uses_configured_depths_and_reranking() -> None:
    from backend.app.agent import _repo_for_mode

    settings = SimpleNamespace(
        data_backend="postgres",
        database_url="dsn",
        voyage_api_key="voyage",
        enable_reranking=True,
        retrieval_candidate_depth=17,
        retrieval_final_depth=2,
    )
    with patch("backend.app.agent.get_settings", return_value=settings):
        repo = _repo_for_mode("phase3")
    assert repo.enable_reranking is True
    assert repo.candidate_depth == 17
    assert repo.final_depth == 2


def test_benchmark_explicitly_compares_legacy_and_deep_reranking() -> None:
    from backend.eval.run import _generate_benchmark_md

    results = [
        {
            "mode": "hybrid+rerank",
            "backend": "memory",
            "n_queries": 1,
            "n_answerable": 1,
            "candidate_depth": 3,
            "final_depth": 3,
            "estimated_cost": {"total_cost_usd": 0.001},
            "avg_precision_at_3": 1,
            "avg_recall_at_3": 1,
            "p50_latency_s": 0.1,
            "p95_latency_s": 0.1,
        },
        {
            "mode": "hybrid+rerank-deep",
            "backend": "memory",
            "n_queries": 1,
            "n_answerable": 1,
            "candidate_depth": 20,
            "final_depth": 3,
            "estimated_cost": {"total_cost_usd": 0.002},
            "avg_precision_at_3": 1,
            "avg_recall_at_3": 1,
            "p50_latency_s": 0.1,
            "p95_latency_s": 0.1,
        },
    ]
    output = __import__("tempfile").NamedTemporaryFile(suffix=".md", delete=False).name
    _generate_benchmark_md(results, __import__("pathlib").Path(output))
    content = __import__("pathlib").Path(output).read_text()
    assert "hybrid+rerank" in content and "hybrid+rerank-deep" in content
    assert "3 candidates" in content and "20 candidates" in content
