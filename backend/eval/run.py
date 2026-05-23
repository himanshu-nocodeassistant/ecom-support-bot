"""Evaluation runner for the ecom-support-bot retrieval pipeline.

Usage:
    python -m backend.eval.run --mode hybrid
    python -m backend.eval.run --all-modes
    python -m backend.eval.run --chunking-audit   # 5b: compare fixed vs semantic chunking
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).parent
QUERIES_PATH = EVAL_DIR / "queries.json"
RESULTS_DIR = EVAL_DIR / "results"

# Voyage pricing (as of 2025) — approximate $/million tokens
VOYAGE_EMBED_PRICE_PER_M = 0.02
VOYAGE_RERANK_PRICE_PER_M = 0.05
# Average tokens per query (query + chunk text)
AVG_QUERY_TOKENS = 50
AVG_CHUNK_TOKENS = 80


def load_queries() -> list[dict[str, Any]]:
    with QUERIES_PATH.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _precision_at_k(
    retrieved: list[dict[str, Any]], expected_title: str | None, k: int = 3
) -> float:
    """Fraction of top-k retrieved chunks whose doc title matches expected_source_title."""
    if not expected_title:
        return 0.0
    hits = sum(
        1
        for r in retrieved[:k]
        if r.get("title", "").strip().lower() == expected_title.strip().lower()
    )
    return hits / k


def _recall_at_k(retrieved: list[dict[str, Any]], expected_title: str | None, k: int = 3) -> float:
    """Whether the expected document appears at all in the top-k results (binary)."""
    if not expected_title:
        return 0.0
    hits = sum(
        1
        for r in retrieved[:k]
        if r.get("title", "").strip().lower() == expected_title.strip().lower()
    )
    return 1.0 if hits > 0 else 0.0


def _context_relevance(
    query_embedding: list[float] | None,
    retrieved: list[dict[str, Any]],
    chunk_embeddings: dict[str, list[float]],
) -> float:
    """Average cosine similarity between query embedding and retrieved chunk embeddings."""
    if query_embedding is None or not retrieved:
        return 0.0

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    scores = []
    for r in retrieved:
        key = r.get("content", "")[:80]
        emb = chunk_embeddings.get(key)
        if emb:
            scores.append(cosine(query_embedding, emb))
    return sum(scores) / len(scores) if scores else 0.0


def _answer_correctness(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the generated answer."""
    if not keywords:
        return 1.0
    lower = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in lower) / len(keywords)


def _estimate_cost(n_queries: int, uses_rerank: bool) -> dict[str, float]:
    embed_tokens = n_queries * AVG_QUERY_TOKENS
    embed_cost = (embed_tokens / 1_000_000) * VOYAGE_EMBED_PRICE_PER_M
    rerank_cost = 0.0
    if uses_rerank:
        rerank_tokens = n_queries * (AVG_QUERY_TOKENS + 3 * AVG_CHUNK_TOKENS)
        rerank_cost = (rerank_tokens / 1_000_000) * VOYAGE_RERANK_PRICE_PER_M
    return {
        "embed_cost_usd": round(embed_cost, 6),
        "rerank_cost_usd": round(rerank_cost, 6),
        "total_cost_usd": round(embed_cost + rerank_cost, 6),
    }


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------


def evaluate_mode(
    mode: str,
    queries: list[dict[str, Any]],
    database_url: str,
    voyage_api_key: str | None,
) -> dict[str, Any]:
    from backend.app.data import KNOWLEDGE_BASE, ORDERS
    from backend.app.repository import InMemoryRepository, PostgresRepository

    fallback = InMemoryRepository(orders=ORDERS, knowledge_documents=KNOWLEDGE_BASE)

    uses_rerank = False
    if mode == "keyword":
        repo = fallback
    elif mode == "fulltext":
        repo = PostgresRepository(database_url=database_url, fallback=fallback, voyage_api_key=None)
    elif mode == "hybrid":
        repo = PostgresRepository(
            database_url=database_url, fallback=fallback, voyage_api_key=voyage_api_key
        )
    elif mode == "hybrid+rerank":
        uses_rerank = True
        repo = PostgresRepository(
            database_url=database_url,
            fallback=fallback,
            voyage_api_key=voyage_api_key,
            enable_reranking=True,
        )
    elif mode == "hybrid+rerank+filter":
        uses_rerank = True
        repo = PostgresRepository(
            database_url=database_url,
            fallback=fallback,
            voyage_api_key=voyage_api_key,
            enable_reranking=True,
            enable_metadata_filter=True,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Pre-embed all queries for context relevance scoring (batch call)
    query_embeddings: dict[str, list[float]] = {}
    if voyage_api_key and mode not in ("keyword", "fulltext"):
        try:
            from backend.app.data_loader import embed_queries

            texts = [q["query"] for q in queries]
            vecs = embed_queries(texts, api_key=voyage_api_key)
            query_embeddings = {q["query"]: v for q, v in zip(queries, vecs)}
        except Exception:
            pass

    per_query: list[dict[str, Any]] = []
    for q in queries:
        t0 = time.perf_counter()
        retrieved = repo.search_knowledge(q["query"])
        latency = time.perf_counter() - t0

        p3 = _precision_at_k(retrieved, q["expected_source_title"])
        r3 = _recall_at_k(retrieved, q["expected_source_title"])
        ctx_rel = _context_relevance(query_embeddings.get(q["query"]), retrieved, {})

        per_query.append(
            {
                "id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "precision_at_3": round(p3, 4),
                "recall_at_3": round(r3, 4),
                "context_relevance": round(ctx_rel, 4),
                "latency_s": round(latency, 4),
                "retrieved_titles": [r.get("title") for r in retrieved],
                "expected_title": q["expected_source_title"],
                "top_score": round(float(retrieved[0]["score"]), 4) if retrieved else 0.0,
            }
        )

    answerable = [r for r in per_query if r["expected_title"]]

    def _avg(key: str) -> float:
        vals = [r[key] for r in answerable] if answerable else []
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    latencies = sorted(r["latency_s"] for r in per_query)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    cost = _estimate_cost(len(queries), uses_rerank)

    return {
        "mode": mode,
        "avg_precision_at_3": _avg("precision_at_3"),
        "avg_recall_at_3": _avg("recall_at_3"),
        "avg_context_relevance": _avg("context_relevance"),
        "p50_latency_s": round(p50, 4),
        "p95_latency_s": round(p95, 4),
        "estimated_cost": cost,
        "n_queries": len(queries),
        "n_answerable": len(answerable),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Chunking audit (5b)
# ---------------------------------------------------------------------------


def chunking_audit(
    database_url: str,
    voyage_api_key: str | None,
    knowledge_dir: str,
) -> dict[str, Any]:
    """5b: compare fixed-size vs semantic chunking on the eval query set."""
    from backend.app.data import KNOWLEDGE_BASE, ORDERS
    from backend.app.data_loader import import_knowledge_to_postgres
    from backend.app.repository import InMemoryRepository, PostgresRepository

    queries = load_queries()
    strategies = ["fixed", "semantic"]
    results: dict[str, Any] = {}

    for i, strategy in enumerate(strategies):
        if i > 0:
            print("  Waiting 25s for Voyage rate limit...", flush=True)
            time.sleep(25)
        print(f"\n  Importing chunks with strategy={strategy}...", flush=True)
        import_knowledge_to_postgres(
            database_url=database_url,
            knowledge_dir=knowledge_dir,
            voyage_api_key=voyage_api_key,
            chunk_strategy=strategy,
        )

        repo = PostgresRepository(
            database_url=database_url,
            fallback=InMemoryRepository(orders=ORDERS, knowledge_documents=KNOWLEDGE_BASE),
            voyage_api_key=voyage_api_key,
        )

        per_query: list[dict[str, Any]] = []
        for q in queries:
            t0 = time.perf_counter()
            retrieved = repo.search_knowledge(q["query"])
            latency = time.perf_counter() - t0
            p3 = _precision_at_k(retrieved, q["expected_source_title"])
            per_query.append(
                {
                    "id": q["id"],
                    "query": q["query"],
                    "precision_at_3": round(p3, 4),
                    "latency_s": round(latency, 4),
                    "retrieved_titles": [r.get("title") for r in retrieved],
                    "expected_title": q["expected_source_title"],
                }
            )

        answerable = [r for r in per_query if r["expected_title"]]
        avg_p3 = (
            sum(r["precision_at_3"] for r in answerable) / len(answerable) if answerable else 0.0
        )
        latencies_sorted = sorted(r["latency_s"] for r in per_query)
        p50 = latencies_sorted[len(latencies_sorted) // 2]

        results[strategy] = {
            "strategy": strategy,
            "avg_precision_at_3": round(avg_p3, 4),
            "p50_latency_s": round(p50, 4),
            "chunk_count": _count_chunks(database_url),
            "per_query": per_query,
        }
        print(
            f"  strategy={strategy}: avg_precision@3={avg_p3:.3f}  "
            f"p50={p50:.4f}s  chunks={results[strategy]['chunk_count']}"
        )

    winner = max(strategies, key=lambda s: results[s]["avg_precision_at_3"])
    loser = next(s for s in strategies if s != winner)
    results["winner"] = winner
    results["decision"] = (
        f"Strategy '{winner}' wins with avg_precision@3="
        f"{results[winner]['avg_precision_at_3']:.3f} vs "
        f"{results[loser]['avg_precision_at_3']:.3f}"
    )
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_chunks(database_url: str) -> int:
    try:
        import psycopg

        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("select count(*) from knowledge_chunks")
                return cur.fetchone()[0]
    except Exception:
        return -1


def _print_comparison_table(mode_results: list[dict[str, Any]]) -> None:
    cols = f"{'Mode':<28} {'P@3':>6} {'R@3':>6} {'CtxRel':>8} {'P50':>8} {'P95':>8} {'Cost$':>8}"
    print("\n" + cols)
    print("-" * len(cols))
    for r in mode_results:
        cost = r.get("estimated_cost", {}).get("total_cost_usd", 0.0)
        print(
            f"{r['mode']:<28} "
            f"{r['avg_precision_at_3']:>6.3f} "
            f"{r['avg_recall_at_3']:>6.3f} "
            f"{r['avg_context_relevance']:>8.3f} "
            f"{r['p50_latency_s']:>7.4f}s "
            f"{r['p95_latency_s']:>7.4f}s "
            f"{cost:>8.6f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportBot eval runner")
    parser.add_argument(
        "--mode",
        choices=["keyword", "fulltext", "hybrid", "hybrid+rerank", "hybrid+rerank+filter"],
    )
    parser.add_argument("--all-modes", action="store_true", help="Run all retrieval modes")
    parser.add_argument(
        "--chunking-audit", action="store_true", help="5b: fixed vs semantic chunking"
    )
    parser.add_argument("--knowledge-dir", default="backend/knowledge")
    args = parser.parse_args()

    import os

    from dotenv import dotenv_values

    env = dotenv_values(".env")
    database_url = os.getenv("DATABASE_URL") or env.get("DATABASE_URL")
    voyage_api_key = os.getenv("VOYAGE_API_KEY") or env.get("VOYAGE_API_KEY")

    if not database_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_queries()

    if args.chunking_audit:
        print("=== 5b: Chunking Strategy Audit ===")
        result = chunking_audit(database_url, voyage_api_key, args.knowledge_dir)
        out_path = RESULTS_DIR / "chunking_audit.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nDecision: {result['decision']}")
        print(f"Results saved to {out_path}")
        return

    all_modes = ["keyword", "fulltext", "hybrid", "hybrid+rerank", "hybrid+rerank+filter"]
    modes = all_modes if args.all_modes else ([args.mode] if args.mode else None)
    if not modes:
        parser.print_help()
        sys.exit(1)

    all_results = []
    for mode in modes:
        print(f"Running mode={mode}...", flush=True)
        result = evaluate_mode(mode, queries, database_url, voyage_api_key)
        all_results.append(result)
        out_path = RESULTS_DIR / f"{mode.replace('+', '_')}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(
            f"  P@3={result['avg_precision_at_3']:.3f}  "
            f"R@3={result['avg_recall_at_3']:.3f}  "
            f"p50={result['p50_latency_s']:.4f}s  "
            f"cost=${result['estimated_cost']['total_cost_usd']:.6f}"
        )

    if len(all_results) > 1:
        _print_comparison_table(all_results)
        comparison_path = RESULTS_DIR / "comparison.json"
        comparison_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nComparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
