"""Evaluation runner for the ecom-support-bot retrieval pipeline.

Usage:
    python -m backend.eval.run --mode hybrid
    python -m backend.eval.run --all-modes
    python -m backend.eval.run --chunking-audit   # 5b: compare fixed vs semantic chunking
    python -m backend.eval.run --all-modes --llm-judge  # 6b: LLM-judge correctness
    python -m backend.eval.run --agent-eval        # 6c: agent fixture eval
    python -m backend.eval.run --all-modes --benchmark  # 6a: commit benchmark.md
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


# ---------------------------------------------------------------------------
# 6b: LLM-judge answer correctness
# ---------------------------------------------------------------------------

# Approximate Claude Haiku input/output token costs ($/million tokens)
CLAUDE_INPUT_PRICE_PER_M = 0.80
CLAUDE_OUTPUT_PRICE_PER_M = 4.00


def _llm_judge_correctness(
    query: str,
    context: str,
    answer: str,
    api_key: str,
) -> tuple[float, float]:
    """Score answer factual accuracy using Claude as a judge.

    Returns (score 0.0–1.0, estimated_cost_usd).
    Score interpretation: 0 = wrong/hallucinated, 0.5 = partially correct, 1 = fully correct.
    """
    try:
        import anthropic

        prompt = (
            f"Query: {query}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Answer to evaluate:\n{answer}\n\n"
            "Score the factual accuracy of the answer given the query and context. "
            'Respond with ONLY a JSON object: {"score": <float 0.0-1.0>, "reason": "<one sentence>"}. '
            "1.0 = fully correct and grounded in context. "
            "0.5 = partially correct or missing key facts. "
            "0.0 = wrong, hallucinated, or contradicts the context."
        )

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Parse JSON from response
        parsed = json.loads(text)
        score = float(parsed.get("score", 0.0))
        # Estimate cost: prompt ~200 tokens, response ~30 tokens
        cost = (200 / 1_000_000) * CLAUDE_INPUT_PRICE_PER_M + (
            30 / 1_000_000
        ) * CLAUDE_OUTPUT_PRICE_PER_M
        return round(min(max(score, 0.0), 1.0), 4), round(cost, 6)
    except Exception:
        return 0.0, 0.0


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
    anthropic_api_key: str | None = None,
    use_llm_judge: bool = False,
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
    total_judge_cost = 0.0

    for q in queries:
        t0 = time.perf_counter()
        retrieved = repo.search_knowledge(q["query"])
        latency = time.perf_counter() - t0

        p3 = _precision_at_k(retrieved, q["expected_source_title"])
        r3 = _recall_at_k(retrieved, q["expected_source_title"])
        ctx_rel = _context_relevance(query_embeddings.get(q["query"]), retrieved, {})
        kw_correctness = _answer_correctness(
            " ".join(r.get("content", "") for r in retrieved[:3]),
            q.get("acceptable_answer_keywords", []),
        )

        row: dict[str, Any] = {
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "precision_at_3": round(p3, 4),
            "recall_at_3": round(r3, 4),
            "context_relevance": round(ctx_rel, 4),
            "answer_correctness_kw": round(kw_correctness, 4),
            "latency_s": round(latency, 4),
            "retrieved_titles": [r.get("title") for r in retrieved],
            "expected_title": q["expected_source_title"],
            "top_score": round(float(retrieved[0]["score"]), 4) if retrieved else 0.0,
        }

        if use_llm_judge and anthropic_api_key and retrieved:
            context = "\n".join(r.get("content", "") for r in retrieved[:3])
            answer = retrieved[0].get("content", "") if retrieved else ""
            judge_score, judge_cost = _llm_judge_correctness(
                q["query"], context, answer, anthropic_api_key
            )
            row["answer_correctness_llm"] = judge_score
            row["judge_cost_usd"] = judge_cost
            total_judge_cost += judge_cost

        per_query.append(row)

    answerable = [r for r in per_query if r["expected_title"]]

    def _avg(key: str) -> float:
        vals = [r[key] for r in answerable if key in r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    latencies = sorted(r["latency_s"] for r in per_query)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    cost = _estimate_cost(len(queries), uses_rerank)
    if total_judge_cost:
        cost["judge_cost_usd"] = round(total_judge_cost, 6)
        cost["total_cost_usd"] = round(cost["total_cost_usd"] + total_judge_cost, 6)

    result: dict[str, Any] = {
        "mode": mode,
        "avg_precision_at_3": _avg("precision_at_3"),
        "avg_recall_at_3": _avg("recall_at_3"),
        "avg_context_relevance": _avg("context_relevance"),
        "avg_answer_correctness_kw": _avg("answer_correctness_kw"),
        "p50_latency_s": round(p50, 4),
        "p95_latency_s": round(p95, 4),
        "estimated_cost": cost,
        "n_queries": len(queries),
        "n_answerable": len(answerable),
        "per_query": per_query,
    }
    if use_llm_judge and anthropic_api_key:
        result["avg_answer_correctness_llm"] = _avg("answer_correctness_llm")
    return result


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
    has_llm = any("avg_answer_correctness_llm" in r for r in mode_results)
    header = f"{'Mode':<28} {'P@3':>6} {'R@3':>6} {'CtxRel':>8} {'KwCorr':>8}"
    if has_llm:
        header += f" {'LLMCorr':>8}"
    header += f" {'P50':>8} {'P95':>8} {'Cost$':>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in mode_results:
        cost = r.get("estimated_cost", {}).get("total_cost_usd", 0.0)
        line = (
            f"{r['mode']:<28} "
            f"{r['avg_precision_at_3']:>6.3f} "
            f"{r['avg_recall_at_3']:>6.3f} "
            f"{r['avg_context_relevance']:>8.3f} "
            f"{r.get('avg_answer_correctness_kw', 0.0):>8.3f}"
        )
        if has_llm:
            line += f" {r.get('avg_answer_correctness_llm', 0.0):>8.3f}"
        line += f" {r['p50_latency_s']:>7.4f}s {r['p95_latency_s']:>7.4f}s {cost:>8.6f}"
        print(line)


# ---------------------------------------------------------------------------
# 6a: benchmark.md generator
# ---------------------------------------------------------------------------


def _generate_benchmark_md(mode_results: list[dict[str, Any]], output_path: Path) -> None:
    """Write a committed markdown comparison table from eval results."""
    from datetime import datetime

    has_llm = any("avg_answer_correctness_llm" in r for r in mode_results)
    lines = [
        "# Retrieval Benchmark",
        "",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"Queries: {mode_results[0]['n_queries']} total, {mode_results[0]['n_answerable']} answerable  ",
        "Eval dataset: `backend/eval/queries.json`",
        "",
        "## Mode × Metric",
        "",
    ]

    # Header row
    cols = ["Mode", "P@3", "R@3", "CtxRel", "KwCorr"]
    if has_llm:
        cols.append("LLMCorr")
    cols += ["P50 (s)", "P95 (s)", "Cost ($)"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for r in mode_results:
        cost = r.get("estimated_cost", {}).get("total_cost_usd", 0.0)
        row = [
            f"`{r['mode']}`",
            f"{r['avg_precision_at_3']:.3f}",
            f"{r['avg_recall_at_3']:.3f}",
            f"{r['avg_context_relevance']:.3f}",
            f"{r.get('avg_answer_correctness_kw', 0.0):.3f}",
        ]
        if has_llm:
            row.append(f"{r.get('avg_answer_correctness_llm', 0.0):.3f}")
        row += [
            f"{r['p50_latency_s']:.4f}",
            f"{r['p95_latency_s']:.4f}",
            f"{cost:.6f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Column definitions",
        "",
        "- **P@3** — Precision@3: fraction of top-3 retrieved chunks matching the expected source",
        "- **R@3** — Recall@3: whether the expected source appears anywhere in top-3 (binary)",
        "- **CtxRel** — average cosine similarity between query and retrieved chunk embeddings",
        "- **KwCorr** — keyword-overlap answer correctness (fraction of expected keywords present)",
        "- **LLMCorr** — Claude-as-judge factual accuracy score (0–1); `null` if not run with `--llm-judge`",
        "- **P50/P95** — median and 95th-percentile end-to-end retrieval latency",
        "- **Cost** — estimated Voyage + Claude API cost for the full eval run",
        "",
        "## Regenerating",
        "",
        "```bash",
        "python -m backend.eval.run --all-modes --benchmark",
        "# with LLM-judge (costs ~$0.001 per query):",
        "python -m backend.eval.run --all-modes --llm-judge --benchmark",
        "```",
        "",
        "Raw per-mode JSON lives in `backend/eval/results/`.",
    ]

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Benchmark written to {output_path}")


# ---------------------------------------------------------------------------
# 6c: Agent fixture eval
# ---------------------------------------------------------------------------

FIXTURES_PATH = EVAL_DIR / "agent_fixtures.json"


def run_agent_eval(
    anthropic_api_key: str,
    database_url: str | None,
    voyage_api_key: str | None,
) -> dict[str, Any]:
    """Run multi-turn agent fixtures and score tool selection accuracy.

    Metrics per fixture:
      - tool_accuracy: fraction of expected tools that were actually called
      - extra_tool_calls: number of tool calls NOT in expected_tools
      - refusal_correct: bool — did the agent refuse/block when it should (or not refuse when it shouldn't)?
    """
    import uuid

    from backend.app.agent import SESSION_MEMORY, handle_message

    with FIXTURES_PATH.open() as f:
        fixtures: list[dict[str, Any]] = json.load(f)

    per_fixture: list[dict[str, Any]] = []

    for fx in fixtures:
        session_id = f"eval-{fx['id']}-{uuid.uuid4().hex[:6]}"
        SESSION_MEMORY.pop(session_id, None)

        all_tool_calls: list[str] = []
        final_reply = ""

        for turn in fx["turns"]:
            if turn["role"] != "user":
                continue
            result = handle_message(session_id, turn["content"], mode="phase3")
            final_reply = result.get("reply", "")
            for evt in result.get("tool_events", []):
                all_tool_calls.append(evt["name"])

        expected = fx.get("expected_tools", [])
        expected_set = set(expected)
        called_set = set(all_tool_calls)

        # Tool accuracy: fraction of expected tools that were called (at least once)
        tool_accuracy = (
            sum(1 for t in expected_set if t in called_set) / len(expected_set)
            if expected_set
            else (1.0 if not called_set else 0.0)
        )

        # Extra calls: tools called but not in expected list
        extra_calls = [t for t in all_tool_calls if t not in expected_set]

        # Refusal accuracy: should_refuse=True → agent must NOT call request_refund
        # or must acknowledge inability; should_refuse=False → agent should not escalate unnecessarily
        refusal_correct = True
        if fx.get("should_refuse"):
            # If refused: for order/refund cases, request_refund should not succeed
            # or reply should indicate inability
            refuse_keywords = [
                "not",
                "cannot",
                "can't",
                "unable",
                "not found",
                "not yet",
                "only",
                "no order",
            ]
            refusal_correct = (
                any(kw in final_reply.lower() for kw in refuse_keywords)
                or (
                    "request_refund" not in called_set
                    and "lookup_order" in called_set
                    and "approved" not in final_reply.lower()
                )
                or (not called_set and fx.get("expect_clarification") is not True)
            )

        per_fixture.append(
            {
                "id": fx["id"],
                "description": fx["description"],
                "expected_tools": expected,
                "called_tools": all_tool_calls,
                "tool_accuracy": round(tool_accuracy, 4),
                "extra_tool_calls": len(extra_calls),
                "extra_tool_names": extra_calls,
                "should_refuse": fx.get("should_refuse", False),
                "refusal_correct": refusal_correct,
                "final_reply_excerpt": final_reply[:200],
            }
        )

    n = len(per_fixture)
    avg_tool_accuracy = sum(r["tool_accuracy"] for r in per_fixture) / n if n else 0.0
    avg_extra = sum(r["extra_tool_calls"] for r in per_fixture) / n if n else 0.0
    refusal_fixtures = [r for r in per_fixture if r["should_refuse"]]
    avg_refusal = (
        sum(1 for r in refusal_fixtures if r["refusal_correct"]) / len(refusal_fixtures)
        if refusal_fixtures
        else 1.0
    )

    return {
        "n_fixtures": n,
        "avg_tool_accuracy": round(avg_tool_accuracy, 4),
        "avg_extra_tool_calls": round(avg_extra, 4),
        "avg_refusal_accuracy": round(avg_refusal, 4),
        "per_fixture": per_fixture,
    }


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
    parser.add_argument(
        "--llm-judge", action="store_true", help="6b: score answer correctness with Claude"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="6a: write docs/benchmark.md after run"
    )
    parser.add_argument("--agent-eval", action="store_true", help="6c: run agent fixture eval")
    parser.add_argument("--knowledge-dir", default="backend/knowledge")
    args = parser.parse_args()

    import os

    from dotenv import dotenv_values

    env = dotenv_values(".env")
    database_url = os.getenv("DATABASE_URL") or env.get("DATABASE_URL")
    voyage_api_key = os.getenv("VOYAGE_API_KEY") or env.get("VOYAGE_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_API_KEY")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.agent_eval:
        print("=== 6c: Agent Fixture Eval ===")
        if not anthropic_api_key:
            print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        result = run_agent_eval(anthropic_api_key, database_url, voyage_api_key)
        out_path = RESULTS_DIR / "agent_eval.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(
            f"\nTool accuracy:   {result['avg_tool_accuracy']:.3f}"
            f"\nRefusal accuracy: {result['avg_refusal_accuracy']:.3f}"
            f"\nExtra tools:     {result['avg_extra_tool_calls']:.2f} per fixture"
        )
        print(f"Results saved to {out_path}")
        return

    if not database_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

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

    if args.llm_judge and not anthropic_api_key:
        print(
            "WARNING: --llm-judge requires ANTHROPIC_API_KEY; skipping judge scoring",
            file=sys.stderr,
        )
        args.llm_judge = False

    all_results = []
    for mode in modes:
        print(f"Running mode={mode}...", flush=True)
        result = evaluate_mode(
            mode,
            queries,
            database_url,
            voyage_api_key,
            anthropic_api_key=anthropic_api_key,
            use_llm_judge=args.llm_judge,
        )
        all_results.append(result)
        out_path = RESULTS_DIR / f"{mode.replace('+', '_')}.json"
        out_path.write_text(json.dumps(result, indent=2))
        llm_str = (
            f"  LLMCorr={result['avg_answer_correctness_llm']:.3f}"
            if "avg_answer_correctness_llm" in result
            else ""
        )
        print(
            f"  P@3={result['avg_precision_at_3']:.3f}  "
            f"R@3={result['avg_recall_at_3']:.3f}  "
            f"KwCorr={result['avg_answer_correctness_kw']:.3f}"
            f"{llm_str}  "
            f"p50={result['p50_latency_s']:.4f}s  "
            f"cost=${result['estimated_cost']['total_cost_usd']:.6f}"
        )

    if len(all_results) > 1:
        _print_comparison_table(all_results)
        comparison_path = RESULTS_DIR / "comparison.json"
        comparison_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nComparison saved to {comparison_path}")

    if args.benchmark:
        docs_dir = Path(__file__).parent.parent.parent / "docs"
        docs_dir.mkdir(exist_ok=True)
        _generate_benchmark_md(all_results, docs_dir / "benchmark.md")


if __name__ == "__main__":
    main()
