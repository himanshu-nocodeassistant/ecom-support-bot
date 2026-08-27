from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from .config import get_settings
from .data import KNOWLEDGE_BASE, ORDERS


class Repository(Protocol):
    def get_order(self, order_id: str) -> dict[str, Any] | None: ...

    def search_knowledge(self, query: str, k: int | None = None) -> list[dict[str, Any]]: ...


@dataclass
class InMemoryRepository:
    orders: dict[str, dict[str, Any]]
    knowledge_documents: list[dict[str, Any]]

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        return self.orders.get(order_id)

    def search_knowledge(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        import re

        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "can",
            "do",
            "does",
            "for",
            "have",
            "how",
            "i",
            "is",
            "it",
            "my",
            "of",
            "or",
            "the",
            "to",
            "what",
            "when",
            "where",
            "why",
            "you",
        }
        query_words = {
            token for token in re.findall(r"[a-z0-9]+", query.lower()) if token not in stop_words
        }
        ranked = []
        for doc in self.knowledge_documents:
            haystack_tokens = set(
                re.findall(r"[a-z0-9]+", f"{doc['title']} {doc['content']}".lower())
            )
            score = sum(1 for word in query_words if word in haystack_tokens)
            if score:
                ranked.append(
                    {
                        "id": doc["id"],
                        "title": doc["title"],
                        "category": doc["category"],
                        "score": score,
                        "content": doc["content"],
                    }
                )
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[: k or 3]


def rewrite_query(query: str, api_key: str) -> str:
    """Rewrite a colloquial customer query into clean retrieval terms via Claude Haiku.

    Turns "my blender is making a weird noise" → "blender malfunction noise troubleshooting".
    Falls back to the original query on any error. Costs ~$0.0001 per call.
    """
    try:
        import anthropic

        prompt = (
            "Rewrite the following customer support question into concise retrieval terms "
            "(3-6 words) suitable for a product knowledge base search. "
            "Return ONLY the rewritten query, nothing else.\n\n"
            f"Question: {query}"
        )
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=32,
            messages=[{"role": "user", "content": prompt}],
        )
        rewritten = response.content[0].text.strip()
        return rewritten if rewritten else query
    except Exception:
        return query


class PostgresRepository:
    def __init__(
        self,
        database_url: str,
        fallback: InMemoryRepository,
        voyage_api_key: str | None = None,
        enable_reranking: bool = False,
        enable_query_rewriting: bool = False,
        anthropic_api_key: str | None = None,
        candidate_depth: int = 20,
        final_depth: int = 3,
        retrieval_mode: str = "weighted",
        rrf_k: int = 60,
    ) -> None:
        self.database_url = database_url
        self.fallback = fallback
        self.voyage_api_key = voyage_api_key
        self.enable_reranking = enable_reranking
        self.enable_query_rewriting = enable_query_rewriting
        self.anthropic_api_key = anthropic_api_key
        if candidate_depth < 1 or final_depth < 1:
            raise ValueError("candidate_depth and final_depth must be positive")
        self.candidate_depth = candidate_depth
        self.final_depth = final_depth
        if retrieval_mode not in {"weighted", "rrf"}:
            raise ValueError("retrieval_mode must be 'weighted' or 'rrf'")
        self.retrieval_mode = retrieval_mode
        self.rrf_k = max(1, rrf_k)
        self.tracer = None

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        try:
            import psycopg
        except ImportError:
            return self.fallback.get_order(order_id)

        query = """
            select order_id, customer_name, status, shipping_date, delivery_estimate, item, delivered
            from support_orders
            where order_id = %s
        """
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (order_id,))
                    row = cur.fetchone()
        except psycopg.Error:
            return self.fallback.get_order(order_id)
        if not row:
            return self.fallback.get_order(order_id)
        return {
            "order_id": row[0],
            "customer_name": row[1],
            "status": row[2],
            "shipping_date": str(row[3]) if row[3] else None,
            "delivery_estimate": str(row[4]) if row[4] else None,
            "item": row[5],
            "delivered": bool(row[6]),
        }

    def search_knowledge(self, query_text: str, k: int | None = None) -> list[dict[str, Any]]:
        requested_final_depth = self.final_depth if k is None else max(1, k)
        candidate_depth = max(self.candidate_depth, requested_final_depth)
        try:
            import psycopg  # noqa: F401
        except ImportError:
            return self.fallback.search_knowledge(query_text, k=requested_final_depth)

        retrieval_query = query_text
        if self.enable_query_rewriting and self.anthropic_api_key:
            retrieval_query = rewrite_query(query_text, api_key=self.anthropic_api_key)

        if self.voyage_api_key and self.retrieval_mode == "rrf":
            results = self._rrf_search(retrieval_query, k=candidate_depth)
        elif self.voyage_api_key:
            results = self._hybrid_search(retrieval_query, k=candidate_depth)
        else:
            results = self._fulltext_search(retrieval_query, k=candidate_depth)

        if self.enable_reranking and self.voyage_api_key and results:
            results = self._rerank(query_text, results, k=requested_final_depth)

        return results[:requested_final_depth]

    def _sparse_search(self, query_text: str, k: int = 20) -> list[dict[str, Any]]:
        """Run PostgreSQL full-text search as an independent ranked list."""
        import psycopg

        started = time.monotonic()

        sql = """
            select kd.id, kd.title, kd.category, kc.chunk_text,
                   ts_rank(kc.search_vector, plainto_tsquery('english', %(q)s)) as score
            from knowledge_chunks kc join knowledge_documents kd on kd.id = kc.document_id
            where kc.search_vector @@ plainto_tsquery('english', %(q)s)
            order by score desc, kd.id asc limit %(limit)s
        """
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, {"q": query_text, "limit": k})
                    rows = cur.fetchall()
        except psycopg.Error:
            if self.tracer:
                self.tracer.observe(
                    "sparse_retrieval",
                    {
                        "candidate_depth": k,
                        "document_ids": [],
                        "stage_ranks": [],
                        "degraded": True,
                        "latency_ms": round((time.monotonic() - started) * 1000, 2),
                    },
                )
            return []
        results = [
            {"id": r[0], "title": r[1], "category": r[2], "content": r[3], "score": float(r[4])}
            for r in rows
        ]
        if self.tracer:
            self.tracer.observe(
                "sparse_retrieval",
                {
                    "candidate_depth": k,
                    "document_ids": [r["id"] for r in results],
                    "stage_ranks": list(range(1, len(results) + 1)),
                    "degraded": not results,
                    "latency_ms": round((time.monotonic() - started) * 1000, 2),
                },
            )
        return results

    def _dense_search(self, query_text: str, k: int = 20) -> list[dict[str, Any]]:
        """Run pgvector search as an independent ranked list."""
        import psycopg

        started = time.monotonic()

        from .data_loader import embed_query

        try:
            embedding = embed_query(query_text, api_key=self.voyage_api_key)
            emb = "[" + ",".join(str(v) for v in embedding) + "]"
            sql = """
                select kd.id, kd.title, kd.category, kc.chunk_text,
                       (1 - (kc.embedding <=> %(emb)s::vector)) as score
                from knowledge_chunks kc join knowledge_documents kd on kd.id = kc.document_id
                where kc.embedding is not null
                order by score desc, kd.id asc limit %(limit)s
            """
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, {"emb": emb, "limit": k})
                    rows = cur.fetchall()
        except Exception:
            if self.tracer:
                self.tracer.observe(
                    "dense_retrieval",
                    {
                        "candidate_depth": k,
                        "document_ids": [],
                        "stage_ranks": [],
                        "degraded": True,
                        "latency_ms": round((time.monotonic() - started) * 1000, 2),
                    },
                )
            return []
        results = [
            {"id": r[0], "title": r[1], "category": r[2], "content": r[3], "score": float(r[4])}
            for r in rows
        ]
        if self.tracer:
            self.tracer.observe(
                "dense_retrieval",
                {
                    "candidate_depth": k,
                    "document_ids": [r["id"] for r in results],
                    "stage_ranks": list(range(1, len(results) + 1)),
                    "degraded": not results,
                    "latency_ms": round((time.monotonic() - started) * 1000, 2),
                },
            )
        return results

    def _rrf_search(self, query_text: str, k: int = 20) -> list[dict[str, Any]]:
        sparse = self._sparse_search(query_text, k=k)
        dense = self._dense_search(query_text, k=k)
        fused = self._rrf_fuse(sparse, dense, k=self.rrf_k)
        if self.tracer:
            self.tracer.observe(
                "fusion",
                {
                    "candidate_depth": k,
                    "document_ids": [r["id"] for r in fused],
                    "stage_ranks": [
                        {
                            "id": r["id"],
                            "sparse": r.get("sparse_rank"),
                            "dense": r.get("dense_rank"),
                            "fused": r.get("fused_rank"),
                        }
                        for r in fused
                    ],
                    "degraded": not sparse or not dense,
                },
            )
        if not fused:
            fallback = self.fallback.search_knowledge(query_text, k=k)
            for result in fallback:
                result["degraded"] = "rrf_no_results"
            return fallback
        if not sparse or not dense:
            for result in fused:
                result["degraded"] = (
                    "rrf_sparse_unavailable" if not sparse else "rrf_dense_unavailable"
                )
        return fused

    @staticmethod
    def _rrf_fuse(
        sparse: list[dict[str, Any]], dense: list[dict[str, Any]], k: int = 60
    ) -> list[dict[str, Any]]:
        """Fuse lists by document id. Ties are stable and resolved by id."""
        entries: dict[str, dict[str, Any]] = {}
        for stage, results in (("sparse", sparse), ("dense", dense)):
            for rank, result in enumerate(results, start=1):
                doc_id = str(result["id"])
                item = entries.setdefault(doc_id, dict(result))
                item[f"{stage}_rank"] = rank
                item[f"{stage}_score"] = float(result.get("score", 0.0))
                item["fused_score"] = item.get("fused_score", 0.0) + 1.0 / (k + rank)
        ordered = sorted(entries.values(), key=lambda item: (-item["fused_score"], str(item["id"])))
        for rank, item in enumerate(ordered, start=1):
            item.setdefault("sparse_rank", None)
            item.setdefault("sparse_score", None)
            item.setdefault("dense_rank", None)
            item.setdefault("dense_score", None)
            item["fused_rank"] = rank
        return ordered

    def _fulltext_search(self, query_text: str, k: int = 3) -> list[dict[str, Any]]:
        import psycopg

        sql = """
            select
                kd.id,
                kd.title,
                kd.category,
                kc.chunk_text,
                ts_rank(kc.search_vector, plainto_tsquery('english', %(q)s)) as score
            from knowledge_chunks kc
            join knowledge_documents kd on kd.id = kc.document_id
            where kc.search_vector @@ plainto_tsquery('english', %(q)s)
            order by score desc, kd.id asc
            limit %(limit)s
        """
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, {"q": query_text, "limit": max(k * 2, 6)})
                    rows = cur.fetchall()
        except psycopg.Error:
            return self.fallback.search_knowledge(query_text, k=k)
        if not rows:
            return []
        return [
            {
                "id": row[0],
                "title": row[1],
                "category": row[2],
                "content": row[3],
                "score": float(row[4]),
            }
            for row in rows
        ][:k]

    def _hybrid_search(self, query_text: str, k: int = 3) -> list[dict[str, Any]]:
        import logging

        import psycopg

        from .data_loader import embed_query

        try:
            query_embedding = embed_query(query_text, api_key=self.voyage_api_key)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "hybrid search degraded to fulltext: embed_query failed (%s: %s)",
                type(exc).__name__,
                exc,
            )
            results = self._fulltext_search(query_text, k=k)
            for r in results:
                r["degraded"] = "embed_query_failed"
            return results

        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        # Hybrid: 30% full-text rank + 70% cosine similarity
        sql = """
            select
                kd.id,
                kd.title,
                kd.category,
                kc.chunk_text,
                (
                    0.3 * ts_rank(kc.search_vector, plainto_tsquery('english', %(q)s))
                    + 0.7 * (1 - (kc.embedding <=> %(emb)s::vector))
                ) as score
            from knowledge_chunks kc
            join knowledge_documents kd on kd.id = kc.document_id
            where kc.embedding is not null
            order by score desc
            limit %(limit)s
        """
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql, {"q": query_text, "emb": embedding_str, "limit": max(k * 2, 6)}
                    )
                    rows = cur.fetchall()
        except psycopg.Error as exc:
            logging.getLogger(__name__).warning(
                "hybrid search degraded to fulltext: query failed (%s)", exc
            )
            results = self._fulltext_search(query_text, k=k)
            for r in results:
                r["degraded"] = "hybrid_query_failed"
            return results
        if not rows:
            results = self._fulltext_search(query_text, k=k)
            for r in results:
                r["degraded"] = "hybrid_no_rows"
            return results
        return [
            {
                "id": row[0],
                "title": row[1],
                "category": row[2],
                "content": row[3],
                "score": float(row[4]),
            }
            for row in rows
        ][:k]

    def _rerank(
        self, query_text: str, results: list[dict[str, Any]], k: int = 3
    ) -> list[dict[str, Any]]:
        """Post-retrieval reranking via Voyage AI rerank API."""
        try:
            import voyageai

            client = voyageai.Client(api_key=self.voyage_api_key)
            docs = [r["content"] for r in results]
            reranked = client.rerank(query_text, docs, model="rerank-2-lite", top_k=k)
            reranked_results = []
            for item in reranked.results:
                r = dict(results[item.index])
                r["score"] = float(item.relevance_score)
                r["reranked"] = True
                reranked_results.append(r)
            if self.tracer:
                self.tracer.observe(
                    "reranking",
                    {
                        "candidate_depth": len(results),
                        "final_depth": k,
                        "document_ids": [r["id"] for r in reranked_results],
                        "degraded": False,
                    },
                )
            return reranked_results
        except Exception:
            # A reranker outage must remain observable to callers and metrics.
            degraded_results = []
            for result in results[:k]:
                degraded = dict(result)
                degraded["degraded"] = "rerank_failed"
                degraded_results.append(degraded)
            if self.tracer:
                self.tracer.observe(
                    "reranking",
                    {
                        "candidate_depth": len(results),
                        "final_depth": k,
                        "document_ids": [r["id"] for r in degraded_results],
                        "degraded": True,
                    },
                )
            return degraded_results


@lru_cache(maxsize=1)
def get_repository() -> Repository:
    settings = get_settings()
    fallback = InMemoryRepository(orders=ORDERS, knowledge_documents=KNOWLEDGE_BASE)

    if settings.data_backend == "postgres" and settings.database_url:
        return PostgresRepository(
            database_url=settings.database_url,
            fallback=fallback,
            voyage_api_key=settings.voyage_api_key,
            enable_reranking=settings.enable_reranking,
            candidate_depth=settings.retrieval_candidate_depth,
            final_depth=settings.retrieval_final_depth,
            retrieval_mode=getattr(settings, "retrieval_mode", "weighted"),
        )

    return fallback
