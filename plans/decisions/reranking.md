# Reranking Decision

## Updated finding (15-doc KB, 62-query eval — re-verified 2026-07-05)

The initial Gap 7 spot-check tested 3 queries and found reranking did not change result ordering
on any of them. That check was insufficient: the 3 queries happened to return well-separated
hybrid scores where the reranker agreed with the existing order.

The full `--all-modes` live eval against the 15-doc / 75-chunk Supabase KB (run after
`eval-audit-degraded-fallback.md` fixed the silent hybrid→fulltext degradation under Voyage
rate limits — see that doc for context):

| Mode | NDCG@5 | H@1 | P@3 (doc) |
|---|---|---|---|
| hybrid | 0.934 | 0.904 | 0.647 |
| hybrid+rerank | 0.960 | 0.942 | 0.647 |

Reranking improves every position metric. NDCG@5 gains +0.026 and H@1 gains +0.038.
At 15 documents the candidate pool is large enough that the reranker's cross-encoder signal
adds real value over cosine similarity alone.

## Decision

`enable_reranking=False` remains the code default (off unless opted in), but the data
no longer supports the claim that reranking has no benefit at this KB scale. The correct
threshold for enabling it by default is a product decision about latency budget (reranking
adds ~326ms and 15× the Voyage cost per query — P50 2.107s vs 2.433s, P95 3.811s vs 4.395s)
not a quality argument.

**Recommendation:** enable reranking in production where P95 latency ≤ 5 s is acceptable.
Re-evaluate if the KB grows beyond 500 documents and the latency cost grows proportionally.

## Initial spot-check finding (superseded)

Original 3-query test found no ordering change on:
- "My purchase arrived broken, what are my options?"
- "How do I return a damaged item?"
- "What is the refund policy for defective products?"

This was a valid test of those 3 queries; it was not a valid basis for a general conclusion.
