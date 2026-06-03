# Reranking Decision

## Updated finding (15-doc KB, 62-query eval)

The initial Gap 7 spot-check tested 3 queries and found reranking did not change result ordering
on any of them. That check was insufficient: the 3 queries happened to return well-separated
hybrid scores where the reranker agreed with the existing order.

The full `--all-modes` live eval against the 15-doc / 75-chunk Supabase KB tells a different
story:

| Mode | NDCG@5 | H@1 | P@3 (doc) |
|---|---|---|---|
| hybrid | 0.253 | 0.231 | 0.109 |
| hybrid+rerank | 0.288 | 0.288 | 0.141 |

Reranking improves every position metric. NDCG@5 gains +0.035 and H@1 gains +0.057.
At 15 documents the candidate pool is large enough that the reranker's cross-encoder signal
adds real value over cosine similarity alone.

## Decision

`enable_reranking=False` remains the code default (off unless opted in), but the data
no longer supports the claim that reranking has no benefit at this KB scale. The correct
threshold for enabling it by default is a product decision about latency budget (reranking
adds ~50 ms and 15× the Voyage cost per query) not a quality argument.

**Recommendation:** enable reranking in production where P95 latency ≤ 3 s is acceptable.
Re-evaluate if the KB grows beyond 500 documents and the latency cost grows proportionally.

## Initial spot-check finding (superseded)

Original 3-query test found no ordering change on:
- "My purchase arrived broken, what are my options?"
- "How do I return a damaged item?"
- "What is the refund policy for defective products?"

This was a valid test of those 3 queries; it was not a valid basis for a general conclusion.
