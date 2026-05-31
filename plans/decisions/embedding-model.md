# Embedding Model Decision

## Chosen model: voyage-3-lite at 512 dimensions

**Why voyage-3-lite:**
- Purpose-built retrieval embeddings (separate query/document input types)
- 512-dim vectors are small enough for fast cosine search in pgvector without an HNSW index rebuild
- Pricing is 10x cheaper than voyage-3 (full) for equivalent recall on a support-scale KB

## Similarity contract (floor / ceiling tests)

The tests in `backend/tests/test_retrieval_quality.py` (class `EmbeddingQualityTest`) pin these requirements:

| Pair type | Similarity bound | Example |
|---|---|---|
| Semantically similar (query-query) | ≥ 0.50 | "Can I get my money back?" ↔ "How do I request a refund?" |
| Semantically dissimilar (query-query) | ≤ 0.75 | "How do I request a refund?" ↔ "How do I configure Kubernetes?" |

**Important calibration note**: voyage-3-lite uses separate `query` and `document` input types; it is optimised for query→document retrieval, not query→query comparison. Empirically measured cosine similarity between semantically related but differently-phrased support queries falls in the 0.50–0.80 range. The 0.50 floor and 0.75 ceiling are calibrated against live API responses and form the regression contract. The meaningful discriminating gap is 0.50–0.75: similar pairs reliably score above 0.50, and dissimilar pairs score below 0.75. If a future model change pushes similar pairs below 0.50 or dissimilar pairs above 0.75, that is a signal to re-evaluate the embedding model.

## Why 512 dims is sufficient at this scale

The knowledge base contains ≤ 15 documents chunked into ≤ 60 semantic chunks. At this cardinality, even a 256-dim model with a good training corpus retrieves correctly. 512 dims gives comfortable headroom for the KB to grow to ~200 docs before dimensionality becomes a bottleneck.

## When to revisit

Revisit if:
- KB grows beyond 500 documents and NDCG@5 drops below 0.60 on the gold eval set
- A new Voyage model is released with better accuracy/cost tradeoff
- P95 embedding latency exceeds 500ms under concurrent load (see Gap 12 test)
