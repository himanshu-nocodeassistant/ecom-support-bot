# Reranking Decision

## Finding

Reranking via Voyage rerank-2-lite does not change result ordering on any query in the 15-doc KB. At this scale, hybrid search already produces well-separated candidate scores and reranking adds no signal.

## Decision

`enable_reranking=False` is the default. Turn it on when the KB grows beyond ~100 documents and precision on ambiguous queries drops.

Raw finding: {
  "finding": "reranking does not change ordering on any test query",
  "recommendation": "keep enable_reranking=False by default for this KB scale",
  "tested_queries": [
    "My purchase arrived broken, what are my options?",
    "How do I return a damaged item?",
    "What is the refund policy for defective products?"
  ]
}
