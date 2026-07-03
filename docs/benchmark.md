# Retrieval Benchmark

Generated: 2026-07-03 17:16 UTC  
Results from: Postgres (Supabase), 2026-07-03  
Queries: 62 total, 52 answerable  
Eval dataset: `backend/eval/queries.json`

## Mode × Metric

| Mode | P@3 (title) | R@3 (title) | P@3 (doc) | R@3 (doc) | H@1 | H@3 | H@5 | H@10 | NDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `keyword` | 0.301 | 0.904 | 0.301 | 0.904 | 0.673 | 0.904 | 0.923 | 0.942 | 0.819 | 0.787 |
| `fulltext` | 0.301 | 0.904 | 0.301 | 0.904 | 0.673 | 0.904 | 0.923 | 0.942 | 0.819 | 0.787 |
| `hybrid` | 0.301 | 0.904 | 0.301 | 0.904 | 0.673 | 0.904 | 0.923 | 0.942 | 0.819 | 0.787 |
| `hybrid+rerank` | 0.301 | 0.904 | 0.301 | 0.904 | 0.712 | 0.923 | 0.942 | 0.942 | 0.848 | 0.816 |

## Quality, Latency & Cost

| Mode | CtxRel | KwCorr | P50 (s) | P95 (s) | Cost ($) |
| --- | --- | --- | --- | --- | --- |
| `keyword` | 0.000 | 0.769 | 0.0001 | 0.0001 | 0.000062 |
| `fulltext` | 0.000 | 0.769 | 2.3364 | 3.7999 | 0.000062 |
| `hybrid` | 0.412 | 0.769 | 2.7144 | 5.5521 | 0.000062 |
| `hybrid+rerank` | 0.412 | 0.769 | 3.0301 | 6.1534 | 0.000961 |

## Pareto: Cost vs Quality and Latency vs Quality

### Cost ($) vs NDCG@5

<svg xmlns="http://www.w3.org/2000/svg" width="480" height="320" font-family="sans-serif" font-size="11">
<text x="240" y="18" text-anchor="middle" font-size="13" font-weight="bold">Cost vs Quality (NDCG@5)</text>
<line x1="60" y1="260" x2="420" y2="260" stroke="#888" stroke-width="1"/>
<line x1="60" y1="30" x2="60" y2="260" stroke="#888" stroke-width="1"/>
<text x="240" y="310" text-anchor="middle" fill="#555">Cost ($)</text>
<text x="12" y="160" text-anchor="middle" fill="#555" transform="rotate(-90,12,160)">NDCG@5</text>
<circle cx="90.0" cy="243.3" r="6" fill="#4f46e5" opacity="0.85"/>
<text x="99.0" y="247.3" fill="#4f46e5">keyword</text>
<circle cx="90.0" cy="243.3" r="6" fill="#0891b2" opacity="0.85"/>
<text x="99.0" y="247.3" fill="#0891b2">fulltext</text>
<circle cx="90.0" cy="243.3" r="6" fill="#059669" opacity="0.85"/>
<text x="99.0" y="247.3" fill="#059669">hybrid</text>
<circle cx="390.0" cy="76.7" r="6" fill="#d97706" opacity="0.85"/>
<text x="399.0" y="80.7" fill="#d97706">hybrid+rerank</text>
</svg>

### P95 Latency (s) vs NDCG@5

<svg xmlns="http://www.w3.org/2000/svg" width="480" height="320" font-family="sans-serif" font-size="11">
<text x="240" y="18" text-anchor="middle" font-size="13" font-weight="bold">Latency vs Quality (NDCG@5)</text>
<line x1="60" y1="260" x2="420" y2="260" stroke="#888" stroke-width="1"/>
<line x1="60" y1="30" x2="60" y2="260" stroke="#888" stroke-width="1"/>
<text x="240" y="310" text-anchor="middle" fill="#555">P95 Latency (s)</text>
<text x="12" y="160" text-anchor="middle" fill="#555" transform="rotate(-90,12,160)">NDCG@5</text>
<circle cx="90.0" cy="243.3" r="6" fill="#4f46e5" opacity="0.85"/>
<text x="99.0" y="247.3" fill="#4f46e5">keyword</text>
<circle cx="275.3" cy="243.3" r="6" fill="#0891b2" opacity="0.85"/>
<text x="284.3" y="247.3" fill="#0891b2">fulltext</text>
<circle cx="360.7" cy="243.3" r="6" fill="#059669" opacity="0.85"/>
<text x="369.7" y="247.3" fill="#059669">hybrid</text>
<circle cx="390.0" cy="76.7" r="6" fill="#d97706" opacity="0.85"/>
<text x="399.0" y="80.7" fill="#d97706">hybrid+rerank</text>
</svg>

## Column definitions

- **P@3 (title)** — Precision@3 by title string match *(deprecated — biased against chunked modes; kept for historical comparison)*
- **R@3 (title)** — Recall@3 by title string match *(deprecated — kept for historical comparison)*
- **P@3 (doc)** — Precision@3 by document ID: fraction of top-3 chunks from the correct document.
- **R@3 (doc)** — Recall@3 by document ID: whether any top-3 chunk belongs to the correct document (binary).
- **H@k** — Hit Rate@k: binary 1 if the correct document appears anywhere in the top-k results. Production retrieval only returns 3 results; H@5/H@10 and NDCG@5 are computed from an eval-only deeper retrieval (top 10) so the columns reflect real candidates, not a re-sliced top-3 list.
- **NDCG@5** — Normalised Discounted Cumulative Gain at 5 (single-relevant-doc; higher rank = higher score).
- **MRR** — Mean Reciprocal Rank: 1/rank of the first relevant chunk retrieved.
- **CtxRel** — average cosine similarity between query and retrieved chunk embeddings
- **KwCorr** — keyword-overlap answer correctness (fraction of expected keywords present)
- **LLMCorr** — Claude-as-judge factual accuracy score (0–1); `null` if not run with `--llm-judge`
- **P50/P95** — median and 95th-percentile end-to-end retrieval latency
- **Cost** — estimated Voyage + Claude API cost for the full eval run

## Regenerating

```bash
python -m backend.eval.run --all-modes --benchmark
# with LLM-judge (costs ~$0.001 per query):
python -m backend.eval.run --all-modes --llm-judge --benchmark
```

Raw per-mode JSON lives in `backend/eval/results/`.

## Trend (last 1 comparable run)

**NDCG@5**: <svg xmlns="http://www.w3.org/2000/svg" width="80" height="20"><circle cx="40" cy="10" r="2" fill="#4f46e5"/></svg>

**MRR**: <svg xmlns="http://www.w3.org/2000/svg" width="80" height="20"><circle cx="40" cy="10" r="2" fill="#4f46e5"/></svg>

*1 run recorded in `docs/benchmark-history.jsonl` (n_docs=15, metric_version=doc-id-v2); runs from a different KB size or metric version are excluded.*
