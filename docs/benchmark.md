# Retrieval Benchmark

Generated: 2026-05-31 10:28 UTC  
Results from: Postgres (Supabase), 2026-05-31  
Queries: 62 total, 52 answerable  
Eval dataset: `backend/eval/queries.json`

## Mode × Metric

| Mode | P@3 (title) | R@3 (title) | P@3 (doc) | R@3 (doc) | H@1 | H@3 | H@5 | H@10 | NDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `keyword` | 0.301 | 0.904 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `fulltext` | 0.083 | 0.250 | 0.083 | 0.250 | 0.211 | 0.250 | 0.250 | 0.250 | 0.233 | 0.228 |
| `hybrid` | 0.109 | 0.269 | 0.109 | 0.269 | 0.231 | 0.269 | 0.269 | 0.269 | 0.253 | 0.247 |
| `hybrid+rerank` | 0.141 | 0.288 | 0.141 | 0.288 | 0.288 | 0.288 | 0.288 | 0.288 | 0.288 | 0.288 |

## Quality, Latency & Cost

| Mode | CtxRel | KwCorr | P50 (s) | P95 (s) | Cost ($) |
| --- | --- | --- | --- | --- | --- |
| `keyword` | 0.000 | 0.769 | 0.0001 | 0.0001 | 0.000062 |
| `fulltext` | 0.000 | 0.186 | 1.7603 | 2.0703 | 0.000062 |
| `hybrid` | 0.180 | 0.238 | 2.0294 | 2.5008 | 0.000062 |
| `hybrid+rerank` | 0.191 | 0.248 | 2.0839 | 2.5970 | 0.000961 |

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
<circle cx="90.0" cy="108.6" r="6" fill="#0891b2" opacity="0.85"/>
<text x="99.0" y="112.6" fill="#0891b2">fulltext</text>
<circle cx="90.0" cy="97.5" r="6" fill="#059669" opacity="0.85"/>
<text x="99.0" y="101.5" fill="#059669">hybrid</text>
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
<circle cx="329.2" cy="108.6" r="6" fill="#0891b2" opacity="0.85"/>
<text x="338.2" y="112.6" fill="#0891b2">fulltext</text>
<circle cx="378.9" cy="97.5" r="6" fill="#059669" opacity="0.85"/>
<text x="387.9" y="101.5" fill="#059669">hybrid</text>
<circle cx="390.0" cy="76.7" r="6" fill="#d97706" opacity="0.85"/>
<text x="399.0" y="80.7" fill="#d97706">hybrid+rerank</text>
</svg>

## Column definitions

- **P@3 (title)** — Precision@3 by title string match *(deprecated — biased against chunked modes; kept for historical comparison)*
- **R@3 (title)** — Recall@3 by title string match *(deprecated — kept for historical comparison)*
- **P@3 (doc)** — Precision@3 by document ID: fraction of top-3 chunks from the correct document.
- **R@3 (doc)** — Recall@3 by document ID: whether any top-3 chunk belongs to the correct document (binary).
- **H@k** — Hit Rate@k: binary 1 if the correct document appears anywhere in the top-k results.
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

## Trend (last 20 CI runs)

**NDCG@5**: <svg xmlns="http://www.w3.org/2000/svg" width="80" height="20"><polyline points="2.0,2.0 40.0,2.0 78.0,18.0" fill="none" stroke="#4f46e5" stroke-width="1.5"/></svg>

**MRR**: <svg xmlns="http://www.w3.org/2000/svg" width="80" height="20"><polyline points="2.0,2.0 40.0,2.0 78.0,18.0" fill="none" stroke="#4f46e5" stroke-width="1.5"/></svg>

*3 runs recorded in `docs/benchmark-history.jsonl`.*
