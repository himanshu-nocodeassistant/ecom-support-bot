# Retrieval Benchmark

Generated: 2026-07-05 15:32 UTC  
Results from: Postgres (Supabase), 2026-07-05  
Queries: 62 total, 52 answerable  
Eval dataset: `backend/eval/queries.json`

## Mode × Metric

| Mode | P@3 (title) | R@3 (title) | P@3 (doc) | R@3 (doc) | H@1 | H@3 | H@5 | H@10 | NDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `keyword` | 0.301 | 0.904 | 0.301 | 0.904 | 0.673 | 0.904 | 0.923 | 0.942 | 0.819 | 0.787 |
| `fulltext` | 0.083 | 0.250 | 0.083 | 0.250 | 0.211 | 0.250 | 0.250 | 0.250 | 0.233 | 0.228 |
| `hybrid` | 0.647 | 0.942 | 0.647 | 0.942 | 0.904 | 0.942 | 0.962 | 0.981 | 0.934 | 0.927 |
| `hybrid+rerank` | 0.647 | 0.942 | 0.647 | 0.942 | 0.942 | 0.962 | 0.981 | 0.981 | 0.960 | 0.954 |

## Quality, Latency & Cost

| Mode | CtxRel | KwCorr | KwCorr chars | P50 (s) | P95 (s) | Cost ($) |
| --- | --- | --- | --- | --- | --- | --- |
| `keyword` | — | 0.769 | 519 | 0.0001 | 0.0001 | $0 |
| `fulltext` | — | 0.186 | 46 | 1.7978 | 2.9921 | $0 |
| `hybrid` | 0.554 | 0.807 | 345 | 2.1068 | 3.8115 | 0.000062 |
| `hybrid+rerank` | 0.554 | 0.807 | 345 | 2.4330 | 4.3947 | 0.000961 |

## Pareto: Cost vs Quality and Latency vs Quality

### Cost ($) vs NDCG@5

<svg xmlns="http://www.w3.org/2000/svg" width="480" height="320" font-family="sans-serif" font-size="11">
<text x="240" y="18" text-anchor="middle" font-size="13" font-weight="bold">Cost vs Quality (NDCG@5)</text>
<line x1="60" y1="260" x2="420" y2="260" stroke="#888" stroke-width="1"/>
<line x1="60" y1="30" x2="60" y2="260" stroke="#888" stroke-width="1"/>
<text x="240" y="310" text-anchor="middle" fill="#555">Cost ($)</text>
<text x="12" y="160" text-anchor="middle" fill="#555" transform="rotate(-90,12,160)">NDCG@5</text>
<circle cx="90.0" cy="108.9" r="6" fill="#4f46e5" opacity="0.85"/>
<text x="99.0" y="112.9" fill="#4f46e5">keyword</text>
<circle cx="90.0" cy="243.3" r="6" fill="#0891b2" opacity="0.85"/>
<text x="99.0" y="247.3" fill="#0891b2">fulltext</text>
<circle cx="90.0" cy="82.7" r="6" fill="#059669" opacity="0.85"/>
<text x="99.0" y="86.7" fill="#059669">hybrid</text>
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
<circle cx="90.0" cy="108.9" r="6" fill="#4f46e5" opacity="0.85"/>
<text x="99.0" y="112.9" fill="#4f46e5">keyword</text>
<circle cx="294.3" cy="243.3" r="6" fill="#0891b2" opacity="0.85"/>
<text x="303.3" y="247.3" fill="#0891b2">fulltext</text>
<circle cx="350.2" cy="82.7" r="6" fill="#059669" opacity="0.85"/>
<text x="359.2" y="86.7" fill="#059669">hybrid</text>
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
- **CtxRel** — average cosine similarity between query and retrieved chunk embeddings. `—` for `keyword`/`fulltext` modes (no embeddings computed; not measured, not zero).
- **KwCorr** — keyword-overlap answer correctness: fraction of expected keywords present in the concatenated top-3 retrieved *content*. **Not valid for cross-mode comparison** — modes that return whole documents (e.g. `keyword`) search far more text per query than modes that return short chunks, so a higher score there reflects text volume, not better retrieval. Only compare KwCorr across runs of the *same* mode over time. See `KwCorr chars`.
- **KwCorr chars** — average character count of the text KwCorr was computed over, per mode. Included so a KwCorr gap is legible as a text-volume artifact rather than a quality difference.
- **CtxRelLLM** — Claude-as-judge relevance score (0–1) for the top retrieved chunk vs. the query; `—` if not run with `--llm-judge`. Judges retrieval, not a generated answer.
- **P50/P95** — median and 95th-percentile end-to-end retrieval latency. **Not comparable across backends**: `keyword` uses an in-memory dict (~0.0001s), while `fulltext`/`hybrid` measure a Supabase network round-trip (~1–2s). The difference is infrastructure, not algorithm quality.
- **Cost** — estimated Voyage + Claude API cost for the full eval run. Formula-derived, not metered. `$0` for modes that issue no API calls (`keyword`, `fulltext`).

## Regenerating

```bash
python -m backend.eval.run --all-modes --benchmark
# with LLM-judge (costs ~$0.001 per query):
python -m backend.eval.run --all-modes --llm-judge --benchmark
```

Raw per-mode JSON lives in `backend/eval/results/`.

## Trend (last 2 comparable runs)

**NDCG@5**: <svg xmlns="http://www.w3.org/2000/svg" width="80" height="20"><polyline points="2.0,18.0 78.0,2.0" fill="none" stroke="#4f46e5" stroke-width="1.5"/></svg>

**MRR**: <svg xmlns="http://www.w3.org/2000/svg" width="80" height="20"><polyline points="2.0,18.0 78.0,2.0" fill="none" stroke="#4f46e5" stroke-width="1.5"/></svg>

*2 runs recorded in `docs/benchmark-history.jsonl` (n_docs=15, metric_version=doc-id-v2); runs from a different KB size or metric version are excluded.*
