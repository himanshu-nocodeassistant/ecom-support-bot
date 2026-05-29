# Decision: Why keyword retrieval beat hybrid in the Phase 6 baseline

**Status:** Open — investigation pending (Phase 9a)
**Owner:** Himanshu

---

## The finding

The Phase 6 baseline committed at `backend/eval/results/comparison.json` shows in-memory keyword retrieval outperforming every Postgres-backed mode on the labeled query set:

| Mode                    | P@3   | R@3   | KwCorr |
|-------------------------|-------|-------|--------|
| `keyword` (in-memory)   | 0.290 | 0.870 | 0.687  |
| `fulltext`              | 0.044 | 0.130 | 0.084  |
| `hybrid`                | 0.116 | 0.174 | 0.145  |
| `hybrid+rerank`         | 0.116 | 0.174 | 0.145  |
| `hybrid+rerank+filter`  | 0.087 | 0.174 | 0.128  |

`backend/eval/thresholds.json` declares `hybrid+rerank+filter` as `best_mode`, which is the worst performer in the table. The CI regression gate is therefore guarding the wrong mode.

---

## Working hypotheses

### H1 — Metric bias against chunked retrieval

`_precision_at_k` in `backend/eval/run.py` checks string equality between retrieved chunk titles and `expected_source_title`. Two consequences:

- `InMemoryRepository.search_knowledge` returns up to 3 whole documents. A single correct hit produces P@3 = 1/3.
- `PostgresRepository.search_knowledge` returns up to 3 chunks. If all 3 chunks share the source title, P@3 = 1.0; if some come from other documents, the metric drops sharply — even though the correct document was retrieved.

**Test:** switch the metric from title-equality to `chunk.document_id == expected_doc_id` and re-run all modes.

### H2 — Knowledge-base-size effect

The KB has 4 documents producing ~6 chunks. With so few candidates:

- Full-text and embedding scoring have almost no separation to do
- The 70/30 hybrid weighting amplifies embedding noise without the doc count to offset it
- Voyage rerank on 3 candidates can only reorder them, not surface anything new
- The keyword-based `_infer_category` pre-filter (`repository.py:75-102`) can reduce candidates to 1–2, frequently the wrong one

**Test:** grow the KB to ≥ 15 docs, re-run all modes, see whether the ranking flips.

---

## Decision

*(To be filled after Phase 9a investigation.)*

- Which hypothesis was load-bearing
- Before/after numbers
- Whether `_infer_category` survived
- New `best_mode` written into `thresholds.json`
- One paragraph for the README explaining what changed and why

---

## Possible outcomes

If H1 (metric bias): deliverable is a chunk-aware metric and re-run with updated rankings.

If H2 (KB size): deliverable is a mode-performance-vs-KB-size plot showing where (or whether) the crossover happens.

If both: both deliverables apply.
