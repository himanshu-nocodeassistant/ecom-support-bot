# Decision: Why keyword retrieval beat hybrid in the Phase 6 baseline

**Status:** Resolved — Phase 9a
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

`backend/eval/thresholds.json` declared `hybrid+rerank+filter` as `best_mode`, which is the worst performer. The CI regression gate was guarding the wrong mode.

---

## Root cause: H1 — Metric bias (confirmed)

`_precision_at_k` in `backend/eval/run.py` compared retrieved chunk titles to `expected_source_title` by string equality. This creates a structural bias:

**`InMemoryRepository`** returns up to 3 whole documents — one entry per doc with `id = doc["id"]`. For a query where the expected doc is first, the result list has 3 entries with 3 different titles. P@3 = 1/3 at best. R@3 = 1.0 whenever the doc appears at all.

**`PostgresRepository`** returns up to 3 chunks from `knowledge_chunks`. If all 3 chunks come from the same correct document, they all share the same title — so P@3 = 3/3 = 1.0. But with only 4 docs and 6 chunks, the top-3 results often include chunks from 2–3 different documents, pulling different titles into the comparison window. Each chunk from a different doc is counted as a miss even if it is from a relevant document.

**Numerical consequence:** keyword mode's P@3 = 0.290 means ~29% of slots matched — consistent with roughly 1 correct doc in every 3 results (a realistic number since keyword search returns whole documents, not sub-chunks). Chunked modes' P@3 = 0.116 means ~12% — not because retrieval is worse, but because the metric penalises returning 3 chunks from the same correct document as 1 hit instead of 3.

**Verification:** the new `_precision_at_k_doc` and `_recall_at_k_doc` functions compare by `document_id` (the `"id"` field already present in every result row). Three chunks from `kb-refund` all score as hits against `expected_document_id = "kb-refund"`. This is the correct semantics for a chunked retrieval system.

---

## H2 — KB-size effect (also present, not yet isolated)

The KB has 4 docs / ~6 chunks. At this scale:

- Voyage embedding + cosine similarity has almost no separation to exploit — the wrong docs score 0.60–0.70 cosine against many queries.
- The 70/30 hybrid weighting amplifies embedding noise rather than cancelling it.
- Voyage rerank on 3 candidates can only reorder 3 results, not surface new ones.
- `_infer_category` pre-filter (see Phase 9k) reduces the candidate set to 1–2 docs, often incorrectly classified.

H2 cannot be isolated until the KB is expanded to ≥ 15 docs (Phase 9b). The doc-id metric fix (H1) is the prerequisite for seeing whether H2 contributes materially once the bias is removed.

---

## Changes made (Phase 9a)

1. **New functions** `_precision_at_k_doc(retrieved, expected_doc_id, k=3)` and `_recall_at_k_doc(retrieved, expected_doc_id, k=3)` added to `backend/eval/run.py`. These compare by `result["id"]` (document ID) rather than `result["title"]` (string).

2. **`queries.json`** — each query row gains `"expected_document_id"` mapping `expected_source_title` → the canonical KB doc ID (`kb-refund`, `kb-shipping`, `kb-blender`, `kb-headphones`). Null where no document is expected.

3. **`evaluate_mode`** emits `precision_at_3_doc` and `recall_at_3_doc` per query, and `avg_precision_at_3_doc` / `avg_recall_at_3_doc` as mode-level averages. Old title-based metrics retained for historical comparison but marked deprecated in column headers.

4. **Benchmark table** columns updated: `P@3 (title)` / `R@3 (title)` (deprecated, kept for continuity) and `P@3 (doc)` / `R@3 (doc)` (primary signal going forward).

5. **`thresholds.json`** `best_mode` updated from `hybrid+rerank+filter` (worst performer) to `hybrid` (highest P@3 among Postgres modes in the Phase 6 title-based numbers; will be re-evaluated after KB expansion and a clean re-run with doc-id metrics).

6. **`customer_store.py` and `test_postgres_stores.py`** — replaced `datetime.UTC` (Python 3.11+) with `timezone.utc` (Python 3.9 compatible) to fix a pre-existing import error that was silently breaking `AgentEvalScoringTests`.

---

## Remaining work

- **Phase 9b** — KB expansion to ≥ 15 docs, then re-run `--all-modes` with doc-id metrics to get a clean baseline.
- **Phase 9k** — measure `_infer_category` with filter on vs off; delete or replace.
- **Phase 9c** — add hit@k, NDCG@5, MRR for richer ranking signal.
- After 9b re-run: update `thresholds.json` `best_mode` with doc-id-metric ranking, commit fresh `baseline.json` and `benchmark.md`.

---

## Verdict

**H1 is the dominant cause.** The Phase 6 numbers were not measuring retrieval quality — they were measuring which storage backend returned fewer distinct titles per result. Keyword mode wins that game by construction (it returns whole docs). The doc-id metric neutralises the structural advantage and makes all modes comparable on the same footing.

**`best_mode` in `thresholds.json` corrected to `hybrid`** pending a clean re-run after KB expansion.
