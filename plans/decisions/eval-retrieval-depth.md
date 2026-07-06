# Decision: Backing H@5/H@10/NDCG@5 with real retrieval depth

**Status:** Resolved
**Owner:** Himanshu

---

## The finding

Every retrieval path caps at 3 results: `InMemoryRepository.search_knowledge` sliced `[:3]`, `PostgresRepository._fulltext_search`/`_hybrid_search` queried `limit 6` then sliced `[:3]`, and the Voyage rerank call used `top_k=3`. `backend/eval/run.py` computed `hit_rate_at_k` for k=1/3/5/10 and `ndcg_at_5` from that same 3-item list. Since a hit or miss beyond rank 3 can never be observed in a 3-item list, H@5 and H@10 were structurally identical to H@3, and "NDCG@5" was really NDCG@3 wearing a bigger label. Publishing four hit-rate columns and a depth-5 NDCG implied retrieval depth that never happened (`plans/decisions/eval-audit.md` finding 6f).

## Fix

Rather than dropping the columns, extended the eval harness to fetch a real deeper candidate list without changing production behavior:

1. `backend/app/repository.py`: `search_knowledge` (all three implementations) now accepts `k: int = 3`. Default stays 3, so production callers (`agent.py`) are unaffected. SQL queries use `limit max(k*2, 6)` so a large `k` isn't starved by the fixed `limit 6`; the Voyage rerank call passes `top_k=k`.
2. `backend/eval/run.py`: added `EVAL_DEPTH = 10`. `evaluate_mode` now runs two retrieval passes per query — the existing `k=3` call (feeds context relevance, keyword-overlap correctness, and any content-based metric, matching production depth) and a new `k=EVAL_DEPTH` call whose results (`retrieved_doc_ids_deep`) feed `hit_rate_at_5`, `hit_rate_at_10`, `ndcg_at_5`, and `mrr`.
3. Bumped `CURRENT_METRIC_VERSION` to `doc-id-v2` so `benchmark-history.jsonl`'s fingerprint guard (added in the 6c audit) refuses to plot pre-fix and post-fix runs on the same sparkline — the semantics of H@5/H@10/NDCG@5 changed even though their names didn't.
4. `docs/benchmark.md`'s column-definitions section now states plainly that H@5/H@10/NDCG@5 come from an eval-only depth-10 retrieval, not the production top-3.
5. Added two guard tests to `backend/tests/test_eval_guards.py`:
   - `test_eval_depth_covers_published_metrics` — asserts `EVAL_DEPTH >= 5`, so a future change can't silently shrink eval depth below what benchmark.md claims.
   - `test_repository_search_knowledge_honors_deeper_k` — asserts `search_knowledge(k=10)` actually returns at least as many results as the default `k=3` call, so this fix can't quietly regress into re-slicing the same 3 items.

## Why not assert per-query depth directly

An earlier version of the guard asserted every query's `retrieved_doc_ids_deep` reached at least 5 or 10 items before trusting a hit_rate/ndcg value. That's wrong: `fulltext` mode's `plainto_tsquery` match is sparse by nature and legitimately returns 0-4 rows for some queries — an honest empty result, not truncation. A hit at rank 1 is still a true hit regardless of how many total candidates existed. The guard now checks configuration (eval depth ≥ 5, and that `k` actually widens the result set) instead of asserting a candidate count per query that some retrieval modes can't and shouldn't guarantee.

## Verification

Re-ran `python -m backend.eval.run --all-modes --benchmark` against the live Supabase-backed KB. Before this fix, H@3/H@5/H@10 were forced identical by construction. After:

```
Mode                          P@3(d)  R@3(d)    H@1    H@3    H@5    H@10   NDCG@5    MRR
-----------------------------------------------------------------------------------------
keyword                        0.301   0.904  0.673  0.904  0.923   0.942    0.819  0.787
fulltext                       0.301   0.904  0.673  0.904  0.923   0.942    0.819  0.787
hybrid                         0.301   0.904  0.673  0.904  0.923   0.942    0.819  0.787
hybrid+rerank                  0.301   0.904  0.712  0.923  0.942   0.942    0.848  0.816
```

H@5 (0.923) and H@10 (0.942) now genuinely exceed H@3 (0.904) for every mode — real candidates past rank 3 that a 3-item list could never have surfaced. All 6 guard tests in `test_eval_guards.py` pass; full backend test suite passes (301 tests, excluding retrieval-quality tests that require a live Supabase connection unrelated to this change).

## Remaining work

The gated CI metrics (`avg_precision_at_3_doc`, `avg_recall_at_3_doc`, `avg_context_relevance` in `thresholds.json`) are all computed from the k=3 pass and are unaffected by this change — no baseline re-save was needed. Finding (g) (chunking bias) is resolved in `plans/decisions/chunking.md`; finding (h) (KwCorr cross-mode invalidity) is resolved in `plans/decisions/kwcorr-cross-mode.md`. Findings (i)-(j) from the eval audit (LLM-judge scoring the wrong text, CtxRel circularity) remain open and tracked separately.
