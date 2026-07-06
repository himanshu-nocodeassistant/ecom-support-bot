# Decision: KwCorr is not valid for cross-mode comparison

**Status:** Resolved (documented + instrumented; metric kept as same-mode-over-time signal only)
**Owner:** Himanshu

---

## The finding

`avg_answer_correctness_kw` ("KwCorr") measures the fraction of expected keywords present in the concatenated top-3 retrieved *content*. Retrieval modes don't all return the same unit of text: `keyword`/`fulltext` modes (in-memory and Postgres full-document search) return whole documents, while chunked modes (`hybrid`, `hybrid+rerank`) return ~220-character chunks. Whole-doc modes therefore search 5-10x more text per query for the same set of expected keywords, so a KwCorr gap between modes mostly reflects text volume, not retrieval quality (`plans/decisions/eval-audit.md` finding 6h in the 2026-07-03 repurposing handoff). It's also mislabeled "answer correctness" — no generated answer is involved, only retrieved content.

## Fix

1. `backend/eval/run.py`: `evaluate_mode` now records `kw_context_chars` per query (length of the text KwCorr was computed over) and `avg_kw_context_chars` per mode.
2. `docs/benchmark.md`'s Quality/Latency/Cost table now publishes a `KwCorr chars` column next to `KwCorr`, so a score gap is legible as a text-volume artifact instead of an implied quality difference.
3. Column definitions in both `docs/benchmark.md` and `docs/eval.md` state plainly that KwCorr is invalid across modes and should only be tracked for the same mode over time (e.g. regression checks).
4. `plans/decisions/chunking.md`'s re-run and the §8 claims ledger in the repurposing handoff already carried the "never claim cross-mode" rule for KwCorr; this doc is the tracked decision record for it, and the benchmark output now backs the rule instead of just documenting it.

## Why not fix the metric itself

Normalizing by content length (e.g. keyword hits per 100 chars) would still not make whole-document and chunk-level retrieval comparable: a whole document contains the answer *and* a lot of irrelevant material, so a length-normalized score would penalize whole-doc modes for verbosity rather than measuring whether the right information was retrieved. That comparison is what `hit_rate_at_k` / `ndcg_at_5` (doc-id based, retrieval-unit-agnostic) already do correctly. KwCorr's proper role is a lightweight non-LLM check that a mode's top-3 retrieval keeps surfacing the same keywords release over release — a same-mode regression signal, not a leaderboard.

## Remaining work

Findings (i) and (j) from the eval audit (LLM-judge scoring the top chunk instead of a generated answer, CtxRel circularity for hybrid modes, unmeasured-as-zero cost/latency columns) remain open and tracked separately.
