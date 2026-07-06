# Decision: Silent fulltext fallback was contaminating hybrid benchmark numbers

**Status:** Resolved
**Owner:** Himanshu

---

## The finding

While re-running the eval to verify the KwCorr cross-mode fix (`plans/decisions/kwcorr-cross-mode.md`, finding 6h in the 2026-07-03 repurposing handoff), repeated `--mode hybrid` runs against the live Supabase KB produced wildly different, non-reproducible retrieval scores (`avg_precision_at_3_doc` swinging between 0.301, 0.147, 0.122, 0.128 across otherwise-identical runs). That ruled out KB drift or embedding non-determinism as the cause and pointed at the retrieval path itself.

Root cause: `embed_query()` (`backend/app/data_loader.py`), the function `PostgresRepository._hybrid_search` uses to embed the incoming query, made a single unretried Voyage API call — unlike `embed_queries()`/`embed_texts()`, which already retried through `_voyage_embed_with_retry`. `_hybrid_search` wrapped that call in a bare `except Exception: return self._fulltext_search(...)` with no logging and no marker on the returned results. Two independent failure modes tripped this same silent path during this investigation:

1. **Voyage rate limiting.** The project's Voyage account has no payment method on file, capping it at **3 requests/minute** (confirmed directly against the API — `RateLimitError: ... reduced rate limits of 3 RPM and 10K TPM`). The eval harness calls `search_knowledge` up to twice per query (k=3 pass and the eval-only k=`EVAL_DEPTH` pass), so a 62-query hybrid run issues well over 100 sequential embed calls — guaranteed to exceed 3 RPM without pacing. Every rate-limited call degraded silently to fulltext, and because fulltext's `plainto_tsquery` match is sparse for paraphrastic queries, many degraded queries returned `retrieved_doc_ids: []` — a hard miss masquerading as a real retrieval result.
2. **Network interruption after sleep.** Mid-investigation, the laptop slept and woke with a dead DNS resolver; every hybrid+rerank query in that run failed with `NameResolutionError` and again degraded silently to fulltext. This confirmed the failure mode isn't specific to rate limiting — any Voyage outage (network blip, API incident, expired key) hits the same unguarded fallback.

The practical consequence: **the previously committed `docs/benchmark.md` is unreliable.** Its Mode × Metric table showed `P@3 (doc)` = `R@3 (doc)` = **0.301 / 0.904 identically for keyword, fulltext, hybrid, and hybrid+rerank** — four different retrieval implementations converging on the exact same score is the fingerprint of widespread fulltext-fallback contamination, not four modes that happen to perform identically. (`hybrid+rerank`'s H@1/H@3/NDCG@5/MRR columns did differ slightly from the others, meaning the contamination was partial — some queries got real hybrid results, most did not.) This also means `plans/decisions/reranking.md`'s comparison (`hybrid` NDCG@5=0.253 vs `hybrid+rerank`=0.288) was plausibly measured against a similarly degraded baseline; see the note added there.

## Fix

1. **`backend/app/data_loader.py`:** `embed_query()` now delegates to `embed_queries()` (`return embed_queries([text], api_key=api_key)[0]`) instead of making its own unretried call, so it inherits the same rate-limit retry behavior as every other embedding call in the codebase.
2. **`backend/app/repository.py`:** all three silent-fallback branches in `_hybrid_search` (embed failure, Postgres query failure, empty result set) now log a `logging.getLogger(__name__).warning(...)` naming the failure, and tag every result they return with `"degraded": "<reason>"` (`embed_query_failed`, `hybrid_query_failed`, or `hybrid_no_rows`). Fulltext fallback still happens — availability is the right call in production — but it's no longer invisible.
3. **`backend/eval/run.py`:** `evaluate_mode` records a per-query `degraded` field (the reason string, or `None`) and an `n_degraded` count per mode. A new `--strict` flag aborts the run with a non-zero exit if any query degraded, instead of silently publishing contaminated numbers; the per-mode summary prints a loud warning even without `--strict`.
4. **`.github/workflows/eval.yml`:** both the gated hybrid-mode eval step and the benchmark-regeneration step now pass `--strict`, so a degraded CI run fails instead of auto-committing bad numbers to `docs/benchmark.md`.
5. **`backend/eval/run.py` pacing:** `evaluate_mode` now sleeps ~21s between each Voyage-triggering call when running `hybrid`/`hybrid+rerank` modes, keeping the harness under 3 RPM proactively instead of relying on retry-after-429 backoff (which, chained across 100+ calls, turned a single run into 40+ minutes of thrashing). This is an eval-harness concern only — production's `embed_query` still makes one call per real user query and was never the bottleneck; the eval loop's sequential single-query calls were.
6. **`backend/tests/test_eval_guards.py`:** added `test_embed_query_retries_through_rate_limits` (asserts `embed_query` delegates to `embed_queries`, red before the fix) and `test_hybrid_search_marks_degraded_results_instead_of_silent_fallback` (asserts a query served by the fulltext fallback carries `degraded`, red before the fix).

## Verification

Re-ran each mode against the live Supabase KB with the fix in place, throttled to stay under 3 RPM, confirming `n_degraded == 0` for all four modes. `hybrid` was additionally run twice back-to-back to check determinism before publishing anything — both runs produced byte-identical metrics:

| Metric | Run 1 | Run 2 |
|---|---|---|
| avg_precision_at_3_doc | 0.6474 | 0.6474 |
| avg_recall_at_3_doc | 0.9423 | 0.9423 |
| avg_ndcg_at_5 | 0.9339 | 0.9339 |
| avg_mrr | 0.9266 | 0.9266 |
| avg_answer_correctness_kw | 0.8071 | 0.8071 |

Clean numbers, before (contaminated, previously committed) vs. after (clean, this fix):

| Mode | P@3(doc)/R@3(doc) before | P@3(doc)/R@3(doc) after | n_degraded |
|---|---|---|---|
| `keyword` | 0.301 / 0.904 | 0.301 / 0.904 (unchanged — never calls embed_query) | 0 |
| `fulltext` | 0.301 / 0.904 | **0.083 / 0.250** | 0 |
| `hybrid` | 0.301 / 0.904 | **0.647 / 0.942** | 0 |
| `hybrid+rerank` | 0.301 / 0.904 | **0.647 / 0.942** (NDCG@5 0.960, MRR 0.954) | 0 |

`fulltext` never calls `embed_query` at all (it's pure Postgres `ts_rank`), so its previously-identical 0.301/0.904 score could not have come from the same mechanism as hybrid's degradation — it was independently wrong, most likely because the committed benchmark.md's rows were generated from a run where fulltext, hybrid, and hybrid+rerank all happened to degrade to (or were dominated by) the same underlying fulltext path, producing near-identical numbers across three structurally different modes by coincidence of the shared fallback. The `n_degraded` counter did not exist at the time that benchmark was generated, so this cannot be confirmed after the fact from the committed artifact alone — only inferred from the signature. This is exactly why the counter and `--strict` gate now exist: the next time this happens, it will be loud instead of retroactively inferred.

All four `results/*.json` files, `docs/benchmark.md`, and `docs/benchmark-history.jsonl` (tagged `metric_version: doc-id-v2`, same fingerprint as the prior entry, so the sparkline stays comparable) were regenerated together in this pass, per the handoff's rule that a metric-affecting fix must regenerate every downstream artifact in the same commit.

## Remaining work

`plans/decisions/reranking.md`'s hybrid vs. hybrid+rerank comparison (0.253/0.288 NDCG@5) was measured before this fix existed and shows the exact same telltale pattern (hybrid's own recorded P@3(doc) of 0.109 there doesn't match either this fix's clean 0.647 or the previously-committed 0.301, i.e. it's a third, also-unverified number). A note has been added there; the comparison should be re-run and superseded now that a clean, deterministic hybrid baseline exists (P@3(doc)=0.6474, NDCG@5=0.9339) — see this doc's verification table for the number to compare against.

Findings (i)-(j) from the eval audit (LLM-judge scoring the top retrieved chunk instead of a generated answer, CtxRel circularity for hybrid modes) remain open and tracked separately, per `plans/decisions/eval-retrieval-depth.md`.
