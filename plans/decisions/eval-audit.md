# Decision: Fixing the fail-open regression gate

**Status:** Resolved
**Owner:** Himanshu

---

## The finding

`backend/eval/results/baseline.json` was a half-completed migration artifact. Phase 9a (see [retrieval-finding.md](retrieval-finding.md)) introduced doc-id metrics (`avg_precision_at_3_doc`, `avg_recall_at_3_doc`) and repointed `thresholds.json`'s `best_mode` from `hybrid+rerank+filter` to `hybrid`, but the committed baseline was never re-saved: it still held the deprecated title-based metrics (`avg_precision_at_3`, `avg_recall_at_3`) under `mode: "hybrid+rerank+filter"`.

`check_regression.py`'s per-metric loop does:

```python
base_val = baseline.get(metric)
curr_val = current.get(metric)
if base_val is None or curr_val is None:
    continue
```

Every metric named in `thresholds.json`'s `metrics_to_gate` (`avg_precision_at_3_doc`, `avg_recall_at_3_doc`, `avg_context_relevance`) was absent from the stale baseline, so all three were silently skipped every run. The gate could never fail — retrieval quality could drop to zero and CI would still print "All metrics within threshold."

## Fix

1. Re-ran `check_regression.py --save-baseline` against the current `hybrid` mode result, which carries doc-id metrics. New baseline: `avg_precision_at_3_doc=0.109`, `avg_recall_at_3_doc=0.2692`, `avg_context_relevance=0.1805`, `mode: "hybrid"` — matching `thresholds.json`'s `best_mode`.
2. Added a `--strict` flag to `check_regression.py`. Without it, a gated metric missing from baseline or the current run still prints a warning and is skipped (permissive for local runs). With `--strict` (now wired into `.github/workflows/eval.yml`), a skipped gated metric is treated as a failure and exits 1 — so a future half-completed metric migration breaks CI instead of passing silently.
3. Added `backend/tests/test_eval_guards.py`: `test_baseline_covers_every_gated_metric` and `test_baseline_mode_matches_best_mode` assert the baseline actually matches what `thresholds.json` gates on; `test_strict_flag_fails_on_missing_gated_metric` exercises `--strict` directly against a synthetic stale baseline.

## Verification

Guard tests were written first and failed against the pre-fix repo state (red), confirming they reproduce the bug:

```
FAILED test_baseline_covers_every_gated_metric — baseline.json lacks gated metrics
  ['avg_precision_at_3_doc', 'avg_recall_at_3_doc']
FAILED test_baseline_mode_matches_best_mode — 'hybrid+rerank+filter' != 'hybrid'
FAILED test_strict_flag_fails_on_missing_gated_metric — TypeError: no --strict flag existed
```

All three pass after the fix (green).

## Remaining work

Other issues catalogued during the same review pass (keyword-mode ID namespace mismatch, benchmark history mixing incomparable runs, hit-rate columns published deeper than actual retrieval depth, chunking audit resting on the deprecated title-based metric) are tracked separately and not part of this fix.
