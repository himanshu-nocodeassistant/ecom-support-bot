"""6d: Regression gate — compare current eval results against committed baseline.

Usage (called by CI):
    python -m backend.eval.check_regression --strict

--strict treats a gated metric missing from baseline.json (or the current
run) as a failure instead of silently skipping it. Without --strict, missing
metrics only print a warning — used for permissive local runs.

Exit codes:
    0  — all metrics within threshold
    1  — one or more metrics regressed beyond threshold (or, in --strict
         mode, one or more gated metrics were skipped)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
THRESHOLDS_PATH = EVAL_DIR / "thresholds.json"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
LIVE_EVAL_METRIC_VERSION = "adversarial-contract-v1"
AGENT_MODEL = "claude-haiku-4-5-20251001"


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _current_best(thresholds: dict) -> dict | None:
    """Load the current run result for the best_mode."""
    best_mode = thresholds["best_mode"]
    fname = best_mode.replace("+", "_") + ".json"
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    return _load_json(p)


def _current_agent() -> dict | None:
    p = RESULTS_DIR / "agent_eval.json"
    return _load_json(p) if p.exists() else None


def _current_adversarial() -> dict | None:
    p = RESULTS_DIR / "adversarial_eval.json"
    return _load_json(p) if p.exists() else None


def validate_live_eval_metadata(
    current: dict, dataset_path: Path, model: str, strict: bool = False
) -> list[str]:
    """Return strict-mode failures for stale or incompatible live-eval results.

    Live results are intentionally ignored by git, so a developer can otherwise
    accidentally gate a change with output produced against a different dataset,
    model, or scoring contract.
    """
    if not strict:
        return []

    metadata = current.get("metadata")
    if not isinstance(metadata, dict):
        return ["  --strict: live eval metadata is missing"]

    expected = {
        "dataset": dataset_path.name,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "model": model,
        "metric_version": LIVE_EVAL_METRIC_VERSION,
    }
    failures = []
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append(
                f"  --strict: live eval metadata {key}={metadata.get(key)!r} "
                f"does not match expected {value!r}"
            )
    return failures


def check_retrieval_regression(
    thresholds: dict, baseline: dict, current: dict, strict: bool = False
) -> tuple[list[str], list[str]]:
    """Returns (failures, skipped). In --strict mode, skipped gated metrics are
    treated as failures so CI can't silently pass with a stale/incomplete baseline."""
    failures: list[str] = []
    skipped: list[str] = []
    max_drop = thresholds["regression_max_drop"]
    for metric in thresholds["metrics_to_gate"]:
        base_val = baseline.get(metric)
        curr_val = current.get(metric)
        if base_val is None or curr_val is None:
            skipped.append(metric)
            continue
        drop = base_val - curr_val
        if drop > max_drop:
            failures.append(
                f"  {metric}: baseline={base_val:.4f}  current={curr_val:.4f}  "
                f"drop={drop:.4f} > threshold={max_drop:.2f}"
            )
    if strict and skipped:
        failures.append(
            f"  --strict: gated metric(s) missing from baseline or current run: {skipped}"
        )
    return failures, skipped


def check_memory_regression(thresholds: dict, baseline: dict, current: dict) -> list[str]:
    """Gate on memory_recall_rate using a minimum floor rather than a baseline drop."""
    failures: list[str] = []
    min_rate = thresholds.get("memory_recall_rate_min")
    if min_rate is None:
        return failures
    curr_val = current.get("memory_recall_rate")
    if curr_val is None:
        return failures
    if curr_val < min_rate:
        failures.append(f"  memory_recall_rate: current={curr_val:.4f} < minimum={min_rate:.2f}")
    return failures


def check_adversarial_regression(
    thresholds: dict, current: dict, strict: bool = False
) -> list[str]:
    """Gate on adversarial metric floors (not baseline-relative)."""
    failures: list[str] = []
    mins = thresholds.get("adversarial_metrics_min", {})
    for metric, min_val in mins.items():
        curr_val = current.get(metric)
        if curr_val is None:
            if strict:
                failures.append(
                    f"  --strict: adversarial metric missing from current run: {metric}"
                )
            continue
        if curr_val < min_val:
            failures.append(f"  {metric}: current={curr_val:.4f} < minimum={min_val:.2f}")
    return failures


def check_agent_regression(
    thresholds: dict, baseline: dict, current: dict, strict: bool = False
) -> list[str]:
    failures: list[str] = []
    max_drop = thresholds.get("agent_regression_max_drop", 0.10)
    for metric in thresholds.get("agent_metrics_to_gate", []):
        base_val = baseline.get(metric)
        curr_val = current.get(metric)
        if base_val is None or curr_val is None:
            if strict:
                missing_from = "baseline" if base_val is None else "current run"
                failures.append(f"  --strict: agent metric missing from {missing_from}: {metric}")
            continue
        drop = base_val - curr_val
        if drop > max_drop:
            failures.append(
                f"  {metric}: baseline={base_val:.4f}  current={curr_val:.4f}  "
                f"drop={drop:.4f} > threshold={max_drop:.2f}"
            )
    return failures


def main(strict: bool = False) -> None:
    if not THRESHOLDS_PATH.exists():
        print("ERROR: thresholds.json not found", file=sys.stderr)
        sys.exit(1)

    if not BASELINE_PATH.exists():
        print(
            "No baseline.json found — skipping regression check. "
            "Run with --save-baseline to establish one.",
            file=sys.stderr,
        )
        sys.exit(0)

    thresholds = _load_json(THRESHOLDS_PATH)
    baseline = _load_json(BASELINE_PATH)

    all_failures: list[str] = []

    # Retrieval regression
    current = _current_best(thresholds)
    if current is None:
        msg = f"no result file for best_mode={thresholds['best_mode']}; skipping retrieval check"
        if strict:
            all_failures.append(f"  --strict: {msg}")
            print(f"ERROR: {msg}", file=sys.stderr)
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
    else:
        retrieval_baseline = baseline.get("retrieval", {})
        failures, skipped = check_retrieval_regression(
            thresholds, retrieval_baseline, current, strict=strict
        )
        if failures:
            print("RETRIEVAL REGRESSION DETECTED:")
            print("\n".join(failures))
            all_failures.extend(failures)
        else:
            print(f"Retrieval OK — mode={thresholds['best_mode']}")
            for m in thresholds["metrics_to_gate"]:
                b = retrieval_baseline.get(m, "n/a")
                c = current.get(m, "n/a")
                print(f"  {m}: baseline={b}  current={c}")
        if skipped and not strict:
            print(
                f"WARNING: gated metric(s) skipped (missing from baseline/current): {skipped}",
                file=sys.stderr,
            )

    # Agent regression
    agent_baseline = baseline.get("agent", {})
    current_agent = _current_agent()
    if current_agent is None:
        msg = "no agent_eval.json found; skipping agent check"
        if strict and thresholds.get("agent_metrics_to_gate"):
            all_failures.append(f"  --strict: {msg}")
            print(f"ERROR: {msg}", file=sys.stderr)
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
    elif not agent_baseline:
        msg = "no agent baseline recorded; skipping agent regression check"
        if strict and thresholds.get("agent_metrics_to_gate"):
            all_failures.append(f"  --strict: {msg}")
            print(f"ERROR: {msg}", file=sys.stderr)
        else:
            print(msg)
    else:
        failures = validate_live_eval_metadata(
            current_agent, EVAL_DIR / "agent_fixtures.json", AGENT_MODEL, strict=strict
        )
        failures.extend(
            check_agent_regression(thresholds, agent_baseline, current_agent, strict=strict)
        )
        if failures:
            print("AGENT REGRESSION DETECTED:")
            print("\n".join(failures))
            all_failures.extend(failures)
        else:
            print("Agent eval OK")
            for m in thresholds.get("agent_metrics_to_gate", []):
                b = agent_baseline.get(m, "n/a")
                c = current_agent.get(m, "n/a")
                print(f"  {m}: baseline={b}  current={c}")

    # Adversarial regression (absolute floor, not baseline-relative)
    current_adversarial = _current_adversarial()
    if current_adversarial is None:
        msg = "no adversarial_eval.json found; skipping adversarial check"
        if strict and thresholds.get("adversarial_metrics_min"):
            all_failures.append(f"  --strict: {msg}")
            print(f"ERROR: {msg}", file=sys.stderr)
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
    elif not thresholds.get("adversarial_metrics_min"):
        print("No adversarial thresholds configured; skipping")
    else:
        failures = validate_live_eval_metadata(
            current_adversarial,
            EVAL_DIR / "adversarial_queries.json",
            AGENT_MODEL,
            strict=strict,
        )
        failures.extend(
            check_adversarial_regression(thresholds, current_adversarial, strict=strict)
        )
        if failures:
            print("ADVERSARIAL REGRESSION DETECTED:")
            print("\n".join(failures))
            all_failures.extend(failures)
        else:
            print("Adversarial eval OK")
            for m, min_val in thresholds.get("adversarial_metrics_min", {}).items():
                c = current_adversarial.get(m, "n/a")
                print(f"  {m}: current={c}  minimum={min_val}")

    if all_failures:
        print(f"\n{len(all_failures)} regression(s) found. Failing CI.")
        sys.exit(1)
    else:
        print("\nAll metrics within threshold. CI passes.")
        sys.exit(0)


def save_baseline() -> None:
    """Snapshot current results as the new baseline."""
    thresholds = _load_json(THRESHOLDS_PATH)
    snapshot: dict = {}

    current = _current_best(thresholds)
    if current:
        snapshot["retrieval"] = {m: current.get(m) for m in thresholds["metrics_to_gate"]}
        snapshot["retrieval"]["mode"] = thresholds["best_mode"]

    agent = _current_agent()
    if agent:
        snapshot["agent"] = {m: agent.get(m) for m in thresholds.get("agent_metrics_to_gate", [])}

    adversarial = _current_adversarial()
    if adversarial:
        adv_keys = list(thresholds.get("adversarial_metrics_min", {}).keys())
        snapshot["adversarial"] = {m: adversarial.get(m) for m in adv_keys}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Baseline saved to {BASELINE_PATH}")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    if "--save-baseline" in sys.argv:
        save_baseline()
    else:
        main(strict="--strict" in sys.argv)
