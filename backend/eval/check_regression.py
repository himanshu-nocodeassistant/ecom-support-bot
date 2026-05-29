"""6d: Regression gate — compare current eval results against committed baseline.

Usage (called by CI):
    python -m backend.eval.check_regression

Exit codes:
    0  — all metrics within threshold
    1  — one or more metrics regressed beyond threshold
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
THRESHOLDS_PATH = EVAL_DIR / "thresholds.json"
BASELINE_PATH = RESULTS_DIR / "baseline.json"


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


def check_retrieval_regression(thresholds: dict, baseline: dict, current: dict) -> list[str]:
    failures: list[str] = []
    max_drop = thresholds["regression_max_drop"]
    for metric in thresholds["metrics_to_gate"]:
        base_val = baseline.get(metric)
        curr_val = current.get(metric)
        if base_val is None or curr_val is None:
            continue
        drop = base_val - curr_val
        if drop > max_drop:
            failures.append(
                f"  {metric}: baseline={base_val:.4f}  current={curr_val:.4f}  "
                f"drop={drop:.4f} > threshold={max_drop:.2f}"
            )
    return failures


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


def check_agent_regression(thresholds: dict, baseline: dict, current: dict) -> list[str]:
    failures: list[str] = []
    max_drop = thresholds.get("agent_regression_max_drop", 0.10)
    for metric in thresholds.get("agent_metrics_to_gate", []):
        base_val = baseline.get(metric)
        curr_val = current.get(metric)
        if base_val is None or curr_val is None:
            continue
        drop = base_val - curr_val
        if drop > max_drop:
            failures.append(
                f"  {metric}: baseline={base_val:.4f}  current={curr_val:.4f}  "
                f"drop={drop:.4f} > threshold={max_drop:.2f}"
            )
    return failures


def main() -> None:
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
        print(
            f"WARNING: no result file for best_mode={thresholds['best_mode']}; skipping retrieval check",
            file=sys.stderr,
        )
    else:
        retrieval_baseline = baseline.get("retrieval", {})
        failures = check_retrieval_regression(thresholds, retrieval_baseline, current)
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

    # Agent regression
    agent_baseline = baseline.get("agent", {})
    current_agent = _current_agent()
    if current_agent is None:
        print("WARNING: no agent_eval.json found; skipping agent check", file=sys.stderr)
    elif not agent_baseline:
        print("No agent baseline recorded; skipping agent regression check")
    else:
        failures = check_agent_regression(thresholds, agent_baseline, current_agent)
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Baseline saved to {BASELINE_PATH}")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    if "--save-baseline" in sys.argv:
        save_baseline()
    else:
        main()
