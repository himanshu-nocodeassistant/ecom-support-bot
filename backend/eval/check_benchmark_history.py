"""Pre-commit guard for docs/benchmark-history.jsonl.

Catches the 6c bug class: history entries missing a fingerprint, or a
commit that mixes entries from different (n_docs, metric_version)
fingerprints without going through append_benchmark_history. See
plans/decisions/eval-audit.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HISTORY_PATH = Path(__file__).parent.parent.parent / "docs" / "benchmark-history.jsonl"


def check(history_path: Path = HISTORY_PATH) -> list[str]:
    if not history_path.exists():
        return []

    errors = []
    fingerprints = set()
    for lineno, line in enumerate(history_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc})")
            continue

        missing = [k for k in ("n_docs", "metric_version") if k not in entry]
        if missing:
            errors.append(
                f"line {lineno}: entry missing {missing}; append via "
                "backend.eval.run.append_benchmark_history, don't hand-edit "
                "this file"
            )
            continue

        fingerprints.add((entry["n_docs"], entry["metric_version"]))

    if len(fingerprints) > 1:
        errors.append(
            f"docs/benchmark-history.jsonl mixes {len(fingerprints)} incomparable "
            f"fingerprints (n_docs, metric_version): {sorted(fingerprints)}. "
            "Reset the file when the KB or metric definitions change instead of "
            "appending across a fingerprint change (see 6c)."
        )

    return errors


def main() -> int:
    errors = check()
    if errors:
        print("benchmark-history validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
