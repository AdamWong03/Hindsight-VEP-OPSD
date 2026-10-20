#!/usr/bin/env python
"""Summarize privileged OPSD trajectory records.

Usage:
    python scripts/summarize_rollouts.py OUTPUT_DIR [OUTPUT_DIR ...]
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def describe(values: list[float]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    return (
        f"mean {st.mean(values):.3f} median {st.median(values):.3f} "
        f"p95 {ordered[int(0.95 * (len(ordered) - 1))]:.3f} max {ordered[-1]:.3f}"
    )


def summarize(directory: Path) -> None:
    records = []
    for path in sorted((directory / "rollouts").glob("trajectories-rank*.jsonl")):
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not records:
        print(f"{directory}: no trajectory records")
        return

    steps = {record["global_step"] for record in records}
    lengths = [record["trajectory"]["length"] for record in records]
    answers = sum(bool(record["trajectory"]["answer"]) for record in records)
    print(f"===== {directory}")
    print(f"  records: {len(records)} over {len(steps)} optimizer steps")
    print(f"  generated tokens: {describe(lengths)}")
    print(f"  answer tag present: {answers}/{len(records)}")
    print(
        "  Teacher log P(sampled token): "
        + describe([record["teacher_sampled_token_logp_mean"] for record in records])
    )
    print(
        "  token JSD: "
        + describe([record["token_jsd_mean"] for record in records])
    )
    for stage in ("rollout", "teacher", "student"):
        values = [
            record["stage_seconds"][stage]
            for record in records
            if stage in record.get("stage_seconds", {})
        ]
        print(f"  {stage} seconds/sample: {describe(values)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for argument in sys.argv[1:]:
        summarize(Path(argument))
