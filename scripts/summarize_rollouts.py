#!/usr/bin/env python
"""Summarize one VEP-OPSD run's rollout artifacts.

Usage:
    python scripts/summarize_rollouts.py OUTPUT_DIR [OUTPUT_DIR ...]

Reads ``rollouts/candidates-rank*.jsonl`` and reports the things that decide
whether a run is healthy and how its time is spent:

  * samples / optimizer steps seen, skip rate (groups below the consistency floor)
  * generated length per candidate and per group (the group maximum is what the
    rollouts actually pay for, because the four sequences decode in lockstep)
  * teacher scores, teacher preference entropy, student scores
  * per-stage seconds (rollout / verify / student) when the run recorded them
  * answer-tag coverage
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_opsd.prompts import is_degenerate_answer  # noqa: E402


def describe(values: list[float]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    return (
        f"mean {st.mean(values):.1f} median {ordered[len(ordered) // 2]:.1f} "
        f"p95 {ordered[int(0.95 * (len(ordered) - 1))]:.1f} max {ordered[-1]:.1f}"
    )


def summarize(directory: Path) -> None:
    records = []
    for path in sorted((directory / "rollouts").glob("candidates-rank*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        print(f"{directory}: no rollout records")
        return

    steps = sorted({record["global_step"] for record in records})
    lengths = [len(candidate["token_ids"]) for record in records for candidate in record["candidates"]]
    # A group can be empty: every candidate was degenerate and no resample committed
    # to a prediction. Those groups are skipped, not trained on.
    group_max = [
        max((len(c["token_ids"]) for c in record["candidates"]), default=0) for record in records
    ]
    empty_groups = sum(1 for record in records if not record["candidates"])
    answers = sum(
        1 for record in records for candidate in record["candidates"] if candidate["answer"]
    )
    if empty_groups:
        print(f"  empty groups (all candidates degenerate): {empty_groups}/{len(records)}")
    skipped = sum(record["skipped"] for record in records)
    reasons = {}
    for record in records:
        if record.get("skip_reason"):
            reasons[record["skip_reason"]] = reasons.get(record["skip_reason"], 0) + 1
    spreads = [record["teacher_spread"] for record in records if "teacher_spread" in record]
    degenerate = [record.get("degenerate_candidates", 0) for record in records]
    resampled = sum(1 for record in records if record.get("resampled"))
    teacher = [score for record in records for score in record["teacher_scores"]]
    student = [score for record in records for score in record["student_scores"]]
    entropies = []
    for record in records:
        probs = record.get("teacher_preference")
        if probs:
            entropies.append(-sum(p * math.log(p) for p in probs if p > 0))

    print(f"===== {directory}")
    print(f"  records {len(records)} over {len(steps)} steps ({steps[0]}..{steps[-1]})")
    print(f"  skipped groups {skipped}/{len(records)} ({100 * skipped / len(records):.1f}%)")
    if "micro_batch_has_signal" in records[0]:
        blind_steps = sorted(
            {
                record["global_step"]
                for record in records
                if not record["micro_batch_has_signal"]
            }
        )
        trained_steps = len({record["global_step"] for record in records}) - len(blind_steps)
        print(
            f"  steps with any trained group: {trained_steps}"
            + (f"; {len(blind_steps)} steps had none (loss 0, no update)" if blind_steps else "")
        )
    if reasons:
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
            print(f"    {reason}: {count}")
    if spreads:
        print(f"  teacher score spread: {describe(spreads)}  (a flat teacher ranks nothing)")
        print(
            "  flat groups (spread < 0.05): "
            f"{100 * sum(1 for value in spreads if value < 0.05) / len(spreads):.0f}%"
        )
    if any(degenerate):
        print(
            f"  degenerate candidates discarded: {sum(degenerate)} "
            f"(groups resampled: {resampled})"
        )
    print(f"  generated tokens/candidate: {describe(lengths)}")
    print(f"  generated tokens/group (decode steps paid): {describe(group_max)}")
    print(f"  answer tag present: {answers}/{len(lengths)}")
    print(f"  teacher scores: {describe(teacher)}")
    refusals = sum(
        1
        for record in records
        for candidate in record["candidates"]
        if is_degenerate_answer(candidate["answer"])
    )
    total = sum(len(record["candidates"]) for record in records)
    print(f"  degenerate/refusal candidates kept: {refusals}/{total}")
    print(f"  student scores: {describe(student)}")
    print(f"  teacher preference entropy: {describe(entropies)} nats (log 4 = {math.log(4):.2f})")

    stage_keys = ("rollout", "verify", "student")
    if all("stage_seconds" in record for record in records):
        print("  stage seconds per sample:")
        for key in stage_keys:
            values = [record["stage_seconds"][key] for record in records if key in record["stage_seconds"]]
            print(f"    {key:8s} {describe(values)}")
        totals = [
            sum(record["stage_seconds"].get(key, 0.0) for key in stage_keys) for record in records
        ]
        print(f"    {'total':8s} {describe(totals)}")
        print(
            f"  => {st.mean(totals):.1f} s/sample; with global batch 128 and 8 ranks "
            f"(16 samples per rank per step) that is ~{st.mean(totals) * 16 / 60:.1f} min/optimizer step"
        )
    else:
        print("  (no stage timings recorded: this run predates the instrumentation)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for argument in sys.argv[1:]:
        summarize(Path(argument))
