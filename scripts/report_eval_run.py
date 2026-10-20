"""Compact report for one evaluated benchmark directory.

Usage:
    python scripts/report_eval_run.py <eval_dir> [<eval_dir> ...]

Prints, per directory, the lmms-eval aggregated metrics and -- for TwiFF -- the
primary all-items answer score (missing answer counts as zero) plus the valid /
conditional means, mirroring the PFSD evaluation convention. Nothing is
recomputed from the model; only recorded artifacts are read.
"""
import json
import sys
from pathlib import Path

import numpy as np


def newest(paths):
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def lmms_metrics(directory):
    results = newest(list(Path(directory).rglob("*_results.json")))
    if results is None:
        return None, None
    payload = json.loads(results.read_text())
    out = {}
    for task, values in (payload.get("results") or {}).items():
        numeric = {k: v for k, v in values.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "alias"}
        if numeric:
            out[task] = numeric
    return out, str(results)


def twiff_answer_scores(directory):
    """All-items mean with missing answers as zero, plus valid/conditional means."""
    judge = newest(list(Path(directory).rglob("*.answer_judge.*.jsonl")))
    if judge is None:
        return None
    rows = [json.loads(line) for line in judge.read_text().splitlines() if line.strip()]
    scores, valid = [], []
    for row in rows:
        value = row.get("judge_answer_score")
        ok = bool(row.get("has_pred_answer_tag")) and isinstance(value, (int, float)) and np.isfinite(value)
        scores.append(float(value) if ok else 0.0)
        valid.append(ok)
    scores, valid = np.array(scores), np.array(valid, dtype=bool)
    summary = newest(list(Path(directory).rglob("*.answer_judge.*.summary.json")))
    return {
        "n": int(len(scores)),
        "all_items_mean": float(scores.mean()) if len(scores) else None,
        "valid_answers": int(valid.sum()),
        "valid_mean": float(scores[valid].mean()) if valid.any() else None,
        "judge_summary": json.loads(summary.read_text()) if summary else None,
        "judge_records": str(judge),
    }


def main(directories):
    for directory in directories:
        path = Path(directory)
        print(f"===== {path}")
        if not path.is_dir():
            print("  (missing directory)")
            continue
        metrics, source = lmms_metrics(path)
        if metrics:
            print(f"  lmms-eval aggregated metrics ({source}):")
            for task, values in sorted(metrics.items()):
                for key, value in sorted(values.items()):
                    print(f"    {task}.{key} = {value:.4f}" if isinstance(value, float) else f"    {task}.{key} = {value}")
        else:
            print("  lmms-eval results_*.json: not found (generation incomplete)")
        answers = twiff_answer_scores(path)
        if answers:
            print("  TwiFF answer score (primary convention: missing answer = 0):")
            print(f"    items={answers['n']} all_items_mean={answers['all_items_mean']:.4f} "
                  f"valid={answers['valid_answers']} valid_mean="
                  f"{answers['valid_mean'] if answers['valid_mean'] is None else round(answers['valid_mean'], 4)}")
            summary = answers["judge_summary"] or {}
            if summary:
                print(f"    judge: n_records={summary.get('n_records')} n_scored_err={summary.get('n_scored_err')} "
                      f"answer_score_mean={summary.get('answer_score_mean')}")
            print(f"    judge records: {answers['judge_records']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
