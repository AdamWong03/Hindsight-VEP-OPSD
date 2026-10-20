#!/usr/bin/env python
"""Pre-flight checks for a Hindsight VEP-OPSD run (no model weights loaded).

Catches the failures that would otherwise kill a cluster job minutes in:

1. every module the training entry point imports actually imports;
2. the checkpoint is a Qwen3-VL directory the trainer accepts;
3. one manifest sample decodes: observed/future frame indices are ordered and
   inside the 1..8 pool, the question survives cleaning, and decord can read
   both frame sets.

Usage:
    python scripts/preflight.py --model DIR --data FILE [--sample-id ID]
    python scripts/preflight.py --model DIR --data FILE --limit 25   # several samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_imports() -> None:
    import vep_opsd.data  # noqa: F401
    import vep_opsd.objective  # noqa: F401
    import vep_opsd.prompts  # noqa: F401
    import vep_opsd.train  # noqa: F401  (imports the entry point + HF classes)
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401

    print("ok  imports (vep_opsd.train, Qwen3VLForConditionalGeneration)")


def check_config(model: str) -> None:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model)
    if config.model_type != "qwen3_vl":
        raise SystemExit(f"FAIL model_type={config.model_type!r}, trainer requires 'qwen3_vl'")
    print(f"ok  config ({config.model_type}, hidden={getattr(config, 'hidden_size', '?')})")


def check_manifest(model: str, data: str, video_root: str | None, limit: int, sample_id: str | None) -> None:
    from vep_opsd.data import VEP50KDataset

    dataset = VEP50KDataset(data, video_root)
    indices = range(len(dataset)) if sample_id is None else [
        index for index in range(len(dataset)) if dataset.rows[index].get("id") == sample_id
    ]
    if not indices:
        raise SystemExit(f"FAIL sample id not found: {sample_id}")
    checked = 0
    for index in indices:
        if checked >= limit:
            break
        sample = dataset[index]
        print(
            f"ok  sample {sample['sample_id']}: observed={sample['observed_indices']} "
            f"future={sample['future_indices']} frames={len(sample['observed_frames'])}+"
            f"{len(sample['future_frames'])} q={sample['question'][:60]!r}"
        )
        checked += 1
    print(f"ok  manifest ({len(dataset)} rows, checked {checked})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Qwen3-VL checkpoint directory")
    parser.add_argument("--data", required=True, help="training manifest JSON")
    parser.add_argument("--video-root", default=None, help="root for relative manifest video paths")
    parser.add_argument("--limit", type=int, default=3, help="manifest samples to decode (default 3)")
    parser.add_argument("--sample-id", default=None, help="check one specific sample id")
    args = parser.parse_args()

    check_imports()
    check_config(args.model)
    check_manifest(args.model, args.data, args.video_root, args.limit, args.sample_id)
    print("\npreflight OK")


if __name__ == "__main__":
    main()
