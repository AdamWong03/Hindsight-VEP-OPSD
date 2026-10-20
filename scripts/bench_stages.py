#!/usr/bin/env python
"""Per-stage timing and equivalence check for the VEP-OPSD rollout/verify/distill loop.

Runs a few real samples through the trainer's own stage helpers and reports

  * wall time for each stage (rollout generation, verifier scoring, student
    scoring),
  * the same stages scored the old way (one forward per candidate), so the
    batching gain is visible,
  * the numeric difference between the two paths (they must agree),
  * peak GPU memory.

Usage:
    python scripts/bench_stages.py --model DIR --data MANIFEST [--samples 2]
    python scripts/bench_stages.py --model DIR --data MANIFEST --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_opsd.config import HindsightOPSDConfig  # noqa: E402
from vep_opsd.data import VEP50KDataset  # noqa: E402
from vep_opsd.objective import (  # noqa: E402
    binary_yes_probability,
    length_normalized_log_probability,
    resolve_yes_no_tokens,
)
from vep_opsd.prompts import student_messages, verifier_messages  # noqa: E402
from vep_opsd.trainer import (  # noqa: E402
    HindsightOPSDTrainer,
    _model_inputs,
    _model_device,
)


def build_stage_runner(model, processor, config):
    """Bind the trainer's stage methods to a light stand-in (no Trainer needed)."""
    stand_in = SimpleNamespace(processor=processor, opsd_config=config, binary_spec=None)
    return stand_in, {
        "rollout": HindsightOPSDTrainer._sample_candidates.__get__(stand_in),
        "verify": HindsightOPSDTrainer._verify_candidate.__get__(stand_in),
        "student_prompt": HindsightOPSDTrainer._student_prompt.__get__(stand_in),
        "student": HindsightOPSDTrainer._student_score.__get__(stand_in),
    }


def verify_per_row(model, processor, spec, example, candidates):
    """Previous implementation: one no-grad forward per candidate."""
    scores = []
    for candidate in candidates:
        messages = verifier_messages(
            example["observed_frames"], example["future_frames"], example["question"], candidate["text"]
        )
        images = list(example["observed_frames"]) + list(example["future_frames"])
        _, inputs = _model_inputs(processor, messages, images, _model_device(model))
        outputs = model(**inputs, use_cache=False, return_dict=True)
        last = int(inputs["attention_mask"].sum().item()) - 1
        scores.append(float(binary_yes_probability(outputs.logits[:, last, :], spec)[0].item()))
    return scores


def student_per_row(model, processor, example, candidates):
    """Previous implementation: one gradient-enabled forward per candidate, full logits."""
    scores = []
    for candidate in candidates:
        messages = student_messages(example["observed_frames"], example["question"])
        _, prompt = _model_inputs(processor, messages, example["observed_frames"], _model_device(model))
        prompt_len = int(prompt["input_ids"].shape[1])
        ids = torch.tensor(candidate["token_ids"], device=_model_device(model), dtype=torch.long)
        inputs = dict(prompt)
        inputs["input_ids"] = torch.cat([prompt["input_ids"], ids.unsqueeze(0)], dim=1)
        inputs["attention_mask"] = torch.cat(
            [
                prompt["attention_mask"],
                torch.ones((1, ids.numel()), device=ids.device, dtype=prompt["attention_mask"].dtype),
            ],
            dim=1,
        )
        if "token_type_ids" in inputs:
            suffix = inputs["token_type_ids"][:, -1:].expand(-1, ids.numel())
            inputs["token_type_ids"] = torch.cat([inputs["token_type_ids"], suffix], dim=1)
        outputs = model(**inputs, use_cache=False, return_dict=True)
        scores.append(length_normalized_log_probability(outputs.logits, prompt_len, ids))
    return torch.stack(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", default=None, help="write the table as JSON")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    config = HindsightOPSDConfig(
        **({} if args.max_new_tokens is None else {"max_new_tokens": args.max_new_tokens})
    )
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=2 * 32 * 32, max_pixels=16384 * 32 * 32
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "right"
    print(f"loading {args.model} (bf16, {args.device})", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device)
    model.eval()
    for parameter in model.model.visual.parameters():
        parameter.requires_grad_(False)

    dataset = VEP50KDataset(args.data, args.video_root)
    stand_in, stages = build_stage_runner(model, processor, config)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def stage_peak(name):
        peaks[name] = max(peaks.get(name, 0.0), torch.cuda.max_memory_allocated() / 2**30)
        torch.cuda.reset_peak_memory_stats()

    totals = {"rollout": 0.0, "verify": 0.0, "verify_per_row": 0.0, "student": 0.0, "student_per_row": 0.0}
    peaks: dict[str, float] = {}
    deltas = {"verify": 0.0, "student": 0.0}
    for index in range(args.samples):
        example = dataset[index]
        started = time.perf_counter()
        candidates = stages["rollout"](model, [example])[0]
        totals["rollout"] += time.perf_counter() - started
        stage_peak("rollout")
        lengths = [len(c["token_ids"]) for c in candidates]

        started = time.perf_counter()
        with torch.no_grad():
            batched_scores = [stages["verify"](model, example, c) for c in candidates]
        totals["verify"] += time.perf_counter() - started
        stage_peak("verify")
        spec = stand_in.binary_spec
        started = time.perf_counter()
        with torch.no_grad():
            per_row_scores = verify_per_row(model, processor, spec, example, candidates)
        totals["verify_per_row"] += time.perf_counter() - started
        stage_peak("verify_per_row")
        deltas["verify"] = max(
            deltas["verify"], max(abs(a - b) for a, b in zip(batched_scores, per_row_scores))
        )

        started = time.perf_counter()
        prompt, prompt_len = stages["student_prompt"](model, example)
        batched_student = torch.stack(
            [stages["student"](model, prompt, prompt_len, c) for c in candidates]
        )
        totals["student"] += time.perf_counter() - started
        stage_peak("student")
        started = time.perf_counter()
        with torch.no_grad():
            per_row_student = student_per_row(model, processor, example, candidates)
        totals["student_per_row"] += time.perf_counter() - started
        stage_peak("student_per_row")
        deltas["student"] = max(
            deltas["student"],
            float((batched_student.detach().float() - per_row_student.detach().float()).abs().max()),
        )
        print(
            f"sample {index}: generated group-max={max(lengths)} mean={sum(lengths)/len(lengths):.0f} "
            f"skipped_would_be={all(score < config.minimum_consistency for score in batched_scores)}",
            flush=True,
        )
        del batched_student, per_row_student, batched_scores, per_row_scores
        torch.cuda.empty_cache()

    peak = torch.cuda.max_memory_allocated() / 2**30
    rows = [
        ("rollout (generate 4x)", totals["rollout"]),
        ("verify  (as now)", totals["verify"]),
        ("verify (before, = now)", totals["verify_per_row"]),
        ("student trimmed (fwd)", totals["student"]),
        ("student full logits (before)", totals["student_per_row"]),
    ]
    print("\nstage                    total s  per sample s   peak GiB")
    for name, value in rows:
        stage = (
            "rollout" if name.startswith("rollout")
            else ("verify" if "batched" in name else "verify_per_row") if name.startswith("verify")
            else ("student" if "batched" in name else "student_per_row")
        )
        print(f"{name:24s} {value:8.2f} {value / args.samples:13.2f} {peaks.get(stage, 0.0):10.1f}")
    print(f"\ntrimmed-logits speedup on the student pass: "
          f"{totals['student_per_row']/max(totals['student'],1e-9):.2f}x")
    print(f"max |delta| vs per-row: verify {deltas['verify']:.2e}, student {deltas['student']:.2e}")
    print(f"peak GPU memory: {peak:.1f} GiB")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "samples": args.samples,
                    "max_new_tokens": config.max_new_tokens,
                    "totals": totals,
                    "deltas": deltas,
                    "peak_gib": peak,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
