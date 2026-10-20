#!/usr/bin/env python
"""Isolate where rollout time goes: decode forward vs HF sampling overhead.

Runs the same prompt through several generation settings on one GPU and reports
seconds per decode step for each, so the cost of the model forward can be
separated from the cost of HF's logits warping/sampling on a 151k vocabulary.

Usage:
    python scripts/bench_generate.py --model DIR --data MANIFEST [--steps 64]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_opsd.data import VEP50KDataset  # noqa: E402
from vep_opsd.prompts import student_messages  # noqa: E402


def time_generate(model, processor, prompt, **kwargs):
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        model.generate(**prompt, use_cache=True, **kwargs)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def manual_decode(model, prompt, steps: int, rows: int):
    """Greedy decode with a KV cache, no HF logits warping at all."""
    ids = prompt["input_ids"].expand(rows, -1).contiguous()
    mask = prompt["attention_mask"].expand(rows, -1).contiguous()
    past = None
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            if past is None:
                inputs = {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "pixel_values": prompt["pixel_values"].repeat(rows, 1),
                    "image_grid_thw": prompt["image_grid_thw"].repeat(rows, 1),
                    "use_cache": True,
                    "return_dict": True,
                    "logits_to_keep": 1,
                }
            else:
                inputs = {
                    "input_ids": ids[:, -1:],
                    "attention_mask": torch.cat([mask, torch.ones((rows, 1), device=ids.device, dtype=mask.dtype)], dim=1),
                    "past_key_values": past,
                    "use_cache": True,
                    "return_dict": True,
                    "logits_to_keep": 1,
                }
                mask = inputs["attention_mask"]
            out = model(**inputs)
            past = out.past_key_values
            next_id = out.logits[:, -1:, :].argmax(-1)
            ids = torch.cat([ids, next_id], dim=1)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--steps", type=int, default=64, help="new tokens per run")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=2 * 32 * 32, max_pixels=16384 * 32 * 32
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    print(f"loading {args.model}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device)
    model.eval()
    model.config.use_cache = True

    example = VEP50KDataset(args.data, args.video_root)[0]
    messages = student_messages(example["observed_frames"], example["question"])
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[rendered], images=list(example["observed_frames"]), return_tensors="pt")
    prompt = {key: value.to(args.device) for key, value in encoded.items()}
    prompt_len = int(prompt["input_ids"].shape[1])
    print(f"prompt tokens: {prompt_len}, new tokens per run: {args.steps}", flush=True)

    rows = [
        ("greedy, 1 row", dict(do_sample=False, num_return_sequences=1, max_new_tokens=args.steps)),
        (
            "sampled, 4 rows (current)",
            dict(
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=4,
                max_new_tokens=args.steps,
            ),
        ),
        (
            "sampled, 1 row",
            dict(
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=1,
                max_new_tokens=args.steps,
            ),
        ),
        (
            "sampled, 4 rows, no warpers",
            dict(do_sample=True, num_return_sequences=4, max_new_tokens=args.steps),
        ),
    ]
    print(f"\n{'setting':30s} {'total s':>9s} {'ms/step':>9s}")
    for name, kwargs in rows:
        # two warm runs, report the second (first pays for allocator growth)
        time_generate(model, processor, prompt, **kwargs)
        elapsed = time_generate(model, processor, prompt, **kwargs)
        print(f"{name:30s} {elapsed:9.2f} {1000 * elapsed / args.steps:9.1f}", flush=True)

    rows_warm = 1
    manual_decode(model, prompt, 8, rows_warm)
    elapsed = manual_decode(model, prompt, args.steps, rows_warm)
    print(f"{'manual decode, 1 row':30s} {elapsed:9.2f} {1000 * elapsed / args.steps:9.1f}")
    elapsed4 = manual_decode(model, prompt, args.steps, 4)
    print(f"{'manual decode, 4 rows':30s} {elapsed4:9.2f} {1000 * elapsed4 / args.steps:9.1f}")

    # Warper cost alone, on a realistic [4, 151936] logits slice.
    from transformers.generation.logits_process import TopPLogitsWarper

    logits = torch.randn(4, model.config.text_config.vocab_size, device=args.device)
    warper = TopPLogitsWarper(0.95, min_tokens_to_keep=1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(200):
        warper(torch.arange(4, device=args.device), logits.clone())
    torch.cuda.synchronize()
    per_call = (time.perf_counter() - started) / 200
    print(f"\nTopPLogitsWarper alone: {1000 * per_call:.1f} ms per call (4 rows, vocab {logits.shape[1]})")


if __name__ == "__main__":
    main()
