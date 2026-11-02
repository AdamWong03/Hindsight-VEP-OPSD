#!/usr/bin/env python
"""Break the token-level distillation loss down by region and by token.

A mean JSD over a whole trajectory hides where the loss lives: the boilerplate
(`<reason>`, `</reason>`, `<answer>`, `</answer>`, common words) is nearly
deterministic for both sides, so it contributes almost nothing, while a handful
of content positions can sit at the ceiling. This script reproduces one training
step's two forwards for a few examples and reports

  * mean JSD per region (format / reasoning / answer),
  * the tokens with the largest JSD, and
  * whether the Student's own rollout mentions privileged evidence it cannot see
    ("the future frames show ..."), which is what distillation from a
    future-conditioned Teacher tends to teach.

Usage:
    python scripts/audit_token_jsd.py --adapter OUT/checkpoint-300 --data MANIFEST
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_opsd.config import VEPPrivilegedOPSDConfig  # noqa: E402
from vep_opsd.data import VEP50KDataset  # noqa: E402
from vep_opsd.objective import (  # noqa: E402
    generated_token_logits,
    generalized_jsd_from_log_probs,
    sampled_token_log_probs,
    student_distribution_on_teacher_support,
    teacher_topk_distribution,
)
from vep_opsd.prompts import student_messages  # noqa: E402
from vep_opsd.trainer import (  # noqa: E402
    _append_trajectory,
    _model_inputs,
    privileged_teacher_messages,
)


def token_regions(tokens: list[str]) -> list[str]:
    """Classify each token by the span it falls into, in decoded text coordinates.

    Token indices and character offsets are different spaces; walking the decoded
    pieces keeps them aligned.
    """
    text = "".join(tokens)
    answer_open = text.find("<answer>")
    answer_close = text.find("</answer>")
    markers = []
    for marker in ("<reason>", "</reason>", "<answer>", "</answer>"):
        start = text.find(marker)
        while start != -1:
            markers.append((start, start + len(marker)))
            start = text.find(marker, start + 1)
    regions = []
    position = 0
    for token in tokens:
        start, end = position, position + len(token)
        position = end
        if any(start < m_end and end > m_start for m_start, m_end in markers):
            regions.append("format")
        elif answer_open != -1 and answer_close != -1 and start >= answer_open and end <= answer_close + len("</answer>"):
            regions.append("answer")
        else:
            regions.append("reasoning")
    return regions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="Student LoRA checkpoint")
    parser.add_argument("--base", default=None, help="base model (default: adapter_config)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import json

    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    adapter = Path(args.adapter)
    base = args.base or json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    config = VEPPrivilegedOPSDConfig()

    processor = AutoProcessor.from_pretrained(base, min_pixels=2 * 32 * 32, max_pixels=16384 * 32 * 32)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    print(f"loading {base} + {adapter}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    model.config.use_cache = True

    dataset = VEP50KDataset(args.data, args.video_root)
    device = torch.device(args.device)
    totals = {"format": [], "reasoning": [], "answer": []}
    leak = 0

    for index in range(min(args.samples, len(dataset))):
        example = dataset[index]
        messages = student_messages(example["observed_frames"], example["question"])
        prompt = _model_inputs(processor, messages, example["observed_frames"], device)
        with torch.no_grad():
            sequences = model.generate(
                **prompt,
                do_sample=True,
                temperature=config.sampling_temperature,
                top_p=config.sampling_top_p,
                max_new_tokens=config.max_new_tokens,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        generated_ids = sequences[0][prompt["input_ids"].shape[1] :].tolist()
        eos = processor.tokenizer.eos_token_id
        if eos in generated_ids:
            generated_ids = generated_ids[: generated_ids.index(eos) + 1]
        text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        if re.search(r"future frame|next frame|ground-truth|reference answer", text, re.I):
            leak += 1

        # Teacher forward (adapter disabled) on the privileged prompt.
        images = list(example["observed_frames"]) + list(example["future_frames"])
        teacher_prompt = _model_inputs(
            processor,
            privileged_teacher_messages(
                example["observed_frames"],
                example["future_frames"],
                example["question"],
                example["reference_answer"],
            ),
            images,
            device,
        )
        teacher_inputs, generated = _append_trajectory(teacher_prompt, generated_ids, device)
        with torch.no_grad(), model.disable_adapter():
            teacher_out = model(**teacher_inputs, use_cache=False, return_dict=True)
        teacher_logits = generated_token_logits(teacher_out.logits, generated.numel())
        target = teacher_topk_distribution(
            teacher_logits,
            top_k=config.distillation_top_k,
            temperature=config.distillation_temperature,
            add_tail_bucket=config.add_tail_bucket,
        )

        student_inputs, _ = _append_trajectory(prompt, generated_ids, device)
        with torch.no_grad():
            student_out = model(**student_inputs, use_cache=False, return_dict=True)
        student_logits = generated_token_logits(student_out.logits, generated.numel())
        student_log_probs = student_distribution_on_teacher_support(
            student_logits,
            target,
            temperature=config.distillation_temperature,
            add_tail_bucket=config.add_tail_bucket,
        )
        _, per_token = generalized_jsd_from_log_probs(
            target.log_probs, student_log_probs, beta=config.jsd_beta,
            temperature=config.distillation_temperature,
        )

        jsd = per_token.float().cpu().tolist()
        tokens = [processor.tokenizer.decode([token]) for token in generated_ids]
        for position, (value, region) in enumerate(zip(jsd, token_regions(tokens))):
            totals[region].append(value)
        worst = sorted(range(len(jsd)), key=lambda i: -jsd[i])[:6]
        print(f"\nsample {index}: len={len(generated_ids)} mean_jsd={sum(jsd)/len(jsd):.4f}")
        print("  worst positions: " + ", ".join(f"{tokens[i]!r}={jsd[i]:.3f}" for i in worst))
        print(f"  answer: {example['reference_answer'][:70]!r}")
        print(f"       -> {re.search(r'<answer>(.*?)</answer>', text, re.S) and re.search(r'<answer>(.*?)</answer>', text, re.S).group(1)[:70]!r}")

    print("\n== mean JSD by region")
    for region, values in totals.items():
        if values:
            print(f"  {region:10s} {sum(values)/len(values):.4f}  over {len(values)} tokens")
    print(f"\nrollouts mentioning privileged evidence: {leak}/{min(args.samples, len(dataset))}")


if __name__ == "__main__":
    main()
