#!/usr/bin/env python
"""Measure the shape of the hindsight teacher signal on real rollouts.

The collapse fix makes the verifier reject non-answers, but the target should
still be *graded*: a slightly wrong prediction has to score above a vague one,
otherwise the temperature-scaled listwise target becomes one-hot and the
objective degrades into best-of-4 imitation.

This samples M candidates per example with the current student prompt, scores
each with the current verifier, and reports:

  * the per-group score profile (how many candidates clear 0.5 / 0.6),
  * the teacher preference entropy after the configured temperature,
  * the skip rates the trainer would apply (all below the consistency floor,
    or a spread under `minimum_teacher_spread`),
  * how often the reference answer itself would be accepted.

Usage:
    python scripts/check_teacher_signal.py --model DIR --data MANIFEST --samples 12
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_opsd.config import HindsightOPSDConfig  # noqa: E402
from vep_opsd.data import VEP50KDataset  # noqa: E402
from vep_opsd.objective import binary_yes_probability, resolve_yes_no_tokens  # noqa: E402
from vep_opsd.prompts import (  # noqa: E402
    is_degenerate_answer,
    student_messages,
    verifier_messages,
)


def entropy(probs) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    config = HindsightOPSDConfig()
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

    dataset = VEP50KDataset(args.data, args.video_root)
    spec = None
    rows = []

    def score(messages, images):
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = processor(text=[rendered], images=list(images), return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in encoded.items()}
        inputs["logits_to_keep"] = 1
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False, return_dict=True)
        return float(binary_yes_probability(outputs.logits[:, -1, :], spec)[0].item())

    for index in range(min(args.samples, len(dataset))):
        example = dataset[index]
        messages = student_messages(example["observed_frames"], example["question"])
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = processor(text=[rendered], images=list(example["observed_frames"]), return_tensors="pt")
        prompt = {key: value.to(args.device) for key, value in encoded.items()}
        prompt_len = int(prompt["input_ids"].shape[1])
        with torch.no_grad():
            sequences = model.generate(
                **prompt,
                do_sample=True,
                temperature=config.sampling_temperature,
                top_p=config.sampling_top_p,
                num_return_sequences=args.candidates,
                max_new_tokens=config.max_new_tokens,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        texts = []
        for sequence in sequences:
            text = processor.tokenizer.decode(
                sequence[prompt_len:].tolist(), skip_special_tokens=False
            ).strip()
            texts.append(text)
        answers = []
        import re

        for text in texts:
            match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
            answers.append(match.group(1).strip() if match else "")

        images = list(example["observed_frames"]) + list(example["future_frames"])
        scores = []
        for text in texts:
            candidate_messages = verifier_messages(
                example["observed_frames"], example["future_frames"], example["question"], text
            )
            if spec is None:
                rendered_candidate = processor.apply_chat_template(
                    candidate_messages, tokenize=False, add_generation_prompt=True
                )
                spec = resolve_yes_no_tokens(processor.tokenizer, rendered_candidate)
            scores.append(score(candidate_messages, images))
        reference_score = score(
            verifier_messages(
                example["observed_frames"],
                example["future_frames"],
                example["question"],
                f"<answer>{example['reference_answer']}</answer>",
            ),
            images,
        )

        degenerate = sum(1 for answer in answers if is_degenerate_answer(answer))
        spread = max(scores) - min(scores)
        probs = torch.softmax(torch.tensor(scores) / config.teacher_temperature, dim=0).tolist()
        rows.append(
            {
                "scores": scores,
                "reference": reference_score,
                "degenerate": degenerate,
                "spread": spread,
                "entropy": entropy(probs),
                "above_half": sum(1 for s in scores if s > 0.5),
            }
        )
        print(
            f"sample {index:2d}: scores {[round(s, 3) for s in scores]} ref {reference_score:.3f} "
            f"degenerate {degenerate}/{len(answers)} spread {spread:.3f} entropy {entropy(probs):.3f}",
            flush=True,
        )

    print(f"\n== {len(rows)} groups")
    all_scores = [s for row in rows for s in row["scores"]]
    print(f"candidate P(Yes): mean {st.mean(all_scores):.3f} median {st.median(all_scores):.3f} "
          f">0.5: {100 * sum(1 for s in all_scores if s > 0.5) / len(all_scores):.0f}%")
    print(f"reference P(Yes): mean {st.mean([row['reference'] for row in rows]):.3f} "
          f"(accepted >0.6: {sum(1 for row in rows if row['reference'] > 0.6)}/{len(rows)})")
    print("groups by number of candidates above 0.5: "
          + ", ".join(f"{k}:{sum(1 for row in rows if row['above_half'] == k)}" for k in range(args.candidates + 1)))
    print(f"teacher entropy after T={config.teacher_temperature}: mean {st.mean([row['entropy'] for row in rows]):.3f} "
          f"(one-hot = 0, uniform = {math.log(args.candidates):.3f})")
    print(f"skip: all-below-floor {sum(1 for row in rows if all(s < config.minimum_consistency for s in row['scores']))}"
          f"/{len(rows)}, flat-teacher {sum(1 for row in rows if row['spread'] < config.minimum_teacher_spread)}/{len(rows)}, "
          f"degenerate {sum(1 for row in rows if row['degenerate'] > 0)}/{len(rows)}")


if __name__ == "__main__":
    main()
