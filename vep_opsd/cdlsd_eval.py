"""Past-only CDLSD inference for short/long depth and latent interventions."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .cdlsd_data import CDLSDDataset, LATENT_TOKEN
from .cdlsd_latent import decode_answer, latent_rollout, sample_answer
from .eipd_eval import normalise, token_f1
from .eipd_latent import LatentProjector


def load_checkpoint(base_model: str, checkpoint: str, device, max_pixels: int):
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(
        checkpoint if (Path(checkpoint) / "preprocessor_config.json").is_file() else base_model,
        min_pixels=2 * 32 * 32,
        max_pixels=max_pixels,
    )
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": [LATENT_TOKEN]})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, checkpoint)
    payload = torch.load(
        Path(checkpoint) / "cdlsd_modules.pt", map_location="cpu", weights_only=False)
    projector = LatentProjector(model.config.text_config.hidden_size)
    projector.load_state_dict(payload["projector"])
    model.add_module("cdlsd_latent_projector", projector)
    model.to(device).eval()
    projector.to(dtype=model.get_input_embeddings().weight.dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, projector, processor


@torch.no_grad()
def evaluate_suite(model, projector, processor, dataset, indices, *, checkpoint: str,
                   device, max_new_tokens: int, seed: int) -> dict:
    """Evaluate short/long/zero/shuffle after one model load and one latent pass.

    Cached states stay on CPU between phases.  Besides eliminating three redundant
    8B checkpoint loads, this guarantees every condition sees byte-identical latent
    states and the same validation examples.
    """
    cache = []
    for offset, index in enumerate(indices):
        item = dataset[index]
        row = item["row"]
        states = latent_rollout(
            model, projector, processor, item["observed"], row["question"],
            short_steps=4, long_steps=8, checkpoint_prefill=False)
        cache.append({
            "row": row,
            "short": states.short.detach().cpu(),
            "long": states.long.detach().cpu(),
        })
        if (offset + 1) % 25 == 0:
            print(f"[cdlsd-eval] latent cache {offset + 1}/{len(indices)}", flush=True)

    donor_offsets = list(range(len(cache)))
    random.Random(seed).shuffle(donor_offsets)
    if len(cache) > 1:
        for offset, donor in enumerate(donor_offsets):
            if donor == offset:
                swap = (offset + 1) % len(cache)
                donor_offsets[offset], donor_offsets[swap] = (
                    donor_offsets[swap], donor_offsets[offset])

    conditions = ("short", "long", "zero", "shuffle")
    samples_by_condition = {}
    summaries = {}
    for condition in conditions:
        results = []
        for offset, entry in enumerate(cache):
            row = entry["row"]
            if condition == "short":
                selected = entry["short"]
            elif condition == "long":
                selected = entry["long"]
            elif condition == "zero":
                selected = torch.zeros_like(entry["short"])
            else:
                selected = cache[donor_offsets[offset]]["short"]
            generated = sample_answer(
                model, processor.tokenizer, row["question"], selected.to(device),
                max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
            prediction = decode_answer(processor.tokenizer, generated.cpu())
            results.append({
                "sample_id": row.get("sample_id"),
                "question": row["question"],
                "gold": row["answer"],
                "prediction": prediction,
                "exact": float(normalise(prediction) == normalise(row["answer"])),
                "token_f1": token_f1(prediction, row["answer"]),
            })
            if (offset + 1) % 25 == 0:
                print(f"[cdlsd-eval] {condition} {offset + 1}/{len(cache)}", flush=True)
        samples_by_condition[condition] = results
        summaries[condition] = {
            "exact_match": sum(row["exact"] for row in results) / len(results),
            "token_f1": sum(row["token_f1"] for row in results) / len(results),
        }

    question_echo = sum(
        token_f1(entry["row"]["question"], entry["row"]["answer"])
        for entry in cache
    ) / len(cache)
    comparisons = {}
    short = samples_by_condition["short"]
    for condition in ("long", "zero", "shuffle"):
        other = samples_by_condition[condition]
        comparisons[f"short_vs_{condition}"] = {
            "identical_predictions": sum(
                left["prediction"] == right["prediction"]
                for left, right in zip(short, other)
            ),
            "short_minus_other_f1": sum(
                left["token_f1"] - right["token_f1"]
                for left, right in zip(short, other)
            ) / len(short),
        }
    return {
        "checkpoint": checkpoint,
        "samples": len(cache),
        "summaries": summaries,
        "question_echo_f1": question_echo,
        "comparisons": comparisons,
        "conditions": samples_by_condition,
    }


@torch.no_grad()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", action="store_true",
                        help="one-load short/long/zero/shuffle diagnostic suite")
    parser.add_argument("--depth", choices=("short", "long"), default="short")
    parser.add_argument("--ablate", choices=("none", "zero", "shuffle"), default="none")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--image-max-pixels", type=int, default=2048 * 32 * 32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    root = Path(args.data_root).resolve()
    manifest = args.manifest or str(root / "val.jsonl")
    dataset = CDLSDDataset(str(root), manifest)
    indices = list(range(len(dataset)))
    if args.limit:
        indices = indices[:args.limit]
    donors = indices.copy()
    random.Random(args.seed).shuffle(donors)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, projector, processor = load_checkpoint(
        args.base_model, args.checkpoint, device, args.image_max_pixels)

    if args.suite:
        payload = evaluate_suite(
            model, projector, processor, dataset, indices,
            checkpoint=args.checkpoint, device=device,
            max_new_tokens=args.max_new_tokens, seed=args.seed)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        print(json.dumps({key: value for key, value in payload.items()
                          if key != "conditions"}, indent=2), flush=True)
        return 0

    results = []
    for offset, index in enumerate(indices):
        item = dataset[index]
        row = item["row"]
        states = latent_rollout(
            model, projector, processor, item["observed"], row["question"],
            short_steps=4, long_steps=8, checkpoint_prefill=False)
        selected = states.short if args.depth == "short" else states.long
        if args.ablate == "zero":
            selected = torch.zeros_like(selected)
        elif args.ablate == "shuffle":
            donor_index = donors[offset]
            if donor_index == index and len(donors) > 1:
                donor_index = donors[(offset + 1) % len(donors)]
            donor = dataset[donor_index]
            donor_states = latent_rollout(
                model, projector, processor, donor["observed"],
                donor["row"]["question"], short_steps=4, long_steps=8,
                checkpoint_prefill=False)
            selected = donor_states.short if args.depth == "short" else donor_states.long
        generated = sample_answer(
            model, processor.tokenizer, row["question"], selected,
            max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0)
        prediction = decode_answer(processor.tokenizer, generated.cpu())
        results.append({
            "sample_id": row.get("sample_id"),
            "question": row["question"],
            "gold": row["answer"],
            "prediction": prediction,
            "exact": float(normalise(prediction) == normalise(row["answer"])),
            "token_f1": token_f1(prediction, row["answer"]),
        })
        if (offset + 1) % 25 == 0:
            print(f"[cdlsd-eval] {offset + 1}/{len(indices)}", flush=True)
    summary = {
        "checkpoint": args.checkpoint,
        "depth": args.depth,
        "ablate": args.ablate,
        "samples": len(results),
        "exact_match": sum(row["exact"] for row in results) / len(results),
        "token_f1": sum(row["token_f1"] for row in results) / len(results),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "samples": results},
                                 indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
