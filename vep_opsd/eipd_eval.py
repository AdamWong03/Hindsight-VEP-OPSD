"""Latent-only VEP inference: (Past, Q) -> Z_{1..K} -> answer.

Evaluation reads only what is available at test time — no Bridge, no future frames,
no gold answer prefix. The answer is decoded greedily through the same bottleneck the
model was trained with (Question + latents), so the number measures the deployed path.

Besides accuracy it reports two latent interventions, because an accurate answer does
not by itself prove the latents are used (a model can answer from the question alone):

    python -m vep_opsd.eipd_eval --checkpoint DIR --ablate zero
    python -m vep_opsd.eipd_eval --checkpoint DIR --ablate shuffle

`zero` replaces Z with zeros, `shuffle` swaps in another sample's Z. A large drop
means the answer really flows through the latent states.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .eipd_latent import inner_model, prefill_only, rollout_latents_shared
from .eipd_stage1 import LATENT_TOKENS, encode_prompt

EOS = ("<|im_end|>", "<|endoftext|>", "</answer>")


def log(message: str) -> None:
    print(f"[eval] {message}", flush=True)


# --------------------------------------------------------------------------- model
def load_model(checkpoint: str, device, base_model: str | None = None):
    from .eipd_stage2 import load_stage1_modules, restore_added_tokens
    from peft import PeftModel

    ckpt = Path(checkpoint)
    source = str(ckpt) if (ckpt / "config.json").is_file() else base_model
    if source is None:
        raise SystemExit("checkpoint has no weights; pass --base-model")

    processor = AutoProcessor.from_pretrained(
        str(ckpt) if (ckpt / "preprocessor_config.json").is_file() else source,
        min_pixels=2 * 32 * 32, max_pixels=4 * 1024 * 1024)
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": LATENT_TOKENS})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    payload = load_stage1_modules(str(ckpt), 0, device)
    hidden = payload["projector"]["residual.weight"].shape[0]
    model = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tokenizer))
    if (ckpt / "adapter_model.safetensors").is_file():
        model = PeftModel.from_pretrained(model, str(ckpt))
    restore_added_tokens(model, payload)

    from .eipd_latent import LatentProjector
    projector = LatentProjector(hidden)
    projector.load_state_dict(payload["projector"])
    model.add_module("latent_projector", projector)
    model.to(device).eval()
    # the model was moved, not cast: the projector must follow the dtype too
    projector.to(dtype=model.get_input_embeddings().weight.dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, projector, processor, tokenizer


# --------------------------------------------------------------------------- decode
@torch.no_grad()
def latent_states(model, projector, processor, row, eipd_root: Path, steps: int, device):
    from PIL import Image

    observed = []
    for relative in row["past_files"]:
        with Image.open(eipd_root / relative) as image:
            observed.append(image.convert("RGB").copy())
    batch, start = encode_prompt(inner_model(model), processor, observed,
                                 row["question"], None, device)
    prefill = {key: batch[key] for key in
               ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
               if batch.get(key) is not None}
    state = prefill_only(model, projector, prefill, start, checkpoint=False)
    return rollout_latents_shared(model, projector, [state], steps=steps)[0]   # (K, d)


@torch.no_grad()
def answer(model, tokenizer, question: str, latents: torch.Tensor, device,
           max_new_tokens: int = 48) -> str:
    """Greedy decode of [Q ; Z ; <answer>...] — the same context used in training."""
    embed = model.get_input_embeddings()
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    marker_ids = tokenizer("<answer>", add_special_tokens=False)["input_ids"]
    prompt = torch.tensor([question_ids + marker_ids], device=device)

    inputs = torch.cat([embed(prompt[:, :len(question_ids)]),
                        latents.to(device).view(1, -1, embed.weight.shape[1]),
                        embed(prompt[:, len(question_ids):])], dim=1)

    out = model(inputs_embeds=inputs, use_cache=True, return_dict=True)
    cache = out.past_key_values
    position = inputs.shape[1]
    token = out.logits[:, -1:].argmax(-1)
    produced: list[int] = []
    for _ in range(max_new_tokens):
        produced.append(int(token))
        text = tokenizer.decode(produced, skip_special_tokens=True)
        if any(marker in text for marker in EOS):
            break
        out = model(inputs_embeds=embed(token), past_key_values=cache,
                    cache_position=torch.tensor([position], device=device),
                    position_ids=torch.tensor([[position]], device=device).expand(3, 1, 1),
                    use_cache=True, return_dict=True)
        cache, position = out.past_key_values, position + 1
        token = out.logits[:, -1:].argmax(-1)
    return tokenizer.decode(produced, skip_special_tokens=True)


# --------------------------------------------------------------------------- metrics
def normalise(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return re.sub(r"\s+", " ", text)


def token_f1(prediction: str, gold: str) -> float:
    p, g = normalise(prediction).split(), normalise(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for token in p:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in g:
        if common.get(token, 0) > 0:
            common[token] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Stage 1 or Stage 2 output directory")
    parser.add_argument("--base-model", default=None,
                        help="only needed when the checkpoint stores an adapter")
    parser.add_argument("--eipd-root", default="/nfs/ofs-llm-ssd/user/shengrenren_i/data/EIPD")
    parser.add_argument("--manifest", default=None, help="defaults to <eipd-root>/val.jsonl")
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 = all samples")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--ablate", choices=("none", "zero", "shuffle"), default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="per-sample JSON, default <ckpt>/eval.json")
    args = parser.parse_args(argv)

    root = Path(args.eipd_root).resolve()
    manifest = Path(args.manifest) if args.manifest else root / "val.jsonl"
    rows = [json.loads(line) for line in
            manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("empty manifest")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, projector, processor, tokenizer = load_model(args.checkpoint, device,
                                                        args.base_model)
    log(f"loaded {args.checkpoint} | K={args.latent_steps} | ablate={args.ablate} "
        f"| {len(rows)} samples")

    # shuffle needs a donor pool measured independently of the evaluated sample
    donors = list(range(len(rows)))
    random.Random(args.seed).shuffle(donors)

    results, exact_hits, f1_total = [], 0, 0.0
    cached: dict[int, torch.Tensor] = {}

    for index, row in enumerate(rows):
        latents = latent_states(model, projector, processor, row, root,
                                args.latent_steps, device)
        if args.ablate == "zero":
            latents = torch.zeros_like(latents)
        elif args.ablate == "shuffle":
            donor = donors[index]
            if donor == index:
                donor = donors[(index + 1) % len(donors)]
            latents = latent_states(model, projector, processor, rows[donor], root,
                                    args.latent_steps, device)

        prediction = answer(model, tokenizer, row["question"], latents, device,
                            args.max_new_tokens)
        exact = float(normalise(prediction) == normalise(row["answer"]))
        f1 = token_f1(prediction, row["answer"])
        exact_hits += exact
        f1_total += f1
        results.append({"sample_id": row["sample_id"], "question": row["question"],
                        "gold": row["answer"], "pred": prediction,
                        "exact": exact, "f1": f1})
        if (index + 1) % 25 == 0:
            log(f"{index + 1}/{len(rows)} exact={exact_hits / (index + 1):.3f} "
                f"f1={f1_total / (index + 1):.3f}")

    summary = {"checkpoint": str(args.checkpoint), "ablate": args.ablate,
               "latent_steps": args.latent_steps, "samples": len(results),
               "exact_match": exact_hits / len(results),
               "token_f1": f1_total / len(results)}
    log(f"exact_match={summary['exact_match']:.4f} token_f1={summary['token_f1']:.4f} "
        f"({len(results)} samples, ablate={args.ablate})")

    output = Path(args.output) if args.output else Path(args.checkpoint) / "eval.json"
    output.write_text(json.dumps({"summary": summary, "samples": results},
                                 indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
