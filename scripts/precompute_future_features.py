#!/usr/bin/env python
"""Precompute frozen Qwen3-VL vision features for every future frame.

The future-state target is

    v_j = L2Norm( MeanPool( phi_0(F_j) ) )

with `phi_0` the frozen Qwen3-VL vision tower (patch features **after** the
spatial merger, i.e. what the language model would actually receive). Because
the tower never changes during Stage 1/2, computing this once turns ~69k ViT
passes per epoch into a single offline pass and removes the future PNGs from
the training data path entirely.

Output layout (sharded, no tens-of-thousands of tiny files):

    EIPD/future_features/
      feature_config.json
      train/shard_000.pt ...
      val/shard_000.pt ...

Every shard is a dict: sample_ids[i] -> {"features": (m,d) float32, "native": [int]}
Usage:
    python scripts/precompute_future_features.py --eipd-root /data/EIPD \
        --model /models/Qwen3-VL-8B-Instruct --split train --shard-size 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def log(message: str) -> None:
    print(f"[future-features] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visual_weight_digest(visual: torch.nn.Module) -> str:
    """Content hash of the tower, so a frozen-target cache cannot silently drift."""
    digest = hashlib.sha256()
    for name, tensor in sorted(visual.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


def load_vision(model_dir: str, device: str):
    """Load the vision tower only; the 8B language model stays on CPU and is dropped."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map={"": "cpu"}, low_cpu_mem_usage=True)
    visual = model.model.visual
    del model
    visual = visual.to(device).eval()
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    return visual


@torch.no_grad()
def encode_frames(visual, image_processor, paths: list[Path], device: str,
                  batch: int = 8) -> torch.Tensor:
    """Mean-pooled, L2-normalised feature per frame; shape (len(paths), d)."""
    from PIL import Image

    outputs = []
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        images = []
        for path in chunk:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        batch_feature = image_processor(images=images, return_tensors="pt")
        pixel_values = batch_feature["pixel_values"].to(device)
        grid_thw = batch_feature["image_grid_thw"].to(device)
        merged, _deepstack = visual(hidden_states=pixel_values, grid_thw=grid_thw)
        # merged: (total_merged_tokens, d) for every image in the chunk
        counts = (grid_thw.prod(-1) // (visual.config.spatial_merge_size ** 2)).tolist()
        cursor = 0
        for count in counts:
            frame = merged[cursor:cursor + count].float().mean(dim=0)
            outputs.append(torch.nn.functional.normalize(frame, dim=-1))
            cursor += int(count)
    return torch.stack(outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eipd-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--manifest", default=None,
                        help="Override {split}.jsonl (e.g. a smoke subset)")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--image-max-pixels", type=int, default=2048 * 32 * 32)
    parser.add_argument("--image-min-pixels", type=int, default=2 * 32 * 32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="stop after N samples (debug)")
    parser.add_argument("--force", action="store_true", help="recompute existing shards")
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="torch CPU threads; 64 (default) is ~5x slower on preprocessing")
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=-1,
                        help="exclusive shard bound, so several processes can split the work")
    parser.add_argument("--vision-dump", default=None,
                        help="pre-saved vision tower, avoids loading the 8B model per worker")
    parser.add_argument("--dump-vision-to", default=None,
                        help="write the vision tower here and exit")
    args = parser.parse_args()

    root = Path(args.eipd_root).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else root / f"{args.split}.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"missing manifest: {manifest}")
    out_dir = root / "future_features" / (args.split if not args.manifest
                                          else f"custom_{manifest.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.cpu_threads))

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=args.image_min_pixels, max_pixels=args.image_max_pixels)
    image_processor = processor.image_processor
    if args.dump_vision_to:
        visual = load_vision(args.model, "cpu")
        torch.save({"state_dict": visual.state_dict(),
                    "vision_config": visual.config.to_dict()}, args.dump_vision_to)
        log(f"vision tower dumped to {args.dump_vision_to}")
        return 0
    if args.vision_dump and Path(args.vision_dump).is_file():
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig)
        payload = torch.load(args.vision_dump, map_location="cpu", weights_only=False)
        visual = Qwen3VLVisionModel(Qwen3VLVisionConfig(**payload["vision_config"]))
        visual.load_state_dict(payload["state_dict"])
        # The dump path must match `load_vision`'s bf16: fp32 here is 3.7x slower
        # (364 vs 98 ms/frame) and would produce different targets.
        visual = visual.to(args.device, torch.bfloat16).eval()
        for parameter in visual.parameters():
            parameter.requires_grad_(False)
        log(f"vision tower loaded from {args.vision_dump}")
    else:
        visual = load_vision(args.model, args.device)

    config = {
        "schema": "eipd-future-features-v1",
        "model_dir": str(Path(args.model).resolve()),
        "visual_weight_sha256": visual_weight_digest(visual),
        "extraction": "qwen3vl-vision-merger-meanpool-l2",
        "feature_dim": int(visual.config.out_hidden_size),
        "spatial_merge_size": int(visual.config.spatial_merge_size),
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "dtype": "float32",
        "normalisation": "l2",
        "pooling": "mean_over_merged_patch_tokens",
        "source": "post_merger_hidden_states",
    }
    # Provenance fields that differ between hosts (a resolved path is not a
    # property of the extraction) must not block a legitimate re-run.
    COMPARED = ("schema", "visual_weight_sha256", "extraction", "feature_dim",
                "spatial_merge_size", "image_max_pixels", "image_min_pixels", "dtype",
                "normalisation", "pooling", "source")
    config_path = out_dir.parent / "feature_config.json"
    if config_path.is_file() and not args.force:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        diff = {key: (previous.get(key), config.get(key)) for key in COMPARED
                if previous.get(key) != config.get(key)}
        if diff:
            details = "; ".join(f"{k}: cached={a!r} requested={b!r}"
                                for k, (a, b) in sorted(diff.items()))
            raise SystemExit(
                f"feature_config.json does not match this extraction ({details}). "
                "Pass --force to rebuild, otherwise Stage 1/2 would train on targets "
                "that are inconsistent with the cached ones.")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log(f"vision tower hash {config['visual_weight_sha256'][:16]}… dim={config['feature_dim']}")

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    log(f"{len(rows)} samples from {manifest.name}")

    pending = 0
    for shard_index, start in enumerate(range(0, len(rows), args.shard_size)):
        if shard_index < args.shard_start:
            continue
        if args.shard_end >= 0 and shard_index >= args.shard_end:
            break
        chunk = rows[start:start + args.shard_size]
        shard_path = out_dir / f"shard_{shard_index:03d}.pt"
        expected_ids = [row["sample_id"] for row in chunk]
        if shard_path.is_file() and not args.force:
            saved = torch.load(shard_path, map_location="cpu", weights_only=False)
            if saved.get("sample_ids") == expected_ids:
                log(f"shard {shard_index:03d} up to date ({len(chunk)} samples)")
                continue
        payload, skipped = {}, 0
        for row in chunk:
            paths = []
            for rel in row["future_files"]:
                path = (root / rel).resolve()
                if not path.is_file():
                    skipped += 1
                    paths = []
                    break
                paths.append(path)
            if not paths:
                continue
            payload[row["sample_id"]] = {
                "features": encode_frames(visual, image_processor, paths, args.device),
                "native": [int(x) for x in row["future_native_indices"]],
                "frame_sha256": [row["frame_sha256"][rel] for rel in row["future_files"]],
            }
        torch.save({"schema": config["schema"], "sample_ids": list(payload),
                    "features": payload}, shard_path)
        pending += 1
        log(f"shard {shard_index:03d}: {len(payload)} samples"
            + (f", {skipped} skipped (missing frames)" if skipped else ""))
    log(f"done: {pending} shard(s) written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
