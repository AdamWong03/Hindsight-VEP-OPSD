"""Data and native Qwen3-VL prompt construction for CDLSD-VEP.

The main method consumes only Past frames, Question and Gold Answer.  EIPD
manifests may contain Bridge/Future fields, but this module deliberately never
loads them.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


LATENT_TOKEN = "<|latent_start|>"
SYSTEM_PROMPT = (
    "Predict the subsequent event from the observed video. "
    "Use continuous latent computation before answering."
)


class CDLSDDataset(Dataset):
    """Past-only VEP records backed by an EIPD-style JSONL manifest."""

    def __init__(self, root: str, manifest: str):
        self.root = Path(root).resolve()
        self.rows = [
            json.loads(line)
            for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"empty manifest: {manifest}")
        required = {"question", "answer", "past_files"}
        for index, row in enumerate(self.rows):
            missing = required.difference(row)
            if missing:
                raise KeyError(f"row {index} misses {sorted(missing)}")
            if not row["past_files"]:
                raise ValueError(f"row {index} has no Past frame")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        row = dict(self.rows[index])
        images = []
        for relative in row["past_files"]:
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        return {"row": row, "observed": images}


def collate_single(items: list[dict]) -> dict:
    if len(items) != 1:
        raise ValueError("CDLSD currently requires per-device batch size 1")
    return items[0]


def encode_multimodal_prefill(backbone, processor, images, question: str, device):
    """Build the native Qwen3-VL multimodal prefill ending in LATENT_TOKEN.

    Keeping ``input_ids``, ``pixel_values`` and ``image_grid_thw`` intact lets
    Qwen3-VL perform its own visual-token replacement and DeepStack injection.
    """
    content = [{"type": "text", "text": "Observed frames in temporal order:"}]
    content.extend({"type": "image", "image": image} for image in images)
    content.append({"type": "text", "text": "Question: " + question})
    text = processor.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    ) + LATENT_TOKEN
    batch = processor(text=[text], images=images, return_tensors="pt")
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        positions, _ = backbone.get_rope_index(
            batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            attention_mask=batch.get("attention_mask"),
        )
    next_position = int(positions[:, :, -1].max().item()) + 1
    keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    return {key: batch[key] for key in keep if batch.get(key) is not None}, next_position
