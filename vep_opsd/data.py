from __future__ import annotations

import json
from pathlib import Path

import decord
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .prompts import clean_twiff_question


def _uniform_frame_pool(num_frames: int, video_length: int) -> list[int]:
    start_frame, end_frames = 1, 1
    if start_frame + 1 >= video_length:
        start_frame = 0
    if video_length - end_frames - 1 <= start_frame:
        end_index = video_length - 1
    else:
        end_index = video_length - end_frames - 1
    count = min(num_frames - 2, end_index - start_frame)
    intervals = np.linspace(start_frame + 1, end_index, count + 1).astype(int)
    middle = [
        (intervals[index] + intervals[index + 1] - 1) // 2
        for index in range(len(intervals) - 1)
    ]
    indices = [start_frame, *middle, end_index]
    return indices + [indices[-1]] * (num_frames - len(indices))


def read_frames_by_clip(video_path: str, clip_indices: list[int]) -> list[Image.Image]:
    """Read 1-based positions from the compact data's uniform eight-frame pool."""

    reader = decord.VideoReader(video_path, num_threads=1)
    pool = _uniform_frame_pool(8, len(reader))
    selected = [pool[index - 1] for index in clip_indices]
    raw = reader.get_batch(selected).asnumpy()
    return [Image.fromarray(frame) for frame in raw]


class VEP50KDataset(Dataset):
    """Observed/future view over a compact VEP/TwiFF JSON manifest."""

    def __init__(self, manifest: str, video_root: str | None = None):
        self.manifest = Path(manifest).resolve()
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.rows = json.load(handle)
        if not isinstance(self.rows, list) or not self.rows:
            raise ValueError("Training manifest must be a non-empty JSON array")
        self.video_root = Path(video_root).resolve() if video_root else self.manifest.parent

    def __len__(self):
        return len(self.rows)

    def _video_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.video_root / path

    def __getitem__(self, index):
        row = self.rows[index]
        observed_indices = [int(value) for value in row.get("image", [])]
        future_indices = [int(value) for value in row.get("reasoning_image", [])]
        sample_id = str(row.get("id", index))
        if not observed_indices or not future_indices:
            raise ValueError(f"Sample {sample_id} lacks observed/future frame indices")
        if min(observed_indices + future_indices) < 1 or max(observed_indices + future_indices) > 8:
            raise ValueError(f"Sample {sample_id} frame indices must be in [1, 8]")
        if max(observed_indices) >= min(future_indices):
            raise ValueError(f"Sample {sample_id} does not have ordered future frames")
        conversations = row.get("conversations") or []
        human = next(
            (turn.get("value", "") for turn in conversations if turn.get("from") == "human"),
            "",
        )
        question = clean_twiff_question(human)
        if not question:
            raise ValueError(f"Sample {sample_id} has no question")
        video_path = self._video_path(str(row.get("video", "")))
        if not video_path.is_file():
            raise FileNotFoundError(f"Training video is missing: {video_path}")
        reference_answer = str(row.get("answer", "")).strip()
        if not reference_answer:
            raise ValueError(f"Sample {sample_id} has no reference answer")
        return {
            "sample_id": sample_id,
            "question": question,
            "reference_answer": reference_answer,
            "observed_frames": read_frames_by_clip(str(video_path), observed_indices),
            "future_frames": read_frames_by_clip(str(video_path), future_indices),
            "observed_indices": observed_indices,
            "future_indices": future_indices,
            "video_path": str(video_path),
        }


def collate_examples(examples):
    return {"examples": examples}
