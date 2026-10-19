from __future__ import annotations

import json
import os
from pathlib import Path


class RolloutWriter:
    def __init__(self, output_dir: str):
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        artifact_dir = Path(output_dir) / "rollouts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.path = artifact_dir / f"candidates-rank{rank:02d}.jsonl"

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

