"""CPU tests of source-level split, metadata construction, and exact native-frame caching."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from vep_opsd.eipd_final_data import source_id, _split, plan, extract_frames, sha256_file


def example(video):
    return {
        "video": video,
        "question": "frame_1 <image> What happens next?",
        "answer": "Frame_4 <rimage> The runner moves. Final Answer: <ans>The runner moves.</ans>",
        "question_images_index": [1],
        "reasoning_images_index": [4],
        "meta_data": {"classification": "Predictive"},
    }


class FinalDataTests(unittest.TestCase):
    def test_source_group(self):
        self.assertEqual(source_id("abcdefghijk_9.mp4"), "abcdefghijk")
        self.assertEqual(source_id("abcdefghijk_10.mp4"), "abcdefghijk")
        self.assertEqual(_split("abcdefghijk", 42), _split(source_id("abcdefghijk_10.mp4"), 42))
        with self.assertRaises(ValueError):
            source_id("plain-video-name.mp4")

    def test_group_dedup_and_train_val_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = [f"vid{i:08d}" for i in range(120)]
            train_key = next(v for v in keys if _split(v, 42) == "train")
            val_key = next(v for v in keys if _split(v, 42) == "val")
            rows = [example(f"{train_key}_0.mp4"), example(f"{train_key}_2.mp4"),
                    example(f"{val_key}_5.mp4")]
            inp = root / "data.jsonl"
            inp.write_text("\n".join(map(json.dumps, rows)) + "\n")
            lists = []
            for i in range(3):
                path = root / f"exclude_{i}.txt"
                path.write_text(f"excluded{i}\n")
                lists.append(str(path))
            quotas = {"Predictive": 1, "Instructional": 0, "Camera": 0}
            cfg = SimpleNamespace(output=str(root/"final"), input=str(inp), hf_dataset=None,
                                  exclusions=lists, train_quotas=json.dumps(quotas),
                                  val_quotas=json.dumps(quotas), reserve_multiplier=2,
                                  seed=42, max_answer_chars=750, max_reason_chars=8000)
            result = plan(cfg)
            self.assertEqual(result["eligible_by_group"]["train/Predictive"], 1)
            self.assertEqual(result["eligible_by_group"]["val/Predictive"], 1)
            selected = [json.loads(x) for x in (root/"final"/"candidates.jsonl").read_text().splitlines()]
            self.assertEqual(len(selected), 2)
            self.assertEqual(len({x["source_id"] for x in selected}), 2)
            self.assertTrue((root/"final"/"candidate_source_ids.txt").is_file())

    def test_lossless_images_and_midpoint_chronology(self):
        class FakeReader:
            def __init__(self, path, num_threads=1): pass
            def __len__(self): return 60
            def get_avg_fps(self): return 30.0
            def get_batch(self, indices):
                class B:
                    def asnumpy(self):
                        return np.stack([np.full((8,8,3), i % 200, dtype=np.uint8)
                                         for i in indices])
                return B()
        class FakeDecord:
            VideoReader = FakeReader
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", {"decord": FakeDecord()}):
            root = Path(tmp)
            info = extract_frames({"image":[1],"reasoning_image":[4]},
                                  root/"fake.mp4", root, "png", 98)
            self.assertLess(max(info["past_native_indices"]), info["bridge_native_index"])
            self.assertLess(info["bridge_native_index"], min(info["future_native_indices"]))
            for name, digest in info["frame_sha256"].items():
                self.assertEqual(sha256_file(root/name), digest)
                with Image.open(root/name) as img:
                    self.assertEqual(img.size, (8,8))


if __name__ == "__main__":
    unittest.main()
