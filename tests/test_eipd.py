import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vep_opsd.eipd_data import uniform_pool, parse_answer, normalize_row, validate_meta, build
from vep_opsd.eipd_objective import eipd_loss
from vep_opsd.eipd_replay import answer_prediction_rows

class EIPDTest(unittest.TestCase):
    def test_twitter_eight_frame_pool(self):
        self.assertEqual(len(uniform_pool(80)), 8)
        self.assertTrue(all(a <= b for a, b in zip(uniform_pool(80), uniform_pool(80)[1:])))
        self.assertEqual(uniform_pool(80)[0], 1)
        self.assertEqual(uniform_pool(80)[-1], 78)

    def test_raw_twitter_row(self):
        row = normalize_row(dict(video="abc_1.mp4", question="frame_1 <image> What happens next?",
                 answer="Frame_3 <rimage> Someone jumps. <ans>They jump.</ans>",
                 question_images_index=[1], reasoning_images_index=[3]))
        self.assertIsNone(validate_meta(row))
        self.assertEqual(row["answer"], "They jump.")
        self.assertEqual(parse_answer("not a gold answer"), "")

    def test_temporal_leakage_guard(self):
        row = dict(video="a.mp4", question="next?", answer="open", image=[4],
                   reasoning_image=[3])
        self.assertEqual(validate_meta(row), "temporal_order_violation")

    def test_objective_different_teacher_has_gradient(self):
        s = torch.randn(4, 13, requires_grad=True)
        p0 = torch.randn(4, 13, requires_grad=True)
        p1 = torch.randn(4, 13, requires_grad=True) + 0.2
        result = eipd_loss(s, p0, p1)
        result.loss.backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertIsNone(p0.grad)
        self.assertIsNone(p1.grad)
        self.assertAlmostEqual(float(result.normalized_weights.sum()), 4.0, places=4)

    def test_zero_increment_has_no_distillation(self):
        s = torch.randn(4, 13, requires_grad=True)
        teacher = torch.randn(4, 13)
        result = eipd_loss(s, teacher, teacher)
        self.assertEqual(float(result.loss), 0.0)
        self.assertEqual(result.active_tokens, 0)

    def test_answer_rows_use_shifted_logits(self):
        logits = torch.arange(12 * 3).view(1, 12, 3)
        result = answer_prediction_rows(logits, prompt_tokens=4, reasoning_tokens=3, answer_tokens=3)
        self.assertEqual(result[:, 0].tolist(), [18, 21, 24])

    def test_build_without_using_model_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            line = dict(video="sample.mp4", question="Next?", answer="<ans>run</ans>",
                        question_images_index=[1], reasoning_images_index=[5])
            p.write_text(json.dumps(line) + "\n", encoding="utf-8")
            args = SimpleNamespace(input=str(p), hf_dataset=None, video_root=tmp,
                         video_map=None, output_dir=str(Path(tmp) / "out"),
                         exclude_video_lists=[], categories=[], train_size=1, val_size=1,
                         oversample=4, val_fraction=0.5, seed=42, strict_counts=False)
            with patch("vep_opsd.eipd_data.inspect_video", return_value=dict(
                     video_path="x.mp4", observed_raw_frames=[1], future_raw_frames=[9],
                     bridge_raw_frame=5, video_length=10, fps=1)):
                result = build(args)
            self.assertEqual(result["train"] + result["val"], 1)

if __name__ == "__main__":
    unittest.main()
