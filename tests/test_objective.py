import unittest

import torch

from vep_opsd.objective import (
    BinaryTokenSpec,
    binary_yes_probability,
    length_normalized_log_probability,
    listwise_cross_entropy,
    teacher_preference,
    truncate_generated_ids,
)


class ObjectiveTest(unittest.TestCase):
    def test_binary_score_uses_only_yes_no(self):
        logits = torch.zeros(1, 8)
        logits[0, 2] = 2.0
        score = binary_yes_probability(logits, BinaryTokenSpec(2, 5, "Yes", "No"))
        self.assertAlmostEqual(float(score), 0.880797, places=5)

    def test_low_group_is_skipped(self):
        preference, skipped = teacher_preference(
            torch.tensor([0.1, 0.2, 0.3, 0.59]), 0.2, 0.6
        )
        self.assertTrue(skipped)
        self.assertAlmostEqual(float(preference.sum()), 1.0, places=6)

    def test_length_normalization_excludes_prompt(self):
        logits = torch.zeros(1, 5, 7)
        generated = torch.tensor([3, 4])
        logits[0, 2, 3] = 3.0
        logits[0, 3, 4] = 3.0
        score = length_normalized_log_probability(logits, 3, generated)
        expected = torch.log_softmax(torch.tensor([0.0, 0, 0, 3, 0, 0, 0]), 0)[3]
        self.assertTrue(torch.allclose(score, expected))

    def test_listwise_gradient_reaches_student_only(self):
        teacher = torch.tensor([0.7, 0.1, 0.1, 0.1])
        student = torch.zeros(4, requires_grad=True)
        listwise_cross_entropy(teacher, student, 1.0).backward()
        self.assertIsNotNone(student.grad)
        self.assertFalse(teacher.requires_grad)

    def test_truncate_keeps_eos(self):
        self.assertEqual(truncate_generated_ids([4, 5, 2, 0, 0], {2}, 0), [4, 5, 2])


if __name__ == "__main__":
    unittest.main()
