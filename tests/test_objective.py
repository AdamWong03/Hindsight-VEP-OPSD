import unittest

import torch

from vep_opsd.objective import (
    generalized_jsd_from_log_probs,
    generated_token_logits,
    student_distribution_on_teacher_support,
    teacher_topk_distribution,
    truncate_generated_ids,
)


class ObjectiveTest(unittest.TestCase):
    def test_topk_plus_tail_is_normalized(self):
        teacher = teacher_topk_distribution(torch.randn(4, 11), 3, 1.0, True)
        self.assertEqual(teacher.log_probs.shape, (4, 4))
        self.assertTrue(
            torch.allclose(
                teacher.log_probs.exp().sum(-1), torch.ones(4), atol=1e-5
            )
        )

    def test_jsd_gradient_reaches_student_only(self):
        teacher_logits = torch.randn(3, 9)
        target = teacher_topk_distribution(teacher_logits, 4, 1.0, True)
        student_logits = torch.randn(3, 9, requires_grad=True)
        student = student_distribution_on_teacher_support(
            student_logits, target, 1.0, True
        )
        loss, per_token = generalized_jsd_from_log_probs(
            target.log_probs, student, beta=0.5
        )
        loss.backward()
        self.assertEqual(per_token.shape, (3,))
        self.assertIsNotNone(student_logits.grad)
        self.assertFalse(target.log_probs.requires_grad)

    def test_generated_rows_exclude_final_unused_row(self):
        logits = torch.randn(1, 4, 7)
        self.assertTrue(torch.equal(generated_token_logits(logits, 3), logits[0, :3]))

    def test_truncate_keeps_eos(self):
        self.assertEqual(truncate_generated_ids([4, 5, 2, 0], {2}, 0), [4, 5, 2])


if __name__ == "__main__":
    unittest.main()
