import unittest

from PIL import Image

from vep_opsd.config import VEPPrivilegedOPSDConfig
from vep_opsd.prompts import (
    extract_final_answer,
    privileged_teacher_messages,
    student_messages,
)


class PromptTest(unittest.TestCase):
    def setUp(self):
        self.frames = [Image.new("RGB", (2, 2))]

    def test_student_never_receives_future_or_gold(self):
        messages = student_messages(self.frames, "What happens next?")
        text = str(messages)
        self.assertNotIn("reference", text.lower())
        self.assertNotIn("ground-truth future", text.lower())

    def test_teacher_receives_future_and_gold(self):
        messages = privileged_teacher_messages(
            self.frames, self.frames, "What happens next?", "The person stands up."
        )
        text = str(messages)
        self.assertIn("Ground-truth future frames", text)
        self.assertIn("The person stands up.", text)

    def test_extract_answer(self):
        self.assertEqual(
            extract_final_answer("<reason>x</reason><answer>The ball lands.</answer>"),
            "The ball lands.",
        )


class ConfigTest(unittest.TestCase):
    def test_defaults_validate(self):
        VEPPrivilegedOPSDConfig().validate()

    def test_invalid_jsd_beta_is_rejected(self):
        with self.assertRaises(ValueError):
            VEPPrivilegedOPSDConfig(jsd_beta=1.0).validate()


if __name__ == "__main__":
    unittest.main()
