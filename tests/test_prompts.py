import unittest

from vep_opsd.config import HindsightOPSDConfig
from vep_opsd.prompts import extract_final_answer, is_degenerate_answer


class DegenerateAnswerTest(unittest.TestCase):
    """The collapsed run's failure mode: non-answers that the verifier scored as Yes."""

    def test_refusals_seen_in_the_collapsed_run_are_degenerate(self):
        for text in (
            "Cannot be determined from the given frames.",
            "Cannot be determined from the frames.",
            "Cannot be determined from the observed frames.",
            "Cannot be determined from the frames provided.",
            "cannot be determined",
            "It is impossible to determine from the observed frames.",
            "There is not enough information to tell.",
            "The future cannot be inferred from a single frame.",
        ):
            self.assertTrue(is_degenerate_answer(text), msg=text)

    def test_concrete_predictions_are_not_degenerate(self):
        for text in (
            "The group of people will start swimming in the water.",
            "The snowboarder will land the trick successfully.",
            "The soccer player will score a goal past the goalkeeper.",
            "The man crosses the street before the car reaches him.",
        ):
            self.assertFalse(is_degenerate_answer(text), msg=text)

    def test_refusals_that_escaped_the_phrase_filter(self):
        """A resumed collapsed run emitted these; the lead-verb rule catches them."""
        for text in (
            "Cannot be known from frames.",
            "Cannot be stated.",
            "Cannot be stated from frames.",
            "Cannot be answered from frames.",
            "Cannot be known from the frames.",
            "Unclear what happens next in this clip.",
            "Uncertain outcome for the player.",
        ):
            self.assertTrue(is_degenerate_answer(text), msg=text)

    def test_predictions_starting_with_a_modal_are_kept(self):
        for text in (
            "The car will stop before the crossing.",
            "It is possible the man turns left at the corner.",
            "Not long after, the dog runs out of the frame.",
        ):
            self.assertFalse(is_degenerate_answer(text), msg=text)

    def test_empty_and_too_short_answers_are_degenerate(self):
        for text in ("", "   ", "unknown", "yes"):
            self.assertTrue(is_degenerate_answer(text), msg=repr(text))

    def test_extract_final_answer_feeds_the_filter(self):
        text = "<reason>the ball is in the air</reason><answer>Cannot be determined from frames.</answer>"
        self.assertTrue(is_degenerate_answer(extract_final_answer(text)))
        text = "<reason>the ball is in the air</reason><answer>The kick goes through the posts.</answer>"
        self.assertFalse(is_degenerate_answer(extract_final_answer(text)))


class ConfigTest(unittest.TestCase):
    def test_new_knobs_validate(self):
        HindsightOPSDConfig().validate()
        with self.assertRaises(ValueError):
            HindsightOPSDConfig(minimum_teacher_spread=1.5).validate()
        with self.assertRaises(ValueError):
            HindsightOPSDConfig(max_resample_rounds=-1).validate()

    def test_candidate_count_stays_fixed(self):
        with self.assertRaises(ValueError):
            HindsightOPSDConfig(candidates_per_example=3).validate()


if __name__ == "__main__":
    unittest.main()
