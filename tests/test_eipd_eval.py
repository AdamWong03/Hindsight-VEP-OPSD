"""Answer-matching helpers for latent-only VEP evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_opsd.eipd_eval import normalise, token_f1


def test_normalise_ignores_case_punctuation_and_spacing():
    assert normalise("A Person, sits  DOWN!") == "a person sits down"
    assert normalise("  ") == ""


def test_exact_match_rewards_the_gold_answer():
    assert token_f1("a person sits down", "A person sits down.") == 1.0


def test_partial_overlap_is_between_zero_and_one():
    score = token_f1("person sits", "a person sits down")
    assert 0.0 < score < 1.0, score


def test_disjoint_answers_score_zero():
    assert token_f1("car drives away", "a person sits down") == 0.0


def test_empty_prediction_matches_only_an_empty_gold():
    assert token_f1("", "") == 1.0
    assert token_f1("", "a person sits down") == 0.0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL  {name}: {error}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
