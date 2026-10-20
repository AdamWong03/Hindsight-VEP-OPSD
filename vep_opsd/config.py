from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HindsightOPSDConfig:
    """Algorithm constants for the single frozen v1 recipe."""

    candidates_per_example: int = 4
    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.95
    # Scores span the full [0, 1]: the verifier accepts the reference answer
    # (mean P(Yes) 0.93, 11/12 samples) and rejects wrong candidates (~0, 88% of
    # sampled candidates). At 0.2 a single accepted candidate gives
    # [0.97, 0.01, 0.01, 0.01] -- a one-hot target, which turns the listwise
    # objective into best-of-4 imitation. 0.5 keeps the ranking graded
    # ([0.78, 0.07, 0.07, 0.07]); 1.0 is the flattest setting measured.
    teacher_temperature: float = 0.5
    student_temperature: float = 1.0
    minimum_consistency: float = 0.6
    max_new_tokens: int = 512
    # A teacher whose four scores are nearly identical ranks nothing: the listwise
    # target is uniform and the loss sits at its ln(M) floor with ~zero gradient.
    # Such a group is skipped instead of being trained on.
    minimum_teacher_spread: float = 0.05
    # Candidates that do not commit to a prediction are discarded and resampled;
    # a group left with fewer than M usable candidates is skipped.
    reject_degenerate_candidates: bool = True
    max_resample_rounds: int = 1

    def validate(self) -> None:
        if self.candidates_per_example != 4:
            raise ValueError("Hindsight VEP-OPSD v1 fixes M=4")
        if self.sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive")
        if not 0 < self.sampling_top_p <= 1:
            raise ValueError("sampling_top_p must be in (0, 1]")
        if self.teacher_temperature <= 0 or self.student_temperature <= 0:
            raise ValueError("teacher/student temperatures must be positive")
        if not 0 <= self.minimum_consistency <= 1:
            raise ValueError("minimum_consistency must be in [0, 1]")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0 <= self.minimum_teacher_spread <= 1:
            raise ValueError("minimum_teacher_spread must be in [0, 1]")
        if self.max_resample_rounds < 0:
            raise ValueError("max_resample_rounds must be >= 0")

    def to_dict(self):
        self.validate()
        return asdict(self)

