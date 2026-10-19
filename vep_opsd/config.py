from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HindsightOPSDConfig:
    """Algorithm constants for the single frozen v1 recipe."""

    candidates_per_example: int = 4
    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.95
    teacher_temperature: float = 0.2
    student_temperature: float = 1.0
    minimum_consistency: float = 0.6
    max_new_tokens: int = 512

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

    def to_dict(self):
        self.validate()
        return asdict(self)

