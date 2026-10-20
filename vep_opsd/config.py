from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VEPPrivilegedOPSDConfig:
    """Algorithm constants for hindsight-privileged token-level OPSD."""

    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.95
    max_new_tokens: int = 512
    distillation_top_k: int = 100
    distillation_temperature: float = 1.0
    jsd_beta: float = 0.5
    add_tail_bucket: bool = True

    def validate(self) -> None:
        if self.sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive")
        if not 0 < self.sampling_top_p <= 1:
            raise ValueError("sampling_top_p must be in (0, 1]")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.distillation_top_k <= 1:
            raise ValueError("distillation_top_k must be greater than one")
        if self.distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be positive")
        if not 0 < self.jsd_beta < 1:
            raise ValueError("jsd_beta must be strictly between zero and one")

    def to_dict(self):
        self.validate()
        return asdict(self)
