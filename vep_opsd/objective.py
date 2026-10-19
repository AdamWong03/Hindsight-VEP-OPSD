from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BinaryTokenSpec:
    yes_id: int
    no_id: int
    yes_text: str
    no_text: str

    def to_dict(self):
        return {
            "yes_id": self.yes_id,
            "no_id": self.no_id,
            "yes_text": self.yes_text,
            "no_text": self.no_text,
        }


def resolve_yes_no_tokens(tokenizer, rendered_verifier_prefix: str) -> BinaryTokenSpec:
    """Resolve one-token Yes/No continuations at the exact scoring boundary."""

    prefix_ids = tokenizer.encode(rendered_verifier_prefix, add_special_tokens=False)
    for yes_text, no_text in (("Yes", "No"), (" Yes", " No"), ("\nYes", "\nNo")):
        yes_full = tokenizer.encode(rendered_verifier_prefix + yes_text, add_special_tokens=False)
        no_full = tokenizer.encode(rendered_verifier_prefix + no_text, add_special_tokens=False)
        if (
            yes_full[: len(prefix_ids)] == prefix_ids
            and no_full[: len(prefix_ids)] == prefix_ids
            and len(yes_full) == len(prefix_ids) + 1
            and len(no_full) == len(prefix_ids) + 1
            and yes_full[-1] != no_full[-1]
        ):
            return BinaryTokenSpec(yes_full[-1], no_full[-1], yes_text, no_text)
    raise ValueError(
        "The active tokenizer does not provide one-token Yes/No continuations at "
        "the verifier scoring boundary"
    )


def binary_yes_probability(next_token_logits: torch.Tensor, spec: BinaryTokenSpec) -> torch.Tensor:
    if next_token_logits.ndim != 2:
        raise ValueError("next_token_logits must have shape [batch, vocab]")
    pair = next_token_logits[:, [spec.yes_id, spec.no_id]].float()
    return F.softmax(pair, dim=-1)[:, 0]


def teacher_preference(
    consistency_scores: torch.Tensor,
    temperature: float,
    minimum_consistency: float,
) -> tuple[torch.Tensor, bool]:
    if consistency_scores.shape != (4,):
        raise ValueError("VEP-OPSD v1 requires exactly four candidate scores")
    if not torch.isfinite(consistency_scores).all():
        raise ValueError("Teacher consistency scores must be finite")
    if (consistency_scores < 0).any() or (consistency_scores > 1).any():
        raise ValueError("Teacher consistency scores must be probabilities")
    skipped = bool((consistency_scores < minimum_consistency).all().item())
    preference = F.softmax(consistency_scores.float() / temperature, dim=0).detach()
    return preference, skipped


def length_normalized_log_probability(
    logits: torch.Tensor,
    prompt_length: int,
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """Mean log probability of generated tokens only."""

    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("logits must have shape [1, sequence, vocab]")
    if generated_ids.ndim != 1 or generated_ids.numel() == 0:
        raise ValueError("generated_ids must be a non-empty 1D tensor")
    end = prompt_length + generated_ids.numel() - 1
    token_logits = logits[0, prompt_length - 1 : end, :]
    if token_logits.shape[0] != generated_ids.numel():
        raise ValueError("Generated-token/logit alignment failed")
    token_logp = F.log_softmax(token_logits.float(), dim=-1).gather(
        -1, generated_ids.to(token_logits.device).unsqueeze(-1)
    ).squeeze(-1)
    return token_logp.mean()


def listwise_cross_entropy(
    teacher_distribution: torch.Tensor,
    student_scores: torch.Tensor,
    student_temperature: float,
) -> torch.Tensor:
    if teacher_distribution.shape != (4,) or student_scores.shape != (4,):
        raise ValueError("VEP-OPSD v1 listwise loss expects four candidates")
    if teacher_distribution.requires_grad:
        raise ValueError("Teacher preferences must be detached")
    log_student = F.log_softmax(student_scores.float() / student_temperature, dim=0)
    return -(teacher_distribution.to(log_student.device) * log_student).sum()


def truncate_generated_ids(
    token_ids: Sequence[int], eos_token_ids: set[int], pad_token_id: int | None
) -> list[int]:
    result: list[int] = []
    for token_id in token_ids:
        token_id = int(token_id)
        if pad_token_id is not None and token_id == pad_token_id and token_id not in eos_token_ids:
            break
        result.append(token_id)
        if token_id in eos_token_ids:
            break
    if not result:
        raise ValueError("Student generated an empty trajectory")
    return result

