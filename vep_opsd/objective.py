from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class TopKDistribution:
    """Teacher-defined support and normalized log probabilities per token."""

    indices: torch.Tensor
    log_probs: torch.Tensor


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


def generated_token_logits(logits: torch.Tensor, generated_length: int) -> torch.Tensor:
    """Select rows that predict generated tokens from a G+1 kept-logit window."""

    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("logits must have shape [1, sequence, vocab]")
    if generated_length <= 0 or logits.shape[1] < generated_length:
        raise ValueError("invalid generated length for logits")
    return logits[0, :generated_length, :]


def _append_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True).clamp(max=-1e-7)
    log_tail = torch.log(-torch.expm1(log_mass))
    return torch.cat([log_probs, log_tail], dim=-1)


def teacher_topk_distribution(
    teacher_logits: torch.Tensor,
    top_k: int,
    temperature: float,
    add_tail_bucket: bool,
) -> TopKDistribution:
    """Compress detached Teacher logits to Teacher Top-K plus optional tail."""

    if teacher_logits.ndim != 2:
        raise ValueError("teacher_logits must have shape [tokens, vocab]")
    if not 1 < top_k <= teacher_logits.shape[-1]:
        raise ValueError("top_k must be in [2, vocab_size]")
    scaled = teacher_logits.detach().float() / temperature
    top_values, top_indices = torch.topk(scaled, k=top_k, dim=-1)
    log_z = torch.logsumexp(scaled, dim=-1, keepdim=True)
    log_probs = top_values - log_z
    if add_tail_bucket and top_k < teacher_logits.shape[-1]:
        log_probs = _append_tail_bucket(log_probs)
    else:
        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    return TopKDistribution(indices=top_indices, log_probs=log_probs.detach())


def student_distribution_on_teacher_support(
    student_logits: torch.Tensor,
    teacher_support: TopKDistribution,
    temperature: float,
    add_tail_bucket: bool,
) -> torch.Tensor:
    """Project Student onto exactly the support selected by the Teacher."""

    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [tokens, vocab]")
    if student_logits.shape[0] != teacher_support.indices.shape[0]:
        raise ValueError("teacher/student token count mismatch")
    scaled = student_logits.float() / temperature
    selected = torch.gather(scaled, dim=-1, index=teacher_support.indices)
    log_z = torch.logsumexp(scaled, dim=-1, keepdim=True)
    log_probs = selected - log_z
    if add_tail_bucket and teacher_support.log_probs.shape[-1] == selected.shape[-1] + 1:
        log_probs = _append_tail_bucket(log_probs)
    else:
        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    return log_probs


def generalized_jsd_from_log_probs(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    beta: float,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean loss and per-token generalized Jensen-Shannon divergence."""

    if teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError("teacher/student distribution shape mismatch")
    if teacher_log_probs.ndim != 2:
        raise ValueError("log probabilities must have shape [tokens, support]")
    if not 0 < beta < 1:
        raise ValueError("beta must be strictly between zero and one")
    teacher_log_probs = teacher_log_probs.to(student_log_probs.device)
    beta_tensor = torch.tensor(beta, device=student_log_probs.device, dtype=student_log_probs.dtype)
    mixture = torch.logsumexp(
        torch.stack(
            [
                teacher_log_probs + torch.log(beta_tensor),
                student_log_probs + torch.log1p(-beta_tensor),
            ]
        ),
        dim=0,
    )
    teacher_kl = (teacher_log_probs.exp() * (teacher_log_probs - mixture)).sum(-1)
    student_kl = (student_log_probs.exp() * (student_log_probs - mixture)).sum(-1)
    per_token = beta_tensor * teacher_kl + (1 - beta_tensor) * student_kl
    per_token = per_token * (temperature**2)
    return per_token.mean(), per_token


def sampled_token_log_probs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Log probability of sampled tokens without materializing full-vocab log-softmax."""

    if logits.ndim != 2 or token_ids.ndim != 1 or logits.shape[0] != token_ids.numel():
        raise ValueError("sampled token/logit alignment failed")
    logits = logits.float()
    selected = logits.gather(-1, token_ids.to(logits.device).unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(logits, dim=-1)
