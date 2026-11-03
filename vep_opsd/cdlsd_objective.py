"""Losses and explicit gradient routing for CDLSD-VEP."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class KDDiagnostics:
    value: torch.Tensor
    per_token: torch.Tensor
    routed: torch.Tensor
    state_grad_norm: torch.Tensor


def response_cross_entropy(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or token_ids.ndim != 1:
        raise ValueError("expected logits (L,V) and token_ids (L,)")
    if logits.shape[0] != token_ids.numel():
        raise ValueError("logit/response length mismatch")
    return F.cross_entropy(logits.float(), token_ids, reduction="mean")


def forward_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor,
               *, token_clip: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-vocabulary KL(teacher || student), response-length normalized."""
    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 2:
        raise ValueError("teacher/student logits must share shape (L,V)")
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    per_token = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    if token_clip is not None and token_clip > 0:
        per_token_for_loss = per_token.clamp(max=float(token_clip))
    else:
        per_token_for_loss = per_token
    return per_token_for_loss.mean(), per_token.detach()


def route_kd_through_states(kd_loss: torch.Tensor, states: torch.Tensor) -> KDDiagnostics:
    """Preserve the KL value while routing its gradient only through ``states``.

    The decoder parameters used to compute ``kd_loss`` receive no direct KD
    gradient.  Shared parameters may still receive KD through the computation
    graph that produced ``states``.
    """
    state_gradient = torch.autograd.grad(
        kd_loss, states, retain_graph=True, create_graph=False, allow_unused=False
    )[0].detach()
    inner = (states * state_gradient).sum()
    routed = kd_loss.detach() + inner - inner.detach()
    return KDDiagnostics(
        value=kd_loss.detach(),
        per_token=torch.empty(0, device=states.device),
        routed=routed,
        state_grad_norm=state_gradient.float().norm().detach(),
    )
