"""Frozen EIPD objective: answer-only, future-evidence-increment, forward KL."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class EIPDResult:
    loss: torch.Tensor
    mean_js: torch.Tensor
    active_tokens: int
    normalized_weights: torch.Tensor

def eipd_loss(student_logits: torch.Tensor,
              no_bridge_logits: torch.Tensor,
              bridge_logits: torch.Tensor,
              *, alpha: float = 0.5, max_weight: float = 4.0,
              eps: float = 1e-8) -> EIPDResult:
    """All inputs are answer-token logits, shaped [T,V], aligned under teacher forcing.

    Teacher logits must be detached. KL(q||student) uses the *full* vocabulary;
    there is no Top-K renormalization and no language-prefix KL.
    """
    if not (student_logits.ndim == 2 and student_logits.shape == no_bridge_logits.shape == bridge_logits.shape):
        raise ValueError("Expected identical [answer tokens, vocabulary] logits")
    if not (0 < alpha <= 1 and max_weight >= 1 and eps > 0):
        raise ValueError("Invalid EIPD hyperparameters")
    if student_logits.shape[0] == 0:
        raise ValueError("Empty answer span")
    with torch.no_grad():
        p0 = F.softmax(no_bridge_logits.detach().float(), dim=-1)
        p1 = F.softmax(bridge_logits.detach().float(), dim=-1)
        q = ((1 - alpha) * p0 + alpha * p1).clamp_min(eps)
        q = q / q.sum(dim=-1, keepdim=True)
        middle = (p0 + p1) / 2
        d0 = torch.where(p0 > 0, p0 * (p0.clamp_min(eps).log() - middle.clamp_min(eps).log()), 0).sum(-1)
        d1 = torch.where(p1 > 0, p1 * (p1.clamp_min(eps).log() - middle.clamp_min(eps).log()), 0).sum(-1)
        d = (0.5 * (d0 + d1)).clamp_min(0)
        if d.sum() <= eps:
            weights = torch.zeros_like(d)
        else:
            # Bounded weighting. Re-normalize after the cap; the mean is 1
            # for active samples (unless zero-divergence tokens dominate).
            weights = d * (d.numel() / d.sum())
            weights = weights.clamp(max=max_weight)
            weights = weights * (d.numel() / weights.sum().clamp_min(eps))
    log_s = F.log_softmax(student_logits.float(), dim=-1)
    per_token = (q * (q.log() - log_s)).sum(-1)
    loss = (per_token * weights).mean()
    return EIPDResult(loss, d.mean().detach(), int((d > eps).sum().item()), weights.detach())
