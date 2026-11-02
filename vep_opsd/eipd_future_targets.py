"""Future-state supervision for EIPD v2.0.

Two things live here:

* `assign_anchors` — deterministic, monotonic mapping from real future-frame
  timestamps to latent indices. Half-up rounding, no dependence on Python's
  banker-rounding, and a hard guarantee that anchor j never collides with
  anchor j-1 and never steals the slots reserved for later anchors.
* `future_prediction_loss` — state alignment (cosine) plus a **vector-level**
  temporal term. The temporal term is L2^2 over the whole difference vector,
  divided by 2(m-1); averaging a per-dimension Huber over 4096 dims would shrink
  it into numerical noise and let the state term silently dominate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AnchorAssignment:
    anchors: tuple[int, ...]        # 1-based latent indices, strictly increasing
    forced: int                     # anchors the monotonic/slots rule had to move
    total: int


def select_future_frames(future_indices: list[int], steps: int) -> list[int]:
    """Keep at most `steps` future frames, evenly spaced over time, last one always.

    Needed once the latent budget drops below the number of supervised frames (the
    data has m up to 7). Selection is deterministic and never invents a frame: the
    kept indices are a subset of the real ones, so no interpolated target is used.
    """
    future = list(future_indices)
    if len(future) <= steps:
        return future
    if steps == 1:
        return [future[-1]]
    last = len(future) - 1
    keep = sorted({int(i * last / (steps - 1) + 0.5) for i in range(steps)})
    keep[-1] = last
    return [future[i] for i in keep]


def assign_anchors(past_end: int, future_indices: Sequence[int], *, steps: int = 16
                   ) -> AnchorAssignment:
    """Map future frame positions onto latent indices in 1..steps.

    u_j = (n_j - n_0) / (n_m - n_0),  k_j = clamp(floor(K*u_j + 0.5), k_{j-1}+1,
    K-(m-j)),  and k_m = K.
    """
    future = list(int(x) for x in future_indices)
    m = len(future)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 1 <= m <= steps:
        raise ValueError(f"need 1..{steps} future frames, got {m}")
    if any(t <= past_end for t in future):
        raise ValueError("future frames must follow the last observed frame")
    if any(a >= b for a, b in zip(future, future[1:])):
        raise ValueError("future frame indices must strictly increase")

    horizon = future[-1] - past_end
    anchors: list[int] = []
    forced = 0
    previous = 0
    for j, timestamp in enumerate(future, start=1):
        if j == m:
            anchor = steps
        else:
            u = (timestamp - past_end) / horizon
            proposed = int(steps * u + 0.5)                  # half-up, explicit
            lower = previous + 1
            upper = steps - (m - j)
            anchor = min(max(proposed, lower), upper)
            if anchor != proposed:
                forced += 1
        if anchor <= previous:
            raise RuntimeError("anchor assignment lost monotonicity")
        anchors.append(anchor)
        previous = anchor
    if anchors[-1] != steps:
        raise RuntimeError("last anchor must be the final latent")
    return AnchorAssignment(tuple(anchors), forced, m)


def future_prediction_loss(predicted: torch.Tensor, target: torch.Tensor, *,
                           eta: float = 0.5) -> dict[str, torch.Tensor]:
    """`predicted`/`target` are (m, d) vectors for the m supervised anchors.

    Returns the components as well as the combined loss so training can log the
    raw values and the gradient scale of each term.
    """
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError("predicted and target must share shape (m, d)")
    m = predicted.shape[0]
    if m == 0:
        raise ValueError("empty anchor set")
    if eta < 0:
        raise ValueError("eta must be non-negative")

    predicted = F.normalize(predicted.float(), dim=-1)
    target = F.normalize(target.detach().float(), dim=-1)     # stop-gradient target

    state = (1.0 - (predicted * target).sum(dim=-1)).mean()
    zero = state.new_zeros(())
    if m >= 2:
        error = (predicted[1:] - predicted[:-1]) - (target[1:] - target[:-1])
        temporal = error.pow(2).sum(dim=-1).mean() / 2.0
    else:
        temporal = zero
    total = state + eta * temporal
    return {"loss": total, "state_loss": state, "temporal_loss": temporal,
            "anchors": torch.tensor(float(m), device=predicted.device)}
