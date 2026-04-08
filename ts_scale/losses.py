"""Training objectives: CE + TS-style auxiliary terms (Stages 1 & 5)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def cross_entropy_lm(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
    )


def attractor_manifold_loss(
    h_first: torch.Tensor, h_last: torch.Tensor, weight: float
) -> torch.Tensor:
    if weight <= 0:
        return torch.tensor(0.0, device=h_first.device)
    return weight * F.mse_loss(h_last, h_first.detach())


def tension_entropy_deficit(tensions: torch.Tensor, weight: float) -> torch.Tensor:
    """
    Non-negative penalty when neighbor-mass is too peaked (low entropy).
    tensions: B,T,W — σ-gated masses over the causal window.
    """
    if weight <= 0:
        return torch.tensor(0.0, device=tensions.device)
    w = tensions.size(-1)
    max_ent = math.log(max(w, 2))
    p = tensions / (tensions.sum(dim=-1, keepdim=True) + 1e-8)
    ent = -(p * (p + 1e-9).log()).sum(dim=-1).mean()
    deficit = torch.clamp(max_ent - ent, min=0.0)
    return weight * deficit
