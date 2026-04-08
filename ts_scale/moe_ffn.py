"""Small MoE FFN with cross-expert mixing (Stage 3: multi-basin routing)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    """
    Router softmax over experts (optionally sparse top-k). Dense combine for small E
    keeps implementation correct and fast enough at E≤8.
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        cross_mix: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.cross_mix = cross_mix
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * hidden_mult),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * hidden_mult, dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        if self.top_k < self.num_experts:
            top_p, top_i = probs.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, top_i, top_p)
            probs = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            out += probs[:, e : e + 1] * expert(flat)
        if self.cross_mix > 0 and self.num_experts > 1:
            mix = torch.stack([expert(flat) for expert in self.experts], dim=0).mean(0)
            out = (1.0 - self.cross_mix) * out + self.cross_mix * mix
        out = out.view(b, t, d)
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
        return out, entropy
