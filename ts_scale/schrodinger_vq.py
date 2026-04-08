"""
Differentiable 'collapse' of discrete latent structure (Stage 4).
VQ with commitment loss; optional Gumbel-softmax straight-through for softer superposition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SchrodingerVQ(nn.Module):
    def __init__(
        self,
        dim: int,
        num_codes: int,
        use_gumbel: bool = False,
        temperature: float = 0.7,
    ) -> None:
        super().__init__()
        self.codebook = nn.Embedding(num_codes, dim)
        nn.init.normal_(self.codebook.weight, std=0.02)
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.use_gumbel = use_gumbel
        self.temperature = temperature

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj_in(h)
        cb = self.codebook.weight
        d = (
            z.pow(2).sum(-1, keepdim=True)
            - 2 * z @ cb.T
            + cb.pow(2).sum(-1).unsqueeze(0).unsqueeze(0)
        )
        if self.use_gumbel and self.training:
            g = -torch.log(-torch.log(torch.rand_like(d) + 1e-8) + 1e-8)
            soft = F.softmax((-d + g) / self.temperature, dim=-1)
            hard = torch.zeros_like(soft)
            hard.scatter_(-1, soft.argmax(dim=-1, keepdim=True), 1.0)
            onehot = hard - soft.detach() + soft
            z_e = onehot @ cb
            z_q = z + (z_e - z).detach()
        else:
            idx = d.argmin(dim=-1)
            z_e = self.codebook(idx)
            if self.training:
                z_q = z + (z_e - z).detach()
            else:
                z_q = z_e
        h_out = h + self.proj_out(z_q)
        commit = F.mse_loss(z_e, z.detach())
        codebook = F.mse_loss(z_e.detach(), z)
        vq_loss = commit + codebook
        return h_out, vq_loss
