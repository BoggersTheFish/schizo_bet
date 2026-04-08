"""
Causal tension-graph propagation: learned pairwise 'tension' σ(f(h_t, h_j))
weights neighbor messages. This is NOT softmax self-attention; there is no
query-key dot-product over the full sequence—only a fixed past window.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_neighbor_stack(x: torch.Tensor, window: int) -> torch.Tensor:
    """x: B,T,D -> neighbors B,T,W,D where slot w is position t-(w+1)."""
    b, t, d = x.shape
    # B, D, T
    xc = x.transpose(1, 2)
    chunks = []
    for k in range(1, window + 1):
        pad = F.pad(xc, (k, 0))
        chunks.append(pad[:, :, :t].unsqueeze(-1))
    return torch.cat(chunks, dim=-1).permute(0, 2, 3, 1)


class CausalTensionGraphLayer(nn.Module):
    def __init__(self, dim: int, window: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.window = window
        hid = max(dim, 64)
        self.tension_net = nn.Sequential(
            nn.Linear(dim * 2, hid),
            nn.SiLU(),
            nn.Linear(hid, hid // 2),
            nn.SiLU(),
            nn.Linear(hid // 2, 1),
        )
        self.wv = nn.Linear(dim, dim, bias=False)
        self.merge = nn.Linear(dim * 2, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, return_tensions: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        nb = _causal_neighbor_stack(x, self.window)
        h_t = x.unsqueeze(2).expand(-1, -1, self.window, -1)
        pair = torch.cat([h_t, nb], dim=-1)
        tau_logits = self.tension_net(pair).squeeze(-1)
        tau = torch.sigmoid(tau_logits)
        v = self.wv(nb)
        msg = (tau.unsqueeze(-1) * v).sum(dim=2)
        y = self.merge(torch.cat([x, msg], dim=-1))
        y = self.norm(x + y)
        y = y + self.ffn(y)
        if return_tensions:
            return y, tau
        return y
