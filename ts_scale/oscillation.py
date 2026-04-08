"""Layer- and position-indexed oscillatory modulation (TS wave bias on states)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class OscillatoryGate(nn.Module):
    """
    Modulates activations with sin(ω·t + φ_ℓ): not a full wave PDE, but a
    cheap inductive bias toward structured temporal filtering.
    """

    def __init__(self, dim: int, strength: float, base_omega: float, layer_idx: int) -> None:
        super().__init__()
        self.dim = dim
        self.strength = strength
        self.omega = base_omega * (1.0 + 0.17 * layer_idx)
        self.register_buffer("phase", torch.tensor(layer_idx * math.pi / 4.0))

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: B, T, D
        b, t, d = x.shape
        if positions is None:
            pos = torch.arange(t, device=x.device, dtype=x.dtype).view(1, t, 1)
        else:
            pos = positions.float().unsqueeze(-1)
        wave = torch.sin(self.omega * pos + self.phase)
        scale = 1.0 + self.strength * wave
        return x * scale
