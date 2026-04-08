"""Warmup + cosine decay learning rate (multiplier for LambdaLR)."""

from __future__ import annotations

import math
from typing import Callable


def warmup_cosine_lr_lambda(
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> Callable[[int], float]:
    """Returns lr_multiplier(step) for use with LambdaLR(optimizer, lr_lambda)."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return min_lr_ratio
        t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        t = min(1.0, max(0.0, t))
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return fn
