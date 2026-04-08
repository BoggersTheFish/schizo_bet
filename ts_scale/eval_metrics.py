"""Stage 5: perplexity and cross-entropy on held-out batches."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn


@torch.no_grad()
def eval_lm_metrics(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    ppl_cap_nats: float = 20.0,
) -> Dict[str, Any]:
    """
    Returns mean NLL in nats/token plus a capped PPL for stable logging.

    When ``mean_nll_nats > ppl_cap_nats``, ``ppl_capped`` is ``exp(ppl_cap_nats)``
    and ``ppl_saturated`` is True — the true PPL is ``exp(mean_nll_nats)`` (often huge).
    """
    model.eval()
    nll = 0.0
    ntok = 0
    for ids, labels in batches:
        ids = ids.to(device)
        labels = labels.to(device)
        out = model(ids, labels=labels)
        b, t = ids.shape
        ce = torch.nn.functional.cross_entropy(
            out["logits"].reshape(-1, out["logits"].size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )
        nll += ce.item()
        ntok += t * b
    mean_nll = nll / max(1, ntok)
    capped = mean_nll > ppl_cap_nats
    return {
        "mean_nll_nats": mean_nll,
        "ppl_capped": math.exp(min(ppl_cap_nats, mean_nll)),
        "ppl_saturated": capped,
        "ppl_true": math.exp(mean_nll) if mean_nll < 80 else float("inf"),
    }


@torch.no_grad()
def perplexity_on_batches(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    """Backward-compatible: returns ``ppl_capped`` from :func:`eval_lm_metrics`."""
    return eval_lm_metrics(model, batches, device)["ppl_capped"]


def training_step_log(
    out: dict[str, torch.Tensor],
) -> dict[str, float]:
    row: dict[str, float] = {"loss": float(out["loss"].detach())}
    for k in (
        "ce",
        "vq",
        "loss_supervised",
        "aux_attractor",
        "aux_tension_ent",
        "aux_moe_deficit",
        "moe_router_ent",
    ):
        if k in out:
            row[k] = float(out[k].detach())
    return row
