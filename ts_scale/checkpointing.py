"""Save / load training state for resume."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch

from ts_scale.config import TSLMConfig


def save_checkpoint(
    path: str,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[torch.amp.GradScaler],
    cfg: TSLMConfig,
    tokenizer_type: str,
    tokenizer_path: Optional[str],
    meta_extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": asdict(cfg),
        "tokenizer_type": tokenizer_type,
        "tokenizer_path": tokenizer_path,
        "meta": meta_extra or {},
    }
    torch.save(payload, path)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "step": step,
                "tokenizer_type": tokenizer_type,
                "tokenizer_path": tokenizer_path,
                "meta": meta_extra or {},
            },
            f,
            indent=2,
        )


def apply_checkpoint_payload(
    payload: Dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[torch.amp.GradScaler],
) -> None:
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[torch.amp.GradScaler],
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    apply_checkpoint_payload(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    return payload


def config_from_checkpoint(payload: Dict[str, Any]) -> TSLMConfig:
    return TSLMConfig(**payload["config"])
