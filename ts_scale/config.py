"""Hyperparameters for scaled TS-LM (Stages 1–5 hooks)."""

from dataclasses import dataclass


@dataclass
class TSLMConfig:
    vocab_size: int = 256
    dim: int = 256
    n_layers: int = 6
    window: int = 16
    ffn_mult: int = 4
    dropout: float = 0.1
    # Stage 1 / wave bias
    oscillation_strength: float = 0.15
    base_omega: float = 0.02
    # Stage 3 MoE
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_every_n_layers: int = 2
    moe_cross_mix: float = 0.12
    # Stage 4 Schrödinger / VQ
    vq_num_codes: int = 512
    vq_commit_weight: float = 0.25
    vq_use_gumbel: bool = False
    vq_temperature: float = 0.7
    # Losses (Stage 1 + 5)
    aux_attractor_weight: float = 0.03
    aux_tension_entropy_weight: float = 0.02
    max_seq_len: int = 512
