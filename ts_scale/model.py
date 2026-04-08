"""
TSLanguageModel: scaled TS philosophy without a vanilla Transformer block stack.

Backbone = stacked causal **tension-graph** layers (learned σ-pairwise gates on a
local past window). Per layer: oscillatory modulation (wave bias), every N-th
layer an MoE FFN (multi-basin), mid-stack Schrödinger VQ (differentiable collapse).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ts_scale.config import TSLMConfig
from ts_scale.losses import (
    attractor_manifold_loss,
    cross_entropy_lm,
    tension_entropy_deficit,
)
from ts_scale.moe_ffn import MoEFFN
from ts_scale.oscillation import OscillatoryGate
from ts_scale.schrodinger_vq import SchrodingerVQ
from ts_scale.tension_graph import CausalTensionGraphLayer


class TSLanguageModel(nn.Module):
    def __init__(self, cfg: TSLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList()
        self.osc_gates = nn.ModuleList()
        self.moe_layers: dict[int, MoEFFN] = {}
        for i in range(cfg.n_layers):
            self.layers.append(
                CausalTensionGraphLayer(cfg.dim, cfg.window, cfg.dropout)
            )
            self.osc_gates.append(
                OscillatoryGate(
                    cfg.dim,
                    cfg.oscillation_strength,
                    cfg.base_omega,
                    layer_idx=i,
                )
            )
            if (i + 1) % cfg.moe_every_n_layers == 0:
                self.moe_layers[i] = MoEFFN(
                    cfg.dim,
                    cfg.ffn_mult,
                    cfg.moe_num_experts,
                    cfg.moe_top_k,
                    cfg.dropout,
                    cfg.moe_cross_mix,
                )
        self._vq_layer_index = cfg.n_layers // 2
        self.schrodinger = SchrodingerVQ(
            cfg.dim,
            cfg.vq_num_codes,
            use_gumbel=cfg.vq_use_gumbel,
            temperature=cfg.vq_temperature,
        )
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_loss_breakdown: bool = False,
    ) -> dict[str, torch.Tensor]:
        x = self.embed(input_ids)
        h0 = x
        tension_list: list[torch.Tensor] = []
        router_ent: list[torch.Tensor] = []
        vq_loss = torch.tensor(0.0, device=x.device)

        for i, layer in enumerate(self.layers):
            x, tau = layer(x, return_tensions=True)
            tension_list.append(tau)
            x = self.osc_gates[i](x)
            if i in self.moe_layers:
                moe_out, ent = self.moe_layers[i](x)
                x = x + moe_out
                router_ent.append(ent)
            if i == self._vq_layer_index:
                x, vq_loss = self.schrodinger(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is None:
            return out

        ce = cross_entropy_lm(logits, labels)

        aux_att = attractor_manifold_loss(
            h0, x, self.cfg.aux_attractor_weight
        )
        aux_ent = torch.tensor(0.0, device=ce.device)
        if tension_list:
            aux_ent = sum(
                tension_entropy_deficit(
                    t, self.cfg.aux_tension_entropy_weight / len(tension_list)
                )
                for t in tension_list
            )

        moe_ent_mean = (
            torch.stack(router_ent).mean()
            if router_ent
            else torch.tensor(0.0, device=ce.device)
        )
        # Small penalty if routing collapses (keeps total loss interpretable vs CE).
        aux_moe = torch.tensor(0.0, device=ce.device)
        if router_ent:
            e_max = math.log(max(self.cfg.moe_num_experts, 2))
            aux_moe = 0.002 * torch.clamp(e_max - moe_ent_mean, min=0.0)

        loss = (
            ce
            + self.cfg.vq_commit_weight * vq_loss
            + aux_att
            + aux_ent
            + aux_moe
        )
        out["loss"] = loss
        if return_loss_breakdown:
            out["ce"] = ce.detach()
            out["vq"] = vq_loss.detach()
            out["aux_attractor"] = aux_att.detach()
            out["aux_tension_ent"] = aux_ent.detach()
            out["aux_moe_deficit"] = aux_moe.detach()
            out["moe_router_ent"] = moe_ent_mean.detach()
            out["loss_supervised"] = (ce + self.cfg.vq_commit_weight * vq_loss).detach()
        return out

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        self.eval()
        cur = prompt_ids
        for _ in range(max_new_tokens):
            if cur.size(1) > self.cfg.max_seq_len:
                cur = cur[:, -self.cfg.max_seq_len :]
            logits = self.forward(cur)["logits"][:, -1] / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            cur = torch.cat([cur, nxt], dim=1)
        return cur
