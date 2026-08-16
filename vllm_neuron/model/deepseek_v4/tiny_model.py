# SPDX-License-Identifier: Apache-2.0
"""Small structurally-faithful DeepSeek-V4 model for CPU correctness gates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .attention import mla_attention_reference
from .compressor import CompressorState, compress_chunk
from .mhc import apply_hyperconnection, hyperconnection_reference
from .moe import hash_topk, routed_topk


@dataclass(frozen=True)
class TinyLayerConfig:
    compress_ratio: int
    mlp_kind: str = "routed_moe"


@dataclass(frozen=True)
class TinyDeepseekV4Config:
    vocab_size: int = 64
    hidden_size: int = 32
    latent_size: int = 512
    num_experts: int = 4
    topk: int = 2
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    rms_norm_eps: float = 1e-6
    hc_eps: float = 1e-6
    layers: tuple[TinyLayerConfig, ...] = (
        TinyLayerConfig(128, "hash_moe"),
        TinyLayerConfig(0, "routed_moe"),
        TinyLayerConfig(4, "routed_moe"),
        TinyLayerConfig(128, "routed_moe"),
    )


@dataclass
class TinyLayerState:
    compressor: CompressorState | None = None
    latent: torch.Tensor | None = None


@dataclass
class TinyModelState:
    layers: list[TinyLayerState]
    num_tokens: int = 0


class TinyMoE(nn.Module):
    def __init__(self, config: TinyDeepseekV4Config, kind: str):
        super().__init__()
        self.kind = kind
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * 2),
                    nn.SiLU(),
                    nn.Linear(config.hidden_size * 2, config.hidden_size),
                )
                for _ in range(config.num_experts)
            ]
        )
        self.shared = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
        )
        self.correction_bias = nn.Parameter(torch.zeros(config.num_experts))
        self.register_buffer(
            "tid2eid",
            torch.stack(
                [
                    (torch.arange(config.vocab_size) + slot).remainder(
                        config.num_experts
                    )
                    for slot in range(config.topk)
                ],
                dim=1,
            ),
        )
        self.topk = config.topk

    def forward(self, hidden: torch.Tensor, token_id: torch.Tensor) -> torch.Tensor:
        logits = self.gate(hidden)
        if self.kind == "hash_moe":
            ids, weights = hash_topk(logits, token_id, self.tid2eid)
        else:
            ids, weights = routed_topk(logits, self.correction_bias, self.topk)
        routed = torch.zeros_like(hidden)
        for slot in range(self.topk):
            routed += self.experts[int(ids[0, slot])](hidden) * weights[
                :, slot : slot + 1
            ]
        return routed + self.shared(hidden)


class TinyHyperConnection(nn.Module):
    def __init__(self, config: TinyDeepseekV4Config):
        super().__init__()
        hc = config.hc_mult
        mix = (2 + hc) * hc
        self.fn = nn.Parameter(torch.randn(mix, hc * config.hidden_size) * 0.02)
        self.base = nn.Parameter(torch.zeros(mix))
        self.scale = nn.Parameter(torch.ones(3))
        self.config = config

    def forward(self, streams: torch.Tensor):
        return hyperconnection_reference(
            streams,
            self.fn,
            self.base,
            self.scale,
            norm_eps=self.config.rms_norm_eps,
            hc_eps=self.config.hc_eps,
            iterations=self.config.hc_sinkhorn_iters,
        )


class TinyHyperHead(nn.Module):
    def __init__(self, config: TinyDeepseekV4Config):
        super().__init__()
        hc = config.hc_mult
        self.fn = nn.Parameter(torch.randn(hc, hc * config.hidden_size) * 0.02)
        self.base = nn.Parameter(torch.zeros(hc))
        self.scale = nn.Parameter(torch.ones(1))
        self.config = config

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = streams.flatten(start_dim=-2).float()
        flat = flat * torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
        )
        weights = torch.sigmoid(
            torch.nn.functional.linear(flat, self.fn.float()) * self.scale
            + self.base
        ) + self.config.hc_eps
        return (weights.unsqueeze(-1) * streams).sum(dim=-2).to(streams.dtype)


class TinyDecoderLayer(nn.Module):
    def __init__(self, config: TinyDeepseekV4Config, layer: TinyLayerConfig):
        super().__init__()
        self.ratio = layer.compress_ratio
        self.latent = nn.Linear(config.hidden_size, config.latent_size, bias=False)
        self.query = nn.Linear(config.hidden_size, config.latent_size, bias=False)
        self.key_weight = nn.Parameter(torch.eye(config.latent_size).unsqueeze(0))
        self.value_weight = nn.Parameter(
            torch.randn(1, config.latent_size, config.hidden_size)
            / config.latent_size**0.5
        )
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.moe = TinyMoE(config, layer.mlp_kind)
        self.attn_hc = TinyHyperConnection(config)
        self.ffn_hc = TinyHyperConnection(config)
        width = max(1, self.ratio)
        self.register_buffer("ape", torch.ones(width) / width)

    def forward_token(
        self, streams: torch.Tensor, token_id: torch.Tensor, state: TinyLayerState
    ) -> tuple[torch.Tensor, TinyLayerState]:
        post, comb, hidden = self.attn_hc(streams)
        projected = self.latent(hidden)
        if self.ratio:
            emitted, compressor = compress_chunk(
                projected, self.ape, self.ratio, state.compressor
            )
        else:
            emitted, compressor = projected, state.compressor
        history = state.latent
        if emitted.numel():
            history = emitted if history is None else torch.cat((history, emitted), dim=0)
        if history is None:
            attended = torch.zeros_like(hidden)
        else:
            attended = mla_attention_reference(
                self.query(hidden).view(1, 1, 1, -1),
                history.view(1, -1, history.shape[-1]),
                self.key_weight,
                self.value_weight,
                sliding_window=128 if self.ratio == 0 else None,
            ).view_as(hidden)
        streams = apply_hyperconnection(streams, self.out(attended), post, comb)
        post, comb, hidden = self.ffn_hc(streams)
        streams = apply_hyperconnection(
            streams, self.moe(hidden, token_id), post, comb
        )
        return streams, TinyLayerState(compressor=compressor, latent=history)


class TinyDeepseekV4ForCausalLM(nn.Module):
    """Batch-one token stream whose cache state survives arbitrary chunking."""

    def __init__(self, config: TinyDeepseekV4Config | None = None):
        super().__init__()
        self.config = config or TinyDeepseekV4Config()
        self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(
            [TinyDecoderLayer(self.config, layer) for layer in self.config.layers]
        )
        self.hyper_head = TinyHyperHead(self.config)
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def new_state(self) -> TinyModelState:
        return TinyModelState([TinyLayerState() for _ in self.layers])

    def forward(
        self, input_ids: torch.Tensor, state: TinyModelState | None = None
    ) -> tuple[torch.Tensor, TinyModelState]:
        if input_ids.ndim != 1:
            raise ValueError("tiny model accepts one token stream [sequence]")
        state = self.new_state() if state is None else state
        logits = []
        for token_id in input_ids:
            hidden = self.embed(token_id).view(1, -1)
            streams = hidden.unsqueeze(-2).expand(-1, self.config.hc_mult, -1)
            next_layers = []
            for layer, layer_state in zip(self.layers, state.layers):
                streams, layer_state = layer.forward_token(
                    streams, token_id.view(1), layer_state
                )
                next_layers.append(layer_state)
            state = TinyModelState(next_layers, state.num_tokens + 1)
            logits.append(self.head(self.hyper_head(streams)))
        return torch.cat(logits, dim=0), state
