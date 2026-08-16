# SPDX-License-Identifier: Apache-2.0
"""Production-shaped DeepSeek-V4 module assembly for CPU integration gates."""

from __future__ import annotations

import torch
from torch import nn

from .config import normalize_config
from .tiny_model import (
    TinyDecoderLayer,
    TinyDeepseekV4Config,
    TinyHyperHead,
    TinyLayerConfig,
    TinyLayerState,
    TinyModelState,
)
from .weight_loaders import ExpertDType, load_checkpoint_weights


def reference_config_from_hf(hf_config) -> TinyDeepseekV4Config:
    """Build a small-expert reference geometry from a validated HF config."""
    normalized = normalize_config(hf_config)
    hidden_size = int(hf_config.hidden_size)
    vocab_size = int(hf_config.vocab_size)
    if hidden_size < 1 or vocab_size < 1:
        raise ValueError("DeepSeek-V4 hidden and vocabulary sizes must be positive")
    return TinyDeepseekV4Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        latent_size=normalized.head_dim,
        num_experts=normalized.n_routed_experts,
        topk=normalized.num_experts_per_tok,
        hc_mult=normalized.hc_mult,
        hc_sinkhorn_iters=normalized.hc_sinkhorn_iters,
        rms_norm_eps=float(getattr(hf_config, "rms_norm_eps", 1e-6)),
        hc_eps=float(getattr(hf_config, "hc_eps", 1e-6)),
        layers=tuple(
            TinyLayerConfig(layer.compress_ratio, layer.mlp.value)
            for layer in normalized.layers
        ),
    )


class DeepseekV4Model(nn.Module):
    """Decoder body with checkpoint-compatible top-level parameter names."""

    def __init__(self, config: TinyDeepseekV4Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TinyDecoderLayer(config, layer) for layer in config.layers]
        )
        self.hyper_head = TinyHyperHead(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def new_state(self) -> TinyModelState:
        return TinyModelState([TinyLayerState() for _ in self.layers])

    def forward(
        self,
        input_ids: torch.Tensor,
        state: TinyModelState | None = None,
    ) -> tuple[torch.Tensor, TinyModelState]:
        if input_ids.ndim != 1:
            raise ValueError("DeepSeek-V4 CPU integration model expects [sequence]")
        state = self.new_state() if state is None else state
        outputs = []
        for token_id in input_ids:
            hidden = self.embed_tokens(token_id).view(1, -1)
            streams = hidden.unsqueeze(-2).expand(-1, self.config.hc_mult, -1)
            next_layers = []
            for layer, layer_state in zip(self.layers, state.layers):
                streams, layer_state = layer.forward_token(
                    streams, token_id.view(1), layer_state
                )
                next_layers.append(layer_state)
            state = TinyModelState(next_layers, state.num_tokens + 1)
            outputs.append(self.norm(self.hyper_head(streams)))
        return torch.cat(outputs, dim=0), state


class DeepseekV4ForCausalLM(nn.Module):
    """Unregistered CPU integration model; cache binding gates registry exposure."""

    is_text_generation_model = True

    def __init__(self, config: TinyDeepseekV4Config):
        super().__init__()
        self.config = config
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None):
        del neuron_config
        return cls(reference_config_from_hf(hf_config))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: TinyModelState | None = None,
    ) -> tuple[torch.Tensor, TinyModelState]:
        hidden, state = self.model(input_ids, state)
        # Keep the T0 gate independent of BLAS batch-shape accumulation order:
        # production decode emits one token row at a time, and different prefill
        # chunkings must not perturb already-computed logits.
        logits = torch.cat([self.compute_logits(row.unsqueeze(0)) for row in hidden])
        return logits, state

    def load_weights(
        self,
        weights,
        *,
        expert_dtype: ExpertDType = "bf16",
    ) -> set[str]:
        return load_checkpoint_weights(self, weights, expert_dtype=expert_dtype)
