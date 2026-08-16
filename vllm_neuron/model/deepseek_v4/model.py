# SPDX-License-Identifier: Apache-2.0
"""Production-shaped DeepSeek-V4 module assembly for CPU integration gates."""

from __future__ import annotations

import torch
from torch import nn

from ..kv_cache import CacheKind, KVSpec, LayerSpec
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
        sliding_window=normalized.sliding_window,
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
        self._kv_caches: dict[str, list[torch.Tensor]] = {}

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None):
        del neuron_config
        return cls(reference_config_from_hf(hf_config))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def get_kv_spec(self) -> KVSpec:
        """Declare the three engine-owned cache layouts used by V4 layers.

        All of these are single-vector MLA layouts.  Every layer owns an
        uncompressed sliding window; compressed layers additionally own their
        long-context latent pages and the compressor's fp32 carry pages.
        """
        specs: list[LayerSpec] = []
        for index, layer in enumerate(self.config.layers):
            prefix = f"model.layers.{index}.self_attn"
            specs.append(
                LayerSpec(
                    name=f"{prefix}.swa_cache",
                    num_kv_heads=1,
                    head_size=self.config.latent_size,
                    dtype=torch.bfloat16,
                    cache_kind=CacheKind.SLIDING_WINDOW_MLA,
                    sliding_window_size=self.config.sliding_window,
                    alignment=128,
                )
            )
            if layer.compress_ratio == 0:
                continue
            ratio = layer.compress_ratio
            specs.append(
                LayerSpec(
                    name=prefix,
                    num_kv_heads=1,
                    head_size=self.config.latent_size,
                    dtype=torch.bfloat16,
                    cache_kind=CacheKind.MLA,
                    compress_ratio=ratio,
                    alignment=128,
                )
            )
            overlap = ratio == 4
            specs.append(
                LayerSpec(
                    name=f"{prefix}.compressor.state_cache",
                    num_kv_heads=1,
                    head_size=2 * (1 + overlap) * self.config.latent_size,
                    dtype=torch.float32,
                    block_size=4 if overlap else 8,
                    cache_kind=CacheKind.COMPRESSOR_STATE,
                    sliding_window_size=(1 + overlap) * ratio,
                    alignment=512,
                )
            )
        return KVSpec(specs)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        expected = {spec.name for spec in self.get_kv_spec().layers}
        actual = set(kv_caches)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"DeepSeek-V4 KV cache keys do not match: missing={missing}, "
                f"unexpected={unexpected}"
            )
        wrong_arity = sorted(
            name for name, tensors in kv_caches.items() if len(tensors) != 1
        )
        if wrong_arity:
            raise ValueError(
                "DeepSeek-V4 caches use one latent tensor; invalid layers: "
                f"{wrong_arity}"
            )
        self._kv_caches = kv_caches

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
