# SPDX-License-Identifier: Apache-2.0
"""Factory for the Qwen3.5-family text decoder (Qwen3.5 / 3.6 / 3.8).

Validates up front so unsupported configurations fail with a clear message
rather than deep inside weight loading or, worse, silently. Follows the shape of
``qwen3/factory.py`` and ``deepseek_v4/factory.py``: subclass ``nn.Module`` to
satisfy vLLM's ``ModelRegistry``, but hand back the real implementation from the
``from_configs`` classmethod.
"""

import logging

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .config import Qwen3_5TextConfig
from .parallel import resolve_sharding

logger = logging.getLogger(__name__)


class Qwen3_5ForCausalLM(nn.Module):
    """Factory that validates config and selects the Qwen3.5 implementation."""

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        config = Qwen3_5TextConfig.from_configs(hf_config, neuron_config)
        cls._validate_config(config, neuron_config)

        from .model import Qwen3_5TextForCausalLM as Model

        return Model.from_configs(config, neuron_config)

    @classmethod
    def _validate_config(
        cls, config: Qwen3_5TextConfig, neuron_config: NeuronConfig | None
    ) -> None:
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for the "
                "Qwen3.5 family. Only BF16 (None or 'bf16') is implemented."
            )

        # The config advertises an MTP layer but no released checkpoint ships
        # MTP weights, so honouring it would build a subtree nothing can fill.
        if config.mtp_num_hidden_layers:
            raise ValueError(
                "mtp_num_hidden_layers="
                f"{config.mtp_num_hidden_layers} but multi-token prediction is "
                "not implemented, and the released Qwen3.5-family checkpoints "
                "contain no MTP weights. Set it to 0."
            )

        # Resolve the sharding policy now so an unsupported TP degree is a
        # startup error rather than a silently wrong shard.
        tp_degree = cls._tp_degree()
        resolve_sharding(config, tp_degree)

        if config.needs_single_shot_prefill:
            # head_dim > 128 cannot use the segmented-attention kernel at all
            # (it raises), so chunked prefill is unavailable for the
            # full-attention layers. Flag it loudly; the engine-level bucket
            # validation enforces the corresponding max_model_len limit.
            logger.warning(
                "Qwen3.5 head_dim=%d exceeds the segmented-attention kernel's "
                "128-element partition bound, so chunked/segmented prefill is "
                "unavailable and prefill must be single-shot. Keep "
                "max_model_len within the single-shot limit.",
                config.head_dim,
            )

    @staticmethod
    def _tp_degree() -> int:
        """Tensor-parallel degree from vLLM, defaulting to 1 outside an engine."""
        try:
            from vllm.config import get_current_vllm_config

            return get_current_vllm_config().parallel_config.tensor_parallel_size
        except Exception:  # pragma: no cover - unit tests run outside an engine
            return 1
