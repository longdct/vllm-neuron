# SPDX-License-Identifier: Apache-2.0
"""Factory for DeepSeek-V4, matching vLLM's ModelRegistry factory pattern.

Mirrors ``vllm_neuron/model/qwen3/factory.py``: a thin ``nn.Module`` that
validates config and delegates to the real implementation in ``model.py``.
This is the "trivial part, done last" step 5 the roadmap describes for the
final registry entry -- registration itself
(``vllm_neuron/model/registry.py``) stays opt-in behind
``VLLM_NEURON_ENABLE_DEEPSEEK_V4=1``, since real checkpoint loading and
full-scale validation (roadmap steps 4-5) are still out of scope.

The previous per-layer ``ComponentRegistry``/``resolve_layer_components``
helper this module used to hold has no equivalent left to select between:
device-shaped ``model.py`` dispatches attention/MLP per layer internally
from ``NormalizedDeepseekV4Config`` directly (see ``DeepseekV4DecoderLayer``),
the same way every other single-implementation model in this plugin does.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class DeepseekV4ForCausalLM(nn.Module):
    """Factory that validates config and selects the DeepSeek-V4 implementation.

    Extends ``nn.Module`` to satisfy vLLM's ``ModelRegistry`` requirements.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        from .model import DeepseekV4ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Reject variants the device path does not implement.

        ``config.py::normalize_config`` (called by ``model.py`` during
        construction) already rejects unrecognized layer/MLP vocabularies,
        unsupported compression ratios, and non-MLA ``num_key_value_heads``.
        This adds the checks that are about *device-path scope*, not config
        shape -- quantization is not implemented here at all yet (roadmap
        step 4/P9).
        """
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16", "fp8"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for "
                "DeepSeek-V4. Expected one of: None, 'bf16', 'fp8'. "
                "'fp8' widens the checkpoint's MXFP4 routed experts to e4m3 "
                "with per-output-channel scales; attention and shared experts "
                "are still dequantized to BF16."
            )
