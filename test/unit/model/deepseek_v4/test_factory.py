# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4's ModelRegistry factory: config validation and delegation.

Replaces the old ``ComponentRegistry``/``resolve_layer_components`` tests --
that per-layer dispatch helper had no equivalent in the roadmap's step 5
factory pattern (``_select_implementation``/``_validate_config``, matching
``qwen3/factory.py``) and per-layer attention/MLP resolution now happens
directly inside ``model.py``'s ``DeepseekV4DecoderLayer`` construction,
covered by ``test/vllm_neuron/test_deepseek_v4_model_assembly.py``'s
``test_hf_config_builds_device_shaped_multi_variant_model``.

This module imports through the ``vllm_neuron`` package ``__init__``, which
needs vLLM (same reason this file is excluded from the CPU-only T0 gate on
the device-validation runbook).
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("vllm")

from transformers import DeepseekV4Config

from vllm_neuron.model.deepseek_v4.factory import DeepseekV4ForCausalLM
from vllm_neuron.model.neuron_config import NeuronConfig


def hf_config():
    return DeepseekV4Config(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        layer_types=["heavily_compressed_attention", "sliding_attention"],
        mlp_layer_types=["moe", "moe"],
    )


def test_from_configs_delegates_to_the_real_model():
    from vllm_neuron.model.deepseek_v4.model import (
        DeepseekV4ForCausalLM as RealModel,
    )

    model = DeepseekV4ForCausalLM.from_configs(hf_config(), None)
    assert isinstance(model, RealModel)


def test_factory_instance_forwards_through_to_the_real_model():
    model = DeepseekV4ForCausalLM(hf_config(), None)
    assert model._model.config.hidden_size == 32


@pytest.mark.parametrize("quantization", ["fp8", "mxfp4", "compressed-tensors"])
def test_validate_config_rejects_quantization_not_yet_implemented(quantization):
    neuron_config = NeuronConfig(quantization=quantization)
    with pytest.raises(ValueError, match="quantization"):
        DeepseekV4ForCausalLM.from_configs(hf_config(), neuron_config)


@pytest.mark.parametrize("quantization", [None, "bf16"])
def test_validate_config_accepts_bf16(quantization):
    neuron_config = NeuronConfig(quantization=quantization)
    model = DeepseekV4ForCausalLM.from_configs(hf_config(), neuron_config)
    assert model is not None
