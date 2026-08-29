# SPDX-License-Identifier: Apache-2.0
"""Factory validation and registry gating for the Qwen3.5 family.

These exercise ``_validate_config`` directly rather than going through
``_select_implementation``, so they stay independent of the device model.
"""

import os
from unittest import mock

import pytest

from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig
from vllm_neuron.model.qwen3_5.factory import Qwen3_5ForCausalLM
from vllm_neuron.model.registry import get_models


class _NeuronConfigStub:
    def __init__(self, quantization=None):
        self.quantization = quantization


# ---------------------------------------------------------------------------
# Registry gating
# ---------------------------------------------------------------------------


def test_not_registered_by_default():
    """Registering unconditionally would advertise support that isn't ready."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VLLM_NEURON_ENABLE_QWEN3_5", None)
        names = [name for name, _ in get_models()]
    assert "Qwen3_5ForCausalLM" not in names
    assert "Qwen3_5ForConditionalGeneration" not in names


def test_registered_under_the_env_gate():
    with mock.patch.dict(os.environ, {"VLLM_NEURON_ENABLE_QWEN3_5": "1"}):
        registered = dict(get_models())

    # Both the text arch and the multimodal wrapper resolve to the text model:
    # the shipped 3.6/3.8 checkpoints declare the ForConditionalGeneration name.
    assert registered["Qwen3_5ForCausalLM"] is Qwen3_5ForCausalLM
    assert registered["Qwen3_5ForConditionalGeneration"] is Qwen3_5ForCausalLM


def test_gate_does_not_disturb_existing_models():
    with mock.patch.dict(os.environ, {"VLLM_NEURON_ENABLE_QWEN3_5": "1"}):
        names = [name for name, _ in get_models()]
    for expected in ("LlamaForCausalLM", "Qwen3ForCausalLM", "GptOssForCausalLM"):
        assert expected in names


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_accepts_bf16():
    config = Qwen3_5TextConfig()
    for quantization in (None, "bf16"):
        Qwen3_5ForCausalLM._validate_config(config, _NeuronConfigStub(quantization))


@pytest.mark.parametrize("quantization", ["mxfp4", "mxfp8", "fp8"])
def test_rejects_quantized_configs(quantization):
    with pytest.raises(ValueError, match="is not supported for the"):
        Qwen3_5ForCausalLM._validate_config(
            Qwen3_5TextConfig(), _NeuronConfigStub(quantization)
        )


def test_rejects_mtp_because_no_checkpoint_ships_those_weights():
    config = Qwen3_5TextConfig(mtp_num_hidden_layers=1)
    with pytest.raises(ValueError, match="multi-token prediction is not implemented"):
        Qwen3_5ForCausalLM._validate_config(config, _NeuronConfigStub())


def test_rejects_unsupported_tp_degree_at_startup():
    """An unsupported TP degree must fail loudly, not shard silently wrong."""
    config = Qwen3_5TextConfig(linear_num_key_heads=3, linear_num_value_heads=6)
    with mock.patch.object(Qwen3_5ForCausalLM, "_tp_degree", staticmethod(lambda: 2)):
        with pytest.raises(ValueError, match="not supported for Gated DeltaNet"):
            Qwen3_5ForCausalLM._validate_config(config, _NeuronConfigStub())


def test_warns_that_head_dim_256_forces_single_shot_prefill(caplog):
    with caplog.at_level("WARNING"):
        Qwen3_5ForCausalLM._validate_config(Qwen3_5TextConfig(), _NeuronConfigStub())
    assert "single-shot" in caplog.text


def test_tp_degree_defaults_to_one_outside_an_engine():
    assert Qwen3_5ForCausalLM._tp_degree() == 1
