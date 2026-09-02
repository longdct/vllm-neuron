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
from vllm_neuron.model.qwen3_5.quantization import (
    Qwen3_5QuantizationSpec,
    WeightFormat,
)
from vllm_neuron.model.registry import get_models


class _NeuronConfigStub:
    def __init__(self, quantization=None, kv_segment_size_buckets=None):
        self.quantization = quantization
        # Set by the engine whenever max_num_batched_tokens < max_model_len,
        # i.e. whenever segmented/chunked prefill is on.
        self.kv_segment_size_buckets = kv_segment_size_buckets


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


@pytest.mark.parametrize("quantization", ["mxfp4", "mxfp8", "int8"])
def test_rejects_quantized_configs(quantization):
    """MX formats need Trn3 micro-scales, which Trainium2 does not have."""
    with pytest.raises(ValueError, match="is not supported for the"):
        Qwen3_5ForCausalLM._validate_config(
            Qwen3_5TextConfig(), _NeuronConfigStub(quantization)
        )


def test_accepts_fp8_and_says_what_it_costs(caplog):
    """FP8 is lossy in a way BF16 is not, and no startup check can verify
    output quality, so accepting it must come with the number attached."""
    with caplog.at_level("WARNING"):
        Qwen3_5ForCausalLM._validate_config(
            Qwen3_5TextConfig(), _NeuronConfigStub("fp8")
        )
    assert "2.6%" in caplog.text
    assert "BF16 baseline" in caplog.text


def test_rejects_a_quantized_checkpoint_rather_than_misreading_it():
    """An FP8 checkpoint must fail at startup, not load as garbage.

    Before this gate existed the load *succeeded*: ``text_weight_mappings``
    maps only ``.weight`` keys, so every ``.weight_scale_inv`` landed in
    ``unexpected_keys`` and was discarded, and the loader then cast the raw
    ``float8_e4m3fn`` bytes to bfloat16. The result is fluent, confidently
    wrong output announced by nothing but a warning flood -- the worst
    available failure mode, and the reason this raises.
    """
    config = Qwen3_5TextConfig()
    config.quant_spec = Qwen3_5QuantizationSpec(
        weight_format=WeightFormat.FP8_BLOCK128
    )
    with pytest.raises(ValueError, match="reinterpreted as BF16"):
        Qwen3_5ForCausalLM._validate_config(config, _NeuronConfigStub())


def test_an_unquantized_checkpoint_still_passes():
    """The complement, so the gate cannot be satisfied by rejecting everything."""
    config = Qwen3_5TextConfig()
    config.quant_spec = None
    Qwen3_5ForCausalLM._validate_config(config, _NeuronConfigStub())


def test_warns_about_mtp_rather_than_refusing_to_load(caplog):
    """An advertised MTP layer must not block a real checkpoint.

    This previously raised, on the premise that no released checkpoint ships
    MTP weights. That premise was wrong -- Qwen3.5-0.8B and Qwen3.8-27B both
    ship 15 ``mtp.*`` tensors and both declare ``mtp_num_hidden_layers=1`` --
    so raising made every real checkpoint unservable at every TP degree.
    Nothing builds an MTP subtree, so the weights are simply skipped.
    """
    config = Qwen3_5TextConfig(mtp_num_hidden_layers=1)
    with caplog.at_level("WARNING"):
        Qwen3_5ForCausalLM._validate_config(config, _NeuronConfigStub())
    assert "multi-token prediction is not implemented" in caplog.text


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


def test_rejects_segmented_prefill_rather_than_silently_ignoring_it():
    """Chunked prefill must fail at startup, not produce confident wrong output.

    ``Qwen3_5Attention.forward_prefill`` has no ``kv_segment_size`` branch --
    unlike ``qwen3``, which dispatches to ``NF.segmented_attention``. Handed a
    chunk, it attends within that chunk and ignores everything cached before
    it. That is not a crash; it is coherent text computed against a truncated
    context, which is the worst way for this to fail.

    The engine sets ``kv_segment_size_buckets`` whenever
    ``max_num_batched_tokens < max_model_len``, so that field is the signal.
    """
    config = Qwen3_5TextConfig()
    assert config.needs_single_shot_prefill, "fixture must have head_dim > 128"
    neuron_config = _NeuronConfigStub(kv_segment_size_buckets=[4096])
    with pytest.raises(ValueError, match="chunked/segmented prefill is unavailable"):
        Qwen3_5ForCausalLM._validate_config(config, neuron_config)


def test_single_shot_prefill_is_accepted_when_no_segment_buckets_are_set():
    """The complement: an empty bucket list is single-shot and must pass."""
    for buckets in (None, []):
        Qwen3_5ForCausalLM._validate_config(
            Qwen3_5TextConfig(), _NeuronConfigStub(kv_segment_size_buckets=buckets)
        )


def test_tp_degree_defaults_to_one_outside_an_engine():
    assert Qwen3_5ForCausalLM._tp_degree() == 1
