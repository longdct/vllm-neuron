# SPDX-License-Identifier: Apache-2.0
"""Reading a Qwen3.5 checkpoint's ``quantization_config``.

The point of every rejection here is that the alternative is not a crash but
*silent corruption*: scale tensors are dropped as unexpected keys and the raw
FP8 bytes are cast to bfloat16, so the model generates fluent nonsense. These
tests pin the refusals.
"""

import pytest

from vllm_neuron.model.qwen3_5.quantization import (
    Qwen3_5QuantizationSpec,
    WeightFormat,
    parse_quantization_config,
    read_quantization_spec,
)

#: The real ``quantization_config`` from ``Qwen/Qwen3.8-27B-FP8``, trimmed to
#: three of its ~512 ``modules_to_not_convert`` entries.
OFFICIAL_FP8 = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "modules_to_not_convert": [
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
        "mtp.pre_fc_norm_hidden",
    ],
    "weight_block_size": [128, 128],
}


# ---------------------------------------------------------------------------
# Unquantized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty", [None, {}])
def test_an_unquantized_checkpoint_parses_to_none(empty):
    """Absent and empty must be indistinguishable from "bf16" to callers."""
    assert parse_quantization_config(empty) is None


def test_a_non_dict_config_is_rejected():
    with pytest.raises(ValueError, match="to be a dict"):
        parse_quantization_config("fp8")


# ---------------------------------------------------------------------------
# The official blockwise-FP8 release
# ---------------------------------------------------------------------------


def test_official_fp8_checkpoint_is_recognized():
    spec = parse_quantization_config(OFFICIAL_FP8)
    assert spec is not None
    assert spec.weight_format is WeightFormat.FP8_BLOCK128
    assert spec.is_quantized()


def test_modules_left_in_full_precision_are_carried_through():
    """~512 modules stay bf16 in the official checkpoint; losing that list
    would mean quantizing tensors the producer deliberately did not."""
    spec = parse_quantization_config(OFFICIAL_FP8)
    assert len(spec.modules_to_not_convert) == 3

    # Matching is by substring: the list holds module paths, the keys tested
    # against it are parameter paths.
    assert spec.keeps_full_precision("model.visual.blocks.0.attn.qkv.weight")
    assert not spec.keeps_full_precision(
        "model.language_model.layers.3.mlp.gate_proj.weight"
    )


def test_an_empty_exclude_list_excludes_nothing():
    """An empty tuple must not accidentally match every key."""
    spec = Qwen3_5QuantizationSpec(weight_format=WeightFormat.FP8_BLOCK128)
    assert not spec.keeps_full_precision("anything.at.all.weight")


def test_a_bf16_spec_reports_itself_unquantized():
    spec = Qwen3_5QuantizationSpec(weight_format=WeightFormat.BF16)
    assert not spec.is_quantized()


# ---------------------------------------------------------------------------
# FP8 variants we cannot read
# ---------------------------------------------------------------------------


def test_a_different_block_size_is_rejected():
    """Dequantizing 128x128 tiles against a 64x64 scale grid would misread
    three quarters of every tile."""
    config = dict(OFFICIAL_FP8, weight_block_size=[64, 64])
    with pytest.raises(ValueError, match="weight_block_size"):
        parse_quantization_config(config)


def test_per_tensor_fp8_without_a_block_size_is_rejected():
    config = {k: v for k, v in OFFICIAL_FP8.items() if k != "weight_block_size"}
    with pytest.raises(ValueError, match="no.*weight_block_size"):
        parse_quantization_config(config)


def test_e5m2_is_rejected():
    """e5m2 trades two mantissa bits for exponent range; reading it as e4m3
    misinterprets every value."""
    with pytest.raises(ValueError, match="e4m3"):
        parse_quantization_config(dict(OFFICIAL_FP8, fmt="e5m2"))


def test_static_activation_scales_are_rejected():
    """The Neuron ROW path computes activation scales at runtime, so a
    checkpoint calibrated for static ones would be served with different
    arithmetic than its producer measured."""
    with pytest.raises(ValueError, match="dynamic"):
        parse_quantization_config(dict(OFFICIAL_FP8, activation_scheme="static"))


def test_modelopt_is_rejected_with_a_reason():
    """Per-tensor static FP8 has nowhere to put its input_scale tensors here."""
    config = {
        "quant_method": "modelopt",
        "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"},
    }
    with pytest.raises(ValueError, match="static input_scale"):
        parse_quantization_config(config)


def test_an_unknown_method_names_what_is_supported():
    with pytest.raises(ValueError, match="Supported"):
        parse_quantization_config({"quant_method": "awq"})


# ---------------------------------------------------------------------------
# compressed-tensors: KV-cache scales pass, quantized weights do not
# ---------------------------------------------------------------------------


def test_compressed_tensors_kv_cache_only_passes_through_as_bf16():
    """KV-cache scales are handled at the platform level and leave the
    weights in bf16, so this is not a quantized-weight checkpoint."""
    config = {"quant_method": "compressed-tensors", "kv_cache_scheme": {"num_bits": 8}}
    assert parse_quantization_config(config) is None


def test_compressed_tensors_with_quantized_weights_is_rejected():
    config = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float"}}},
    }
    with pytest.raises(ValueError, match="quantized weights"):
        parse_quantization_config(config)


def test_fp4_weights_are_rejected_and_say_why():
    """FP4 has no Trainium2 path at all -- there is no FP4 tensor-engine
    format and nkilib cannot quantize to it. The message must say so rather
    than reading as a generic 'unsupported'."""
    config = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float"}}},
    }
    with pytest.raises(ValueError, match="FP4"):
        parse_quantization_config(config)


# ---------------------------------------------------------------------------
# Getting the dict off a HuggingFace config
# ---------------------------------------------------------------------------


def test_reads_from_a_plain_dict():
    spec = read_quantization_spec({"quantization_config": OFFICIAL_FP8})
    assert spec.weight_format is WeightFormat.FP8_BLOCK128


def test_reads_from_an_attribute():
    class Cfg:
        quantization_config = OFFICIAL_FP8

    assert read_quantization_spec(Cfg()).weight_format is WeightFormat.FP8_BLOCK128


def test_falls_back_to_to_dict_when_the_attribute_is_absent():
    """Injected quantization configs live in ``__dict__`` without being
    declared on the pretrained class, so ``getattr`` alone misses them."""

    class Cfg:
        quantization_config = None

        def to_dict(self):
            return {"quantization_config": OFFICIAL_FP8}

    assert read_quantization_spec(Cfg()).weight_format is WeightFormat.FP8_BLOCK128


def test_no_config_at_all_is_none():
    assert read_quantization_spec(None) is None
