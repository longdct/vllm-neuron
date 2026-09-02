# SPDX-License-Identifier: Apache-2.0
"""Assembling the FP8 Qwen3.5 decoder.

These build real modules on the meta device -- no weights, no Neuron -- and
check the things that silently undo quantization if they go wrong: parameter
dtypes, scale shapes, checkpoint mappings, and the blanket dtype cast in
``load_weights``.
"""

import pytest
import torch

from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig
from vllm_neuron.model.qwen3_5.model_fp8 import (
    SCALE_SUFFIX,
    Qwen3_5AttentionFP8,
    Qwen3_5MLPFP8,
    resolve_module_classes,
)
from vllm_neuron.model.qwen3_5.model import Qwen3_5Attention, Qwen3_5MLP
from vllm_neuron.model.qwen3_5.parallel import resolve_sharding
from vllm_neuron.model.qwen3_5.weight_loaders_fp8 import FP8_DTYPE
from vllm_neuron.utils.weight_loader import get_weight_loader

_PMAX = 128


def _small_config(**kw):
    """A tiny but structurally faithful config: both layer kinds, real dims."""
    base = dict(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=128,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    base.update(kw)
    return Qwen3_5TextConfig(**base)


def _policy(config):
    return resolve_sharding(config, 1)


# ---------------------------------------------------------------------------
# Module class selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantization", [None, "bf16"])
def test_bf16_selects_the_unquantized_modules(quantization):
    assert resolve_module_classes(quantization, None) == (
        Qwen3_5Attention,
        Qwen3_5MLP,
    )


def test_fp8_selects_the_quantized_modules():
    assert resolve_module_classes("fp8", None) == (
        Qwen3_5AttentionFP8,
        Qwen3_5MLPFP8,
    )


def test_modules_to_not_convert_keeps_attention_in_bf16():
    """"MLP only" must be reachable by config, not by editing code."""
    attn, mlp = resolve_module_classes("fp8", ["self_attn"])
    assert attn is Qwen3_5Attention
    assert mlp is Qwen3_5MLPFP8


def test_modules_to_not_convert_keeps_the_mlp_in_bf16():
    attn, mlp = resolve_module_classes("fp8", ["mlp"])
    assert attn is Qwen3_5AttentionFP8
    assert mlp is Qwen3_5MLP


def test_excluding_everything_warns_rather_than_silently_doing_nothing(caplog):
    with caplog.at_level("WARNING"):
        resolve_module_classes("fp8", ["mlp", "self_attn"])
    assert "stays BF16" in caplog.text


# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------


def test_mlp_weights_are_fp8_with_a_scale_each():
    config = _small_config()
    with torch.device("meta"):
        mlp = Qwen3_5MLPFP8(config, _policy(config))

    for name in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        weight = getattr(mlp, name)
        scale = getattr(mlp, name + SCALE_SUFFIX)
        assert weight.dtype is FP8_DTYPE, name
        assert scale.dtype is torch.float32, name
        # One scale per output channel, broadcast across the partition dim.
        assert scale.shape == (_PMAX, weight.shape[1]), name


def test_attention_weights_are_fp8_with_a_scale_each():
    config = _small_config()
    with torch.device("meta"):
        attn = Qwen3_5AttentionFP8(config, _policy(config), layer_idx=3)

    for name in ("qkv_proj_weight", "o_proj_weight"):
        weight = getattr(attn, name)
        scale = getattr(attn, name + SCALE_SUFFIX)
        assert weight.dtype is FP8_DTYPE, name
        assert scale.shape == (_PMAX, weight.shape[1]), name


def test_fp8_modules_keep_the_bf16_shapes():
    """Quantization must change dtype only -- a shape change would mean the
    sharding or transposition logic was accidentally restated."""
    config = _small_config()
    with torch.device("meta"):
        bf16 = Qwen3_5MLPFP8(config, _policy(config))
        ref = Qwen3_5MLP(config, _policy(config))
    for name in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        assert getattr(bf16, name).shape == getattr(ref, name).shape


def test_every_fp8_parameter_carries_a_loader():
    """A parameter with no loader is silently filled with the raw checkpoint
    slice -- bf16 bytes into an fp8 tensor."""
    config = _small_config()
    with torch.device("meta"):
        mlp = Qwen3_5MLPFP8(config, _policy(config))
    for name, param in mlp.named_parameters():
        assert get_weight_loader(param).transform is not None, name


def test_scales_are_parameters_so_the_loader_pipeline_fills_them():
    """``checkpoints.py`` iterates ``named_parameters()`` only, so a scale
    declared as a buffer would never be loaded at all."""
    config = _small_config()
    with torch.device("meta"):
        mlp = Qwen3_5MLPFP8(config, _policy(config))
    names = {n for n, _ in mlp.named_parameters()}
    assert "gate_proj_weight" + SCALE_SUFFIX in names
    assert not list(mlp.named_buffers())


# ---------------------------------------------------------------------------
# Checkpoint wiring
# ---------------------------------------------------------------------------


def _build_model(quantization):
    from vllm_neuron.model.neuron_config import NeuronConfig
    from vllm_neuron.model.qwen3_5.model import Qwen3_5TextForCausalLM

    # On-device sampling builds a Sampler against the TP process group, which
    # does not exist outside an engine. Irrelevant to weight precision.
    neuron_config = NeuronConfig(
        quantization=quantization, on_device_sampling_config=None
    )
    config = _small_config(neuron_config=neuron_config)
    with torch.device("meta"):
        return Qwen3_5TextForCausalLM(config)


def test_scale_parameters_map_to_their_weight_checkpoint_key():
    """The scale is derived from the weight, so it reads the same key. Without
    a mapping it would be looked up under its own name, which no checkpoint
    has, and strict loading would fail."""
    model = _build_model("fp8")
    from vllm_neuron.model.qwen3_5.weight_loaders import text_weight_mappings

    mappings = text_weight_mappings(model.config)
    extra = model._fp8_scale_mappings(mappings)

    assert extra, "fp8 model produced no scale mappings"
    for scale_name, key in extra.items():
        weight_name = scale_name[: -len(SCALE_SUFFIX)]
        assert key == mappings[weight_name]

    # The fused QKV weight maps to a list of three keys; its scale must too.
    qkv = [k for k in extra if "qkv_proj_weight" in k]
    assert qkv
    assert isinstance(extra[qkv[0]], list)
    assert len(extra[qkv[0]]) == 3


def test_the_bf16_model_produces_no_scale_mappings():
    model = _build_model(None)
    from vllm_neuron.model.qwen3_5.weight_loaders import text_weight_mappings

    assert model._fp8_scale_mappings(text_weight_mappings(model.config)) == {}


def test_gated_deltanet_layers_stay_bf16_under_fp8():
    """Their five projections are plain nn.Linear with no kernel quant path,
    so 20.7% of the 27B is deliberately left alone. If this ever changes, the
    coverage numbers in the docs change with it."""
    model = _build_model("fp8")
    linear_layer = model.model.layers[0]
    assert linear_layer.is_linear
    assert linear_layer.linear_attn.in_proj_qkv.weight.dtype is torch.bfloat16
    # ...while the MLP on that same layer is quantized.
    assert linear_layer.mlp.gate_proj_weight.dtype is FP8_DTYPE


def test_embeddings_and_lm_head_stay_bf16_under_fp8():
    model = _build_model("fp8")
    assert model.lm_head.weight.dtype is torch.bfloat16
    assert model.model.embed_tokens.weight.dtype is torch.bfloat16


# ---------------------------------------------------------------------------
# Platform gate
# ---------------------------------------------------------------------------


def test_trn2_is_refused_with_the_prefill_reason():
    """Trn2's CTE kernel cannot take a bf16 activation against fp8 weights, so
    prefill never compiles. Measured for ROW and STATIC alike -- this is a
    platform limit, not a mode choice. It must be a startup error, not a NKI
    assertion ten minutes into graph extraction."""
    from unittest import mock

    from vllm_neuron.model.qwen3_5 import model_fp8

    with mock.patch("torch_neuronx.utils.get_platform_target", return_value="trn2"):
        reason = model_fp8.unsupported_platform_reason()
    assert reason is not None
    assert "prefill" in reason
    assert "trn3" in reason


def test_trn3_is_allowed():
    from unittest import mock

    from vllm_neuron.model.qwen3_5 import model_fp8

    with mock.patch("torch_neuronx.utils.get_platform_target", return_value="trn3"):
        assert model_fp8.unsupported_platform_reason() is None


def test_a_cpu_run_with_no_runtime_is_not_this_functions_business():
    """`get_platform_target` raises without NRT. Nothing compiles on a bare CPU
    run anyway, so the gate must not turn that into an error."""
    from unittest import mock

    from vllm_neuron.model.qwen3_5 import model_fp8

    with mock.patch(
        "torch_neuronx.utils.get_platform_target", side_effect=RuntimeError("no NRT")
    ):
        assert model_fp8.unsupported_platform_reason() is None
