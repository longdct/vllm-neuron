# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    RSWASpec,
    SlidingWindowMLASpec,
)
from vllm_neuron.model.kv_cache import CacheKind, LayerSpec
from vllm_neuron.vllm.worker.kv_spec_conversion import layer_spec_to_vllm_spec


def layer(kind, **kwargs):
    return LayerSpec(
        name=f"layer.{kind}",
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        cache_kind=kind,
        **kwargs,
    )


@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_mla_ratios_preserve_native_block_addressing(ratio):
    spec = layer_spec_to_vllm_spec(
        layer(CacheKind.MLA, compress_ratio=ratio), 128, torch.bfloat16
    )
    assert isinstance(spec, MLAAttentionSpec)
    assert spec.block_size == 128
    assert spec.storage_block_size == 128 // ratio
    assert spec.model_version == "deepseek_v4"


def test_compressor_state_uses_sliding_window_lifecycle():
    spec = layer_spec_to_vllm_spec(
        layer(
            CacheKind.COMPRESSOR_STATE,
            block_size=8,
            sliding_window_size=128,
        ),
        32,
        torch.bfloat16,
    )
    assert type(spec) is SlidingWindowMLASpec
    assert spec.block_size == 8
    assert spec.sliding_window == 128
    assert spec.dtype is torch.bfloat16


def test_compressor_state_preserves_declared_fp32_dtype():
    state = LayerSpec(
        name="state",
        num_kv_heads=1,
        head_size=1024,
        dtype=torch.float32,
        block_size=8,
        cache_kind=CacheKind.COMPRESSOR_STATE,
        sliding_window_size=128,
    )
    spec = layer_spec_to_vllm_spec(state, 128, torch.float8_e4m3fn)
    assert spec.dtype is torch.float32


def test_deepseek_swa_uses_single_tensor_mla_layout():
    spec = layer_spec_to_vllm_spec(
        layer(CacheKind.SLIDING_WINDOW_MLA, sliding_window_size=128),
        32,
        torch.bfloat16,
    )
    assert type(spec) is SlidingWindowMLASpec
    assert spec.head_size == 512


def test_rswa_uses_upstream_manager_marker():
    spec = layer_spec_to_vllm_spec(
        layer(CacheKind.RSWA, rswa_window=256), 32, torch.bfloat16
    )
    assert isinstance(spec, RSWASpec)
    assert spec.rswa_window == 256


def test_compressed_block_must_be_integral():
    with pytest.raises(ValueError, match="not divisible"):
        layer_spec_to_vllm_spec(
            layer(CacheKind.MLA, compress_ratio=128), 32, torch.bfloat16
        )
