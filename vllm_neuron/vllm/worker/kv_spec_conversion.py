# SPDX-License-Identifier: Apache-2.0
"""Convert model-facing cache declarations to vLLM 0.26 cache specs."""

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    RSWASpec,
    SlidingWindowSpec,
)

from vllm_neuron.model.kv_cache import CacheKind, LayerSpec


def layer_spec_to_vllm_spec(layer: LayerSpec, block_size: int, dtype):
    common = dict(
        block_size=block_size,
        num_kv_heads=layer.num_kv_heads,
        head_size=layer.head_size,
        dtype=dtype,
    )
    if layer.cache_kind is CacheKind.MLA:
        if block_size % layer.compress_ratio:
            raise ValueError(
                f"layer {layer.name}: block size {block_size} is not divisible "
                f"by compression ratio {layer.compress_ratio}"
            )
        return MLAAttentionSpec(
            **common,
            compress_ratio=layer.compress_ratio,
            alignment=layer.alignment,
            model_version="deepseek_v4",
        )
    if layer.cache_kind in (CacheKind.SLIDING_WINDOW, CacheKind.COMPRESSOR_STATE):
        return SlidingWindowSpec(
            **common, sliding_window=layer.sliding_window_size
        )
    if layer.cache_kind is CacheKind.RSWA:
        return RSWASpec(**common, rswa_window=layer.rswa_window)
    return FullAttentionSpec(
        **common,
        sliding_window=layer.sliding_window_size,
        attention_chunk_size=layer.chunk_size,
    )
