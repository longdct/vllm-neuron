# SPDX-License-Identifier: Apache-2.0
"""Convert model-facing cache declarations to vLLM cache specs.

Targets vLLM 0.24, which already carries the DeepSeek-V4 fields on
``MLAAttentionSpec`` / ``SlidingWindowMLASpec`` (``compress_ratio``,
``alignment``, ``model_version``). The one gap is ``RSWASpec``, added in
0.26 -- see :func:`layer_spec_to_vllm_spec`.
"""

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)

from vllm_neuron.model.kv_cache import CacheKind, LayerSpec


def layer_spec_to_vllm_spec(layer: LayerSpec, block_size: int, dtype):
    block_size = layer.block_size or block_size
    # Attention pages follow cache_config.cache_dtype. Compressor carry is
    # accumulated in fp32 upstream and must not inherit an fp8/bf16 KV choice.
    effective_dtype = (
        layer.dtype if layer.cache_kind is CacheKind.COMPRESSOR_STATE else dtype
    )
    common = dict(
        block_size=block_size,
        num_kv_heads=layer.num_kv_heads,
        head_size=layer.head_size,
        dtype=effective_dtype,
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
    if layer.cache_kind in (
        CacheKind.SLIDING_WINDOW_MLA,
        CacheKind.COMPRESSOR_STATE,
    ):
        return SlidingWindowMLASpec(
            **common,
            sliding_window=layer.sliding_window_size,
            alignment=layer.alignment,
            model_version="deepseek_v4",
        )
    if layer.cache_kind is CacheKind.SLIDING_WINDOW:
        return SlidingWindowSpec(
            **common, sliding_window=layer.sliding_window_size
        )
    if layer.cache_kind is CacheKind.RSWA:
        # ``RSWASpec`` is the one cache spec vLLM 0.24 does not have; it lands in
        # 0.26. ``CacheKind.RSWA`` itself stays -- the synthetic model declares
        # it to exercise heterogeneous layouts, and that path never reaches a
        # vLLM spec. Fail loudly rather than silently degrading to a plain
        # sliding window, which would keep every evicted gap block alive.
        raise NotImplementedError(
            f"layer {layer.name}: R-SWA caches need vLLM's RSWASpec, added in "
            f"vLLM 0.26; this build targets 0.24"
        )
    return FullAttentionSpec(
        **common,
        sliding_window=layer.sliding_window_size,
        attention_chunk_size=layer.chunk_size,
    )
