# SPDX-License-Identifier: Apache-2.0
"""Convert model-facing cache declarations to vLLM cache specs.

Targets vLLM 0.24, which already carries the DeepSeek-V4 fields on
``MLAAttentionSpec`` / ``SlidingWindowMLASpec`` (``compress_ratio``,
``alignment``, ``model_version``). The one gap is ``RSWASpec``, added in
0.26 -- see :func:`layer_spec_to_vllm_spec`.
"""

import logging
import math
from dataclasses import replace

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)

from vllm_neuron.model.kv_cache import CacheKind, LayerSpec

logger = logging.getLogger(__name__)


def layer_spec_to_vllm_spec(layer: LayerSpec, block_size: int, dtype):
    block_size = layer.block_size or block_size

    if layer.cache_kind is CacheKind.MAMBA:
        # Linear-attention state is per request, not per token: vLLM's
        # MambaManager (registered for MambaSpec in
        # single_type_kv_cache_manager.py) allocates one page per request under
        # the default cache mode, and mamba_block_size defaults to
        # max_model_len so the block table is a single column. The per-request
        # state index is that column; slot_mapping does not apply.
        #
        # dtypes come from the layer, not cache_config: the recurrent state is
        # an fp32 accumulator and must not inherit an fp8/bf16 KV choice.
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

        return MambaSpec(
            block_size=block_size,
            shapes=tuple(tuple(shape) for shape in layer.state_shapes),
            dtypes=tuple(layer.state_dtypes),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        )

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


def align_mamba_pages(specs: dict[str, KVCacheSpec]) -> dict[str, KVCacheSpec]:
    """Pad each Mamba page up to a common multiple of the attention pages.

    A hybrid model declares two cache groups with unrelated page sizes, and
    vLLM unifies them only one of two ways (``kv_cache_utils.py:1069-1095``):
    the largest page is an exact multiple of every smaller page, in which case
    it grows the smaller group's block size by the ratio; or the smaller layer
    is an ``AttentionSpec`` whose backend sets ``indexes_kv_by_block_stride``,
    in which case it pads. This backend sets neither, so a real Qwen3.5
    checkpoint dies before any layer is built::

        NotImplementedError: Layer layers.3.self_attn: page size is not
        divisible by the maximum page size and cannot be padded.

    The recurrent state's size has nothing to do with the attention page --
    it is ``conv_dim`` and ``head_v_dim`` against ``num_kv_heads`` and
    ``block_size`` -- so the two agree only by accident. The tiny bring-up
    fixture had to be built around that accident, and the real 27B misses it
    at every block size (remainder 7680).

    ``MambaSpec`` already carries ``page_size_padded`` for exactly this, so
    round the Mamba page up to a multiple of every attention page. The largest
    page is then the Mamba one and divides all the others, which is the first
    branch above -- no change to vLLM and no new backend capability.

    The padding is unused tail space in each page. Because the runner carves
    the states with the page size as the leading stride
    (``neuron_model_runner.py:8546``, mirroring vLLM's own carving), the tail
    is simply never addressed.
    """
    mamba = {n: s for n, s in specs.items() if isinstance(s, MambaSpec)}
    others = {n: s for n, s in specs.items() if not isinstance(s, MambaSpec)}
    if not mamba or not others:
        return specs

    # A common multiple of every attention page, so the padded Mamba page
    # divides cleanly however heterogeneous the attention group is.
    target = 1
    for spec in others.values():
        target = math.lcm(target, spec.page_size_bytes)

    aligned = dict(specs)
    for name, spec in mamba.items():
        page = spec.page_size_bytes
        if page % target == 0:
            continue
        padded = math.ceil(page / target) * target
        aligned[name] = replace(spec, page_size_padded=padded)
        logger.debug(
            "%s: padding Mamba page %d -> %d bytes to align with the %d-byte "
            "attention page",
            name,
            page,
            padded,
            target,
        )

    changed = [n for n in mamba if aligned[n] is not specs[n]]
    if changed:
        example = aligned[changed[0]]
        waste = example.page_size_bytes - specs[changed[0]].page_size_bytes
        logger.warning(
            "Padded %d Mamba page(s) to %d bytes to satisfy vLLM's hybrid "
            "page unification, costing %d bytes (%.1f%%) of unused tail per "
            "page per request.",
            len(changed),
            example.page_size_bytes,
            waste,
            100.0 * waste / example.page_size_bytes,
        )
    return aligned
