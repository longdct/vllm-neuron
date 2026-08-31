# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for DeepSeek-V4 512-wide shared-latent attention."""

from __future__ import annotations

import math

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from nkilib.core.attention.attention_cte import attention_cte
from torch_neuronx.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

# Keep the large-query kernel ABI deliberately small.  Scheduler buckets larger
# than this are normalized by the Python dispatcher below; adding them here
# would create a distinct (and extremely expensive) NKI specialization.
_DIRECT_QUERY_BUCKETS = frozenset((1, 8, 64, 128, 256, 512, 1024))
_PAGED_KERNEL_QUERY_BUCKETS = frozenset(
    (1, 2, 4, 8, 64, 512, 1024, 2048, 4096, 8192)
)
# Retained only as the compact static Q8 geometry used by simulator coverage.
# Production prefill buckets above Q8 use the runtime query loop directly.
_PREFILL_QUERY_TILE = 8
_MAX_RUNTIME_LOOP_TRIPS = 2048
# Gather each query run's sliding history once instead of once per query. Set to
# 0 to restore the per-query vector-indirect gather, which is what the
# bit-exactness test compares against.
_SPAN_GATHER = True
_SCHEDULER_QUERY_BUCKETS = frozenset(
    (1, 2, 4, 8, 64, 512, 1024, 2048, 4096, 8192)
)
_PREFILL_MICROCHUNK = 1024
# Compiled HCA compressed-entry widths.  model.py picks the smallest one
# that still covers the addressable entry capacity; rounding *up* is what
# keeps every query's requested prefix identical, which is the premise
# `_build_uniform_span` rests on.  Never round down.
_HCA_COUNT_BUCKETS = (32, 64, 128, 256, 512, 1024)
# 128 sliding slots plus each compiled compressed width, plus the widths the
# non-paged `shared_latent_mla` entry point accepts on their own.
_HISTORY_LIMITS = frozenset(
    {128, 512, 1024} | {128 + count for count in _HCA_COUNT_BUCKETS}
)


@nki.jit
def _manual_qk_softmax_stage(query, latent, attention_mask, sinks):
    """Produce unnormalized probabilities and FP32 normalization factors."""
    q_count, one, heads, latent_dim = query.shape
    # LNC shards the query bucket across programs (for example a two-row
    # simulator input becomes one row per LNC2 program), so do not restrict
    # this stage to only the engine's outer bucket sizes.
    assert 1 <= q_count <= 1024, "prefill queries must be sliced to <=1024"
    history = latent.shape[1]
    padded_history = math.ceil(history / 512) * 512
    assert one == 1 and 1 <= heads <= 64 and latent_dim == 512
    assert padded_history <= 1536 and padded_history % 512 == 0
    probs_out = nl.ndarray(
        (q_count, heads, padded_history),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    recip_out = nl.ndarray((q_count, heads, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    n_programs = nl.num_programs(0)
    program_id = nl.program_id(0)
    queries_per_program = q_count if q_count == 1 else q_count // n_programs
    assert q_count == 1 or q_count % n_programs == 0
    for local_q_idx in nl.affine_range(queries_per_program):
        q_idx = (
            local_q_idx
            if q_count == 1
            else program_id * queries_per_program + local_q_idx
        )
        q_tiles = []
        k_tiles = []
        for d_idx in nl.static_range(4):
            q_tile = nl.ndarray((128, heads), dtype=query.dtype, buffer=nl.sbuf)
            nisa.dma_transpose(
                dst=q_tile,
                src=query[q_idx, 0, :, d_idx * 128 : (d_idx + 1) * 128],
            )
            q_tiles.append(q_tile)
            k_tile = nl.ndarray(
                (128, padded_history), dtype=latent.dtype, buffer=nl.sbuf
            )
            nisa.memset(k_tile, value=0)
            nisa.dma_transpose(
                dst=k_tile[:, :history],
                src=latent[q_idx, :, d_idx * 128 : (d_idx + 1) * 128],
            )
            k_tiles.append(k_tile)

        scores_psum = nl.ndarray(
            (heads, padded_history), dtype=nl.float32, buffer=nl.psum
        )
        for k_start in nl.static_range(0, padded_history, 512):
            for d_idx in nl.static_range(4):
                nisa.nc_matmul(
                    scores_psum[:, k_start : k_start + 512],
                    q_tiles[d_idx],
                    k_tiles[d_idx][:, k_start : k_start + 512],
                    accumulate=d_idx > 0,
                )

        mask_heads = nl.ndarray(
            (heads, padded_history), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        nisa.memset(mask_heads, value=float("-inf"))
        nisa.dma_copy(
            dst=mask_heads[:, :history],
            src=attention_mask.ap(
                pattern=[[0, heads], [1, history]], offset=q_idx * history
            ),
        )

        raw_scores_stage = nl.ndarray(
            (heads, padded_history), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=raw_scores_stage,
            data=scores_psum,
            op0=nl.multiply,
            operand0=1.0,
        )
        raw_scores_hbm = nl.ndarray(
            (heads, padded_history), dtype=nl.bfloat16, buffer=nl.private_hbm
        )
        nisa.dma_copy(dst=raw_scores_hbm, src=raw_scores_stage)
        raw_scores = nl.ndarray(
            (heads, padded_history), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=raw_scores, src=raw_scores_hbm)
        masked_scores = nl.ndarray(
            (heads, padded_history), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=masked_scores,
            data1=raw_scores,
            data2=mask_heads,
            op=nl.add,
        )
        neg_score_max = nl.ndarray((heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=neg_score_max,
            op=nl.maximum,
            data=masked_scores,
            axis=1,
            negate=True,
        )
        score_bias = nl.ndarray((heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=score_bias,
            data=neg_score_max,
            op0=nl.multiply,
            operand0=1.0 / math.sqrt(512),
        )
        sink_sb = nl.ndarray((heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=sink_sb, src=sinks.reshape((heads, 1)))
        neg_sink = nl.ndarray((heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_sink, data=sink_sb, op0=nl.multiply, operand0=-1.0)
        neg_max = nl.ndarray((heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=neg_max, data1=score_bias, data2=neg_sink, op=nl.minimum)

        row_sum = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        probs = nl.ndarray((heads, padded_history), dtype=nl.bfloat16, buffer=nl.sbuf)
        for k_start in nl.static_range(0, padded_history, 512):
            nisa.activation(
                dst=probs[:, k_start : k_start + 512],
                op=nl.exp,
                data=masked_scores[:, k_start : k_start + 512],
                bias=neg_max,
                scale=1.0 / math.sqrt(512),
                reduce_op=nl.add,
                reduce_res=row_sum if k_start + 512 == padded_history else None,
                reduce_cmd=(
                    nisa.reduce_cmd.reset_reduce
                    if k_start == 0
                    else nisa.reduce_cmd.reduce
                ),
            )
        # The sink is an unscaled logit and contributes only to normalization.
        sink_shifted = nl.ndarray((heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sink_shifted, data1=sink_sb, data2=neg_max, op=nl.add)
        sink_exp = nl.ndarray((heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=sink_exp, op=nl.exp, data=sink_shifted)
        sink_exp_f32 = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=sink_exp_f32, data=sink_exp, op0=nl.multiply, operand0=1.0
        )
        nisa.tensor_tensor(dst=row_sum, data1=row_sum, data2=sink_exp_f32, op=nl.add)
        recip = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=recip, data=row_sum)
        nisa.dma_copy(dst=probs_out[q_idx, :, :], src=probs)
        nisa.dma_copy(dst=recip_out[q_idx, :, :], src=recip)
    return probs_out, recip_out


@nki.jit
def _manual_pv_stage(query, latent, probs_hbm, recip_hbm):
    q_count, _, heads, latent_dim = query.shape
    history = latent.shape[1]
    padded_history = probs_hbm.shape[2]
    result = nl.ndarray(query.shape, dtype=query.dtype, buffer=nl.shared_hbm)
    n_programs = nl.num_programs(0)
    program_id = nl.program_id(0)
    queries_per_program = q_count if q_count == 1 else q_count // n_programs
    assert q_count == 1 or q_count % n_programs == 0
    for local_q_idx in nl.affine_range(queries_per_program):
        q_idx = (
            local_q_idx
            if q_count == 1
            else program_id * queries_per_program + local_q_idx
        )
        recip = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=recip, src=recip_hbm[q_idx, :, :])
        output_psum = nl.ndarray((heads, latent_dim), dtype=nl.float32, buffer=nl.psum)
        for k_start in nl.static_range(0, padded_history, 128):
            p_t = nl.ndarray((128, heads), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_transpose(
                dst=p_t,
                src=probs_hbm[q_idx, :, k_start : k_start + 128],
            )
            value = nl.ndarray((128, latent_dim), dtype=latent.dtype, buffer=nl.sbuf)
            nisa.memset(value, value=0)
            width = min(128, max(0, history - k_start))
            if width:
                nisa.dma_copy(
                    dst=value[:width, :],
                    src=latent[q_idx, k_start : k_start + width, :],
                )
            nisa.nc_matmul(
                output_psum,
                p_t,
                value,
                accumulate=k_start > 0,
            )
        output = nl.ndarray((heads, latent_dim), dtype=query.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=output, data=output_psum, op0=nl.multiply, operand0=recip
        )
        nisa.dma_copy(dst=result[q_idx, 0, :, :], src=output)
    return result


@nki.jit
def _manual_shared_latent_mla_kernel(query, latent, attention_mask, sinks):
    """Direct 512-wide shared-latent attention with staged allocation scopes."""
    probs, recip = _manual_qk_softmax_stage(query, latent, attention_mask, sinks)
    return _manual_pv_stage(query, latent, probs, recip)


def _gather_rows(flat, slots, q, src_start, width, span, dst_start, latent_dim):
    """Copy ``width`` selected cache rows into a contiguous slice of ``span``."""
    offsets = nl.ndarray((width, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=offsets,
        src=slots[q, src_start : src_start + width].reshape((width, 1)),
    )
    stage = nl.ndarray((width, latent_dim), dtype=span.dtype, buffer=nl.sbuf)
    nisa.memset(stage, value=0.0)
    nisa.dma_copy(
        dst=stage,
        src=flat.ap(
            pattern=[[latent_dim, width], [1, latent_dim]],
            vector_offset=offsets,
            indirect_dim=0,
        ),
        oob_mode=nisa.oob_mode.skip,
    )
    nisa.dma_copy(dst=span[dst_start : dst_start + width, :], src=stage)


def _build_sliding_span(cache, slots, first_q, n_q, latent_dim):
    """Gather the union of one query run's sliding windows exactly once.

    Query ``first_q + i`` attends to ``slots[first_q + i]``, the physical rows
    of ``count`` *consecutive* logical positions, so consecutive queries overlap
    by ``count - 1`` of them. The whole run therefore needs only
    ``count + n_q - 1`` distinct rows: all of ``slots[first_q]`` followed by the
    tail of ``slots[first_q + n_q - 1]``.

    Gathering that once turns ``n_q x count`` per-row DMA descriptors -- which
    the backend ``unroll`` pass expands one per row, and which is the dominant
    whole-model compile cost -- into ``count + n_q - 1`` of them plus one affine
    read per query. The rows and their order are unchanged, so the online
    softmax downstream sees exactly the same inputs.
    """
    count = slots.shape[1]
    assert n_q <= 128, "span tail must fit one vector-offset gather"
    span_rows = count + n_q - 1
    flat = cache.reshape((cache.shape[0] * cache.shape[2], latent_dim))
    span = nl.ndarray(
        (span_rows, latent_dim), dtype=cache.dtype, buffer=nl.private_hbm
    )
    _gather_rows(flat, slots, first_q, 0, count, span, 0, latent_dim)
    if n_q > 1:
        tail = n_q - 1
        _gather_rows(
            flat, slots, first_q + tail, count - tail, tail, span, count, latent_dim
        )
    return span


def _build_uniform_span(cache, slots, last_q, latent_dim):
    """Gather a stream whose window is identical for every query in the run.

    HCA requests a fixed prefix sized from the addressable entry capacity, so
    ``recent_compressed_logical_indices`` yields ``start == 0`` for every query
    and ``slots[q]`` is one row repeated ``Q`` times.  Only validity differs,
    and the mask already carries that.  Gathering ``slots[last_q]`` -- the run's
    longest valid prefix, since positions are non-decreasing within a launch --
    serves every query: an entry valid for an earlier query is valid for the
    last one, and one that is not is masked ``-inf`` and contributes exactly
    zero to the online softmax.

    Cost per query run: ``count`` descriptors for the span plus one affine read
    per query, instead of ``n_q x count`` per-row ones.  See
    docs/model-dev/deepseek-v4-mla-callsite-explosion.md.
    """
    count = slots.shape[1]
    flat = cache.reshape((cache.shape[0] * cache.shape[2], latent_dim))
    span = nl.ndarray((count, latent_dim), dtype=cache.dtype, buffer=nl.private_hbm)
    # `vector_offset` is per-partition and so caps at 128 rows per gather;
    # `_build_sliding_span` gets away with a single `_gather_rows` only because
    # its count is exactly 128.
    for start in nl.static_range(0, count, 128):
        width = min(128, count - start)
        _gather_rows(flat, slots, last_q, start, width, span, start, latent_dim)
    return span


def _stream_paged_query(
    query,
    streams,
    sinks,
    q_idx,
    result,
    dynamic_q: bool = False,
    program_query_base: int = 0,
):
    """Stream 128-key cache tiles through FP32 online-softmax state."""
    q_count, _, heads, latent_dim = query.shape
    q_idx_sb = None
    q_idx_reg = None
    query_row = None
    if dynamic_q:
        local_q_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        q_idx_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.register_store(local_q_sb, q_idx)
        nisa.tensor_scalar(
            dst=q_idx_sb,
            data=local_q_sb,
            op0=nl.add,
            operand0=program_query_base,
        )
        q_idx_reg = nisa.register_alloc()
        nisa.register_load(q_idx_reg, q_idx_sb)
        query_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=query_base,
            data=q_idx_sb,
            op0=nl.multiply,
            operand0=heads * latent_dim,
        )
        query_base_reg = nisa.register_alloc()
        nisa.register_load(query_base_reg, query_base)
        query_row = nl.ndarray(
            (heads, latent_dim), dtype=query.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=query_row,
            src=query.reshape((q_count * heads * latent_dim,)).ap(
                pattern=[[latent_dim, heads], [1, latent_dim]],
                scalar_offset=query_base_reg,
                indirect_dim=0,
            ),
        )
    q_tiles = []
    for d_idx in nl.static_range(4):
        q_tile = nl.ndarray((128, heads), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(
            dst=q_tile,
            src=(
                query_row[:, d_idx * 128 : (d_idx + 1) * 128]
                if dynamic_q
                else query[q_idx, 0, :, d_idx * 128 : (d_idx + 1) * 128]
            ),
        )
        q_tiles.append(q_tile)

    sink_sb = nl.ndarray((heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=sink_sb, src=sinks.reshape((heads, 1)))
    running_max = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=running_max, data=sink_sb, op0=nl.multiply, operand0=1.0
    )
    running_sum = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(running_sum, value=1.0)
    running_output = nl.ndarray(
        (heads, latent_dim), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.memset(running_output, value=0.0)

    # Tuple order is significant: compressed history precedes sliding history.
    for stream in streams:
        source = stream[0]
        slots = stream[1]
        mask = stream[2]
        span_base = stream[3]
        count = slots.shape[1]
        flat = (
            None
            if span_base is not None
            else source.reshape((source.shape[0] * source.shape[2], latent_dim))
        )
        for start in nl.static_range(0, count, 128):
            width = min(128, count - start)
            value = nl.ndarray((128, latent_dim), dtype=source.dtype, buffer=nl.sbuf)
            nisa.memset(value, value=0.0)
            if span_base is None:
                offsets = nl.ndarray((width, 1), dtype=nl.int32, buffer=nl.sbuf)
                if dynamic_q:
                    slot_base = nl.ndarray(
                        (1, 1), dtype=nl.int32, buffer=nl.sbuf
                    )
                    nisa.tensor_scalar(
                        dst=slot_base,
                        data=q_idx_sb,
                        op0=nl.multiply,
                        operand0=count,
                        op1=nl.add,
                        operand1=start,
                    )
                    slot_base_reg = nisa.register_alloc()
                    nisa.register_load(slot_base_reg, slot_base)
                    nisa.dma_copy(
                        dst=offsets,
                        src=slots.reshape((q_count * count,)).ap(
                            pattern=[[1, width], [1, 1]],
                            scalar_offset=slot_base_reg,
                            indirect_dim=0,
                        ),
                    )
                else:
                    nisa.dma_copy(
                        dst=offsets,
                        src=slots[q_idx, start : start + width].reshape(
                            (width, 1)
                        ),
                    )
                # NOTE: the backend `unroll` pass expands this vector-indirect
                # gather into one DMA descriptor per gathered row (Q x history of
                # them), which is the dominant whole-model compile cost.  Dropping
                # oob_mode.skip was measured and changes nothing.  Only CSA still
                # arrives here: streams whose windows are contiguous runs of
                # logical positions (sliding) or identical across the run (HCA)
                # take the span path below instead.  See
                # docs/model-dev/deepseek-v4-mla-callsite-explosion.md.
                nisa.dma_copy(
                    dst=value[:width, :],
                    src=flat.ap(
                        pattern=[[latent_dim, width], [1, latent_dim]],
                        vector_offset=offsets,
                        indirect_dim=0,
                    ),
                    oob_mode=nisa.oob_mode.skip,
                )
            else:
                # The rows this query needs are already contiguous inside the
                # run's span, and `q_idx` is a trace-time constant (the kernel is
                # traced once per LNC program), so this is one affine descriptor.
                row = span_base + start
                nisa.dma_copy(dst=value[:width, :], src=source[row : row + width, :])

            scores_psum = nl.ndarray((heads, 128), dtype=nl.float32, buffer=nl.psum)
            for d_idx in nl.static_range(4):
                key_tile = nl.ndarray(
                    (128, 128), dtype=source.dtype, buffer=nl.sbuf
                )
                nisa.dma_transpose(
                    dst=key_tile,
                    src=value[:, d_idx * 128 : (d_idx + 1) * 128],
                )
                nisa.nc_matmul(
                    scores_psum,
                    q_tiles[d_idx],
                    key_tile,
                    accumulate=d_idx > 0,
                )
            scores = nl.ndarray((heads, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=scores,
                data=scores_psum,
                op0=nl.multiply,
                operand0=1.0,
            )
            tile_mask = nl.ndarray((heads, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(tile_mask, value=float("-inf"))
            if dynamic_q:
                mask_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=mask_base,
                    data=q_idx_sb,
                    op0=nl.multiply,
                    operand0=count,
                    op1=nl.add,
                    operand1=start,
                )
                mask_base_reg = nisa.register_alloc()
                nisa.register_load(mask_base_reg, mask_base)
                nisa.dma_copy(
                    dst=tile_mask[:, :width],
                    src=mask.reshape((q_count * count,)).ap(
                        pattern=[[0, heads], [1, width]],
                        scalar_offset=mask_base_reg,
                        indirect_dim=0,
                    ),
                )
            else:
                nisa.dma_copy(
                    dst=tile_mask[:, :width],
                    src=mask.ap(
                        pattern=[[0, heads], [1, width]],
                        offset=q_idx * count + start,
                    ),
                )
            masked_scores = nl.ndarray(
                (heads, 128), dtype=nl.bfloat16, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=masked_scores, data1=scores, data2=tile_mask, op=nl.add
            )
            neg_tile_max = nl.ndarray((heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=neg_tile_max,
                op=nl.maximum,
                data=masked_scores,
                axis=1,
                negate=True,
            )
            neg_scaled_tile_max = nl.ndarray(
                (heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=neg_scaled_tile_max,
                data=neg_tile_max,
                op0=nl.multiply,
                operand0=1.0 / math.sqrt(512),
            )
            neg_scaled_tile_max_f32 = nl.ndarray(
                (heads, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=neg_scaled_tile_max_f32,
                data=neg_scaled_tile_max,
                op0=nl.multiply,
                operand0=1.0,
            )
            neg_running_max = nl.ndarray(
                (heads, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=neg_running_max,
                data=running_max,
                op0=nl.multiply,
                operand0=-1.0,
            )
            neg_merged_max = nl.ndarray(
                (heads, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=neg_merged_max,
                data1=neg_running_max,
                data2=neg_scaled_tile_max_f32,
                op=nl.minimum,
            )
            merged_max = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=merged_max,
                data=neg_merged_max,
                op0=nl.multiply,
                operand0=-1.0,
            )
            prior_shift = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=prior_shift, data1=running_max, data2=neg_merged_max, op=nl.add
            )
            prior_scale = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=prior_scale, op=nl.exp, data=prior_shift)
            neg_merged_max_bf16 = nl.ndarray(
                (heads, 1), dtype=nl.bfloat16, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=neg_merged_max_bf16,
                data=neg_merged_max,
                op0=nl.multiply,
                operand0=1.0,
            )
            probs = nl.ndarray((heads, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            tile_sum = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=probs,
                op=nl.exp,
                data=masked_scores,
                bias=neg_merged_max_bf16,
                scale=1.0 / math.sqrt(512),
                reduce_op=nl.add,
                reduce_res=tile_sum,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )
            scaled_sum = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=scaled_sum,
                data=running_sum,
                op0=nl.multiply,
                operand0=prior_scale,
            )
            nisa.tensor_tensor(
                dst=running_sum, data1=scaled_sum, data2=tile_sum, op=nl.add
            )
            scaled_output = nl.ndarray(
                (heads, latent_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=scaled_output,
                data=running_output,
                op0=nl.multiply,
                operand0=prior_scale,
            )
            probs_t = nl.ndarray((128, heads), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_transpose(dst=probs_t, src=probs)
            tile_output = nl.ndarray(
                (heads, latent_dim), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(tile_output, probs_t, value)
            tile_output_sb = nl.ndarray(
                (heads, latent_dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=tile_output_sb,
                data=tile_output,
                op0=nl.multiply,
                operand0=1.0,
            )
            nisa.tensor_tensor(
                dst=running_output,
                data1=scaled_output,
                data2=tile_output_sb,
                op=nl.add,
            )
            nisa.tensor_scalar(
                dst=running_max, data=merged_max, op0=nl.multiply, operand0=1.0
            )

    reciprocal = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=reciprocal, data=running_sum)
    output = nl.ndarray((heads, latent_dim), dtype=query.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=output, data=running_output, op0=nl.multiply, operand0=reciprocal
    )
    if dynamic_q:
        result_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=result_base,
            data=q_idx_sb,
            op0=nl.multiply,
            operand0=heads * latent_dim,
        )
        result_base_reg = nisa.register_alloc()
        nisa.register_load(result_base_reg, result_base)
        nisa.dma_copy(
            dst=result.reshape((q_count * heads * latent_dim,)).ap(
                pattern=[[latent_dim, heads], [1, latent_dim]],
                scalar_offset=result_base_reg,
                indirect_dim=0,
            ),
            src=output,
        )
    else:
        nisa.dma_copy(dst=result[q_idx, 0, :, :], src=output)


@nki.jit
def _paged_shared_latent_mla_kernel(
    query,
    sliding_cache,
    sliding_slots,
    sliding_mask,
    compressed_cache,
    compressed_slots,
    compressed_mask,
    sinks,
    sliding_contiguous: bool = True,
    compressed_uniform: bool = False,
):
    q_count = query.shape[0]
    assert q_count in (1, 2, 4, 8, 64, 512, 1024, 2048, 4096, 8192)
    assert sliding_slots.shape[1] == 128
    assert compressed_slots.shape[1] in _HCA_COUNT_BUCKETS
    latent_dim = query.shape[3]
    result = nl.ndarray(query.shape, dtype=query.dtype, buffer=nl.shared_hbm)
    n_programs = nl.num_programs(0)
    program_id = nl.program_id(0)
    queries_per_program = q_count if q_count == 1 else q_count // n_programs
    if q_count > 8:
        program_query_base = program_id * queries_per_program

        def process_query(local_q_idx):
            _stream_paged_query(
                query,
                (
                    (
                        compressed_cache,
                        compressed_slots,
                        compressed_mask,
                        None,
                    ),
                    (sliding_cache, sliding_slots, sliding_mask, None),
                ),
                sinks,
                local_q_idx,
                result,
                True,
                program_query_base,
            )

        if queries_per_program > _MAX_RUNTIME_LOOP_TRIPS:
            assert queries_per_program == 2 * _MAX_RUNTIME_LOOP_TRIPS
            nl.fori_loop(0, _MAX_RUNTIME_LOOP_TRIPS, process_query)
            nl.fori_loop(
                _MAX_RUNTIME_LOOP_TRIPS,
                2 * _MAX_RUNTIME_LOOP_TRIPS,
                process_query,
            )
        else:
            nl.fori_loop(0, queries_per_program, process_query)
        return result

    first_q = 0 if q_count == 1 else program_id * queries_per_program
    sliding_span = None
    compressed_span = None
    if _SPAN_GATHER and sliding_contiguous:
        sliding_span = _build_sliding_span(
            sliding_cache, sliding_slots, first_q, queries_per_program, latent_dim
        )
        # `compressed_uniform` is a trace-time Python bool, so this produces a
        # second NEFF specialization rather than a runtime branch.  It is needed
        # because HCA's capacity-derived count and CSA's `index_topk` are both
        # 512 at 64K context, so shape alone cannot tell the two apart.
        if compressed_uniform:
            compressed_span = _build_uniform_span(
                compressed_cache,
                compressed_slots,
                first_q + queries_per_program - 1,
                latent_dim,
            )
    for local_q_idx in nl.affine_range(queries_per_program):
        q_idx = (
            local_q_idx
            if q_count == 1
            else program_id * queries_per_program + local_q_idx
        )
        _stream_paged_query(
            query,
            (
                (
                    compressed_cache if compressed_span is None else compressed_span,
                    compressed_slots,
                    compressed_mask,
                    None if compressed_span is None else 0,
                ),
                (
                    sliding_cache if sliding_span is None else sliding_span,
                    sliding_slots,
                    sliding_mask,
                    None if sliding_span is None else local_q_idx,
                ),
            ),
            sinks,
            q_idx,
            result,
        )
    return result


@nki.jit
def _paged_sliding_latent_mla_kernel(
    query,
    sliding_cache,
    sliding_slots,
    sliding_mask,
    sinks,
    sliding_contiguous: bool = True,
):
    q_count, history = sliding_slots.shape
    assert q_count in (1, 2, 4, 8, 64, 512, 1024, 2048, 4096, 8192)
    assert history == 128
    latent_dim = query.shape[3]
    result = nl.ndarray(query.shape, dtype=query.dtype, buffer=nl.shared_hbm)
    n_programs = nl.num_programs(0)
    program_id = nl.program_id(0)
    queries_per_program = q_count if q_count == 1 else q_count // n_programs
    if q_count > 8:
        program_query_base = program_id * queries_per_program

        def process_query(local_q_idx):
            _stream_paged_query(
                query,
                ((sliding_cache, sliding_slots, sliding_mask, None),),
                sinks,
                local_q_idx,
                result,
                True,
                program_query_base,
            )

        if queries_per_program > _MAX_RUNTIME_LOOP_TRIPS:
            assert queries_per_program == 2 * _MAX_RUNTIME_LOOP_TRIPS
            nl.fori_loop(0, _MAX_RUNTIME_LOOP_TRIPS, process_query)
            nl.fori_loop(
                _MAX_RUNTIME_LOOP_TRIPS,
                2 * _MAX_RUNTIME_LOOP_TRIPS,
                process_query,
            )
        else:
            nl.fori_loop(0, queries_per_program, process_query)
        return result

    first_q = 0 if q_count == 1 else program_id * queries_per_program
    sliding_span = None
    if _SPAN_GATHER and sliding_contiguous:
        sliding_span = _build_sliding_span(
            sliding_cache, sliding_slots, first_q, queries_per_program, latent_dim
        )
    for local_q_idx in nl.affine_range(queries_per_program):
        q_idx = (
            local_q_idx
            if q_count == 1
            else program_id * queries_per_program + local_q_idx
        )
        _stream_paged_query(
            query,
            (
                (
                    sliding_cache if sliding_span is None else sliding_span,
                    sliding_slots,
                    sliding_mask,
                    None if sliding_span is None else local_q_idx,
                ),
            ),
            sinks,
            q_idx,
            result,
        )
    return result


_wrapped_manual_shared_latent_mla = wrap_nki(
    nki.jit()(_manual_shared_latent_mla_kernel)
)
_wrapped_paged_shared_latent_mla = wrap_nki(nki.jit()(_paged_shared_latent_mla_kernel))
_wrapped_paged_sliding_latent_mla = wrap_nki(
    nki.jit()(_paged_sliding_latent_mla_kernel)
)


def shared_latent_mla(
    query: torch.Tensor,
    latent: torch.Tensor,
    validity: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Opaque decode shared-latent MLA for fixed, bounded selected history.

    Query rows are flattened into the kernel batch and the single shared
    latent sequence is consumed through native GQA. Valid entries must be a
    prefix on Neuron, allowing ``bound_max`` to mask padding without exposing
    a query-by-history score tensor in FX.
    """
    if query.ndim != 4 or query.shape[1] != 1:
        raise ValueError("query must have shape [queries,1,heads,512]")
    if latent.ndim != 3 or latent.shape[0] != query.shape[0]:
        raise ValueError("latent must have shape [queries,history,512]")
    if validity.shape != latent.shape[:2]:
        raise ValueError("validity must have shape [queries,history]")
    if sinks.shape != (query.shape[2],):
        raise ValueError("sinks must have shape [heads]")

    if not can_run_kernel(query):
        from .attention import shared_latent_attention

        return shared_latent_attention(
            query, latent, visibility=validity, attention_sinks=sinks
        )

    if query.shape[-1] != 512 or latent.shape[-1] != 512:
        raise ValueError("DeepSeek-V4 NKI MLA requires latent dimension 512")

    q_count = query.shape[0]
    if q_count > _PREFILL_MICROCHUNK:
        if q_count not in _SCHEDULER_QUERY_BUCKETS or q_count % _PREFILL_MICROCHUNK:
            raise RuntimeError(
                "DeepSeek-V4 NKI MLA has unsupported query bucket "
                f"{q_count}; expected one of {sorted(_SCHEDULER_QUERY_BUCKETS)}"
            )
        return torch.cat(
            [
                shared_latent_mla(
                    query[start : start + _PREFILL_MICROCHUNK],
                    latent[start : start + _PREFILL_MICROCHUNK],
                    validity[start : start + _PREFILL_MICROCHUNK],
                    sinks,
                )
                for start in range(0, q_count, _PREFILL_MICROCHUNK)
            ],
            dim=0,
        )
    if q_count not in _DIRECT_QUERY_BUCKETS:
        raise RuntimeError(
            "DeepSeek-V4 NKI MLA has unsupported query bucket "
            f"{q_count}; expected one of {sorted(_DIRECT_QUERY_BUCKETS)}"
        )
    if latent.shape[1] not in _HISTORY_LIMITS:
        raise RuntimeError(
            "DeepSeek-V4 NKI MLA history geometry is unsupported: "
            f"{latent.shape[1]}; expected one of {sorted(_HISTORY_LIMITS)}"
        )
    if query.dtype != torch.bfloat16 or latent.dtype != torch.bfloat16:
        raise RuntimeError("DeepSeek-V4 NKI MLA requires BF16 query and cache")
    attention_mask = torch.where(
        validity,
        torch.zeros((), dtype=torch.bfloat16, device=validity.device),
        torch.full((), float("-inf"), dtype=torch.bfloat16, device=validity.device),
    )
    return _wrapped_manual_shared_latent_mla[2](
        query,
        latent,
        attention_mask,
        sinks.to(torch.bfloat16),
    )


def paged_shared_latent_mla(inputs) -> torch.Tensor:
    """Attend to separate bounded paged-cache streams through one opaque call."""
    from .attention import gather_bounded_paged_latent, shared_latent_attention

    query = inputs.query
    if inputs.sliding_slots.shape != inputs.sliding_valid.shape:
        raise ValueError("sliding slots and validity must have matching [Q,K] shapes")
    if inputs.sliding_slots.shape[0] != query.shape[0]:
        raise ValueError("sliding slots must contain one row per query")
    has_compressed = inputs.compressed_cache is not None
    if has_compressed != all(
        value is not None
        for value in (
            inputs.compressed_slots,
            inputs.compressed_valid,
        )
    ):
        raise ValueError(
            "compressed cache, slots, and validity must be supplied together"
        )
    if inputs.compressed_uniform and not has_compressed:
        raise ValueError("compressed_uniform requires a compressed stream")

    q_count = query.shape[0]

    if not can_run_kernel(query):
        sliding, sliding_valid = gather_bounded_paged_latent(
            inputs.sliding_cache, inputs.sliding_slots, inputs.sliding_valid
        )
        histories = [sliding]
        validities = [sliding_valid]
        if has_compressed:
            compressed, compressed_valid = gather_bounded_paged_latent(
                inputs.compressed_cache,
                inputs.compressed_slots,
                inputs.compressed_valid,
            )
            histories.insert(0, compressed)
            validities.insert(0, compressed_valid)
        return shared_latent_attention(
            query,
            torch.cat(histories, dim=1),
            visibility=torch.cat(validities, dim=1),
            attention_sinks=inputs.sinks,
        )

    if inputs.sliding_slots.shape[1] != 128:
        raise RuntimeError("DeepSeek-V4 paged MLA requires 128 sliding slots")
    compressed_cache = inputs.compressed_cache if has_compressed else None
    compressed_slots = inputs.compressed_slots if has_compressed else None
    compressed_valid = inputs.compressed_valid if has_compressed else None
    history = inputs.sliding_slots.shape[1] + (
        compressed_slots.shape[1] if has_compressed else 0
    )
    if history not in _HISTORY_LIMITS:
        raise RuntimeError(
            f"DeepSeek-V4 paged MLA history geometry is unsupported: {history}"
        )
    if (
        query.shape[0] not in _SCHEDULER_QUERY_BUCKETS
        or query.shape[1] != 1
        or not 1 <= query.shape[2] <= 64
        or query.shape[3] != 512
    ):
        raise RuntimeError(
            "DeepSeek-V4 paged MLA requires [Q,1,H,512] query with 1 <= H <= 64"
        )
    if query.dtype != torch.bfloat16:
        raise RuntimeError("DeepSeek-V4 paged MLA requires BF16 query")
    caches = [("sliding", inputs.sliding_cache)]
    if has_compressed:
        caches.append(("compressed", compressed_cache))
    for name, cache in caches:
        if cache.ndim != 4 or cache.shape[1] != 1 or cache.shape[-1] != 512:
            raise RuntimeError(f"{name} cache must have shape [blocks,1,page,512]")
        if cache.dtype != torch.bfloat16:
            raise RuntimeError(f"{name} cache must be BF16")

    def safe_slots(cache, slots, valid):
        capacity = cache.shape[0] * cache.shape[2]
        address_valid = valid & (slots >= 0) & (slots < capacity)
        return torch.where(address_valid, slots, torch.zeros_like(slots)).to(
            torch.int32
        ), address_valid

    sliding_slots, sliding_valid = safe_slots(
        inputs.sliding_cache, inputs.sliding_slots, inputs.sliding_valid
    )
    zero = torch.zeros((), dtype=torch.bfloat16, device=query.device)
    neg_inf = torch.full((), float("-inf"), dtype=torch.bfloat16, device=query.device)
    sliding_mask = torch.where(sliding_valid, zero, neg_inf)
    compressed_mask = None
    if has_compressed:
        compressed_slots, compressed_valid = safe_slots(
            compressed_cache, compressed_slots, compressed_valid
        )
        compressed_mask = torch.where(compressed_valid, zero, neg_inf)

    def launch(start: int, stop: int) -> torch.Tensor:
        tiled_query = query[start:stop]
        if tiled_query.shape[0] not in _PAGED_KERNEL_QUERY_BUCKETS:
            raise AssertionError("internal paged MLA launch must be Q1 or the tile")
        if not has_compressed:
            return _wrapped_paged_sliding_latent_mla[2](
                tiled_query,
                inputs.sliding_cache,
                sliding_slots[start:stop],
                sliding_mask[start:stop],
                inputs.sinks.to(torch.bfloat16),
                inputs.sliding_contiguous,
            )
        return _wrapped_paged_shared_latent_mla[2](
            tiled_query,
            inputs.sliding_cache,
            sliding_slots[start:stop],
            sliding_mask[start:stop],
            compressed_cache,
            compressed_slots[start:stop],
            compressed_mask[start:stop],
            inputs.sinks.to(torch.bfloat16),
            inputs.sliding_contiguous,
            inputs.compressed_uniform,
        )

    return launch(0, q_count)


def torch_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(
        query.shape[-1]
    )
    if causal:
        q_len, kv_len = query.shape[-2], key.shape[-2]
        qpos = torch.arange(kv_len - q_len, kv_len, device=query.device)[:, None]
        kpos = torch.arange(kv_len, device=query.device)[None, :]
        scores = scores.masked_fill(~(kpos <= qpos)[None], float("-inf"))
    return torch.matmul(scores.softmax(-1), value.float()).to(query.dtype)


def simulate_512_mla(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Execute nkilib's four-tile 512-d attention through the CPU simulator."""
    if query.shape[-1] != 512 or key.shape[-1] != 512 or value.shape[-1] != 512:
        raise ValueError("P2 NKI prototype requires a 512-wide Q/K/V materialization")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key and value must have shape [batch, sequence, 512]")
    if key.shape != value.shape or query.shape[0] != key.shape[0]:
        raise ValueError("NKI MLA batch and context dimensions do not agree")
    return nki.simulate(attention_cte[1])(
        q=query.contiguous(),
        k=key.transpose(1, 2).contiguous(),
        v=value.contiguous(),
        scale=1 / math.sqrt(512),
        causal_mask=causal,
        tp_q=True,
        tp_k=False,
        tp_out=False,
    )
