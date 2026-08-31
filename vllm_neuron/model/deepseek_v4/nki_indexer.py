# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B023
"""Opaque BF16 score/top-k kernel for DeepSeek-V4's projected CSA indexer."""

from __future__ import annotations

import math
import os

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from nkilib.core.topk.rotational_topk_utils import topk_core
from torch_neuronx.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

from .indexer import IndexerSelection, streaming_topk_compressed_entries

_SCHEDULER_QUERY_BUCKETS = frozenset(
    (1, 2, 4, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
)
_MAX_RUNTIME_LOOP_TRIPS = 2048
_INDEXER_QUERIES_PER_LNC = 8
_INDEXER_QUERY_TILE = 2 * _INDEXER_QUERIES_PER_LNC
# Largest query bucket routed to the fully unrolled decode kernel.
_STATIC_DECODE_QUERY_MAX = 8
# Trace-time ceiling on unrolled ``queries_per_program * max_pages`` scan
# bodies, so growth in ``max_model_len`` fails loudly here instead of as a
# compiler instruction-count explosion.
_MAX_STATIC_SCAN_BODIES = 256


def _dynamic_page_loop_forced() -> bool:
    """Route decode back through the runtime page loop for A/B attribution.

    The static decode kernel is the default.  Setting
    ``VLLM_NEURON_DSV4_DYNAMIC_PAGE_LOOP=1`` restores the previous
    ``fori_loop`` path verbatim, which is how a device run attributes a change
    in behavior to this kernel rather than to anything else in the graph.
    """
    return os.environ.get("VLLM_NEURON_DSV4_DYNAMIC_PAGE_LOOP", "0") == "1"


def _use_static_decode_kernel(q_count: int) -> bool:
    """Route a padded decode launch to the fully unrolled kernel."""
    return q_count <= _STATIC_DECODE_QUERY_MAX and not _dynamic_page_loop_forced()


def _emit_scan_page(
    page,
    q_t,
    gate_t,
    keys,
    visible_sb,
    used_count,
    running_scores,
    running_indices,
    topk,
    heads,
    head_dim,
    key_dim,
    paged,
    static_page,
    block_table,
    block_valid,
    block_table_row_base,
    physical_page_stride,
    logical_slots_per_block,
):
    """Emit one 512-entry score page's scan, mask and running top-k merge.

    Every parameter is positional: NKI traces this helper along with its
    caller, and its tracer supports neither keyword-only parameters nor
    keyword arguments at the call site.

    Shared by both indexer kernels so the scoring body has exactly one
    definition.  ``page`` is a Python ``int`` when ``static_page`` is set --
    the decode kernel unrolls its page loop at trace time -- and an
    ``nl.fori_loop`` register otherwise.

    This is deliberately a module-level function that emits directly rather
    than a closure factory: NKI allows an inner function only as a
    fori_loop/while_loop body, and rejects calling one directly.
    """

    page_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    page_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    if static_page:
        # ``page`` is a Python int because the caller unrolled the page
        # loop at trace time, so the page index is a compile-time
        # constant and no runtime register is involved.
        nisa.memset(page_sb, value=page)
    else:
        nisa.register_store(page_sb, page)
    nisa.tensor_scalar(
        dst=page_base,
        data=page_sb,
        op0=nl.multiply,
        operand0=512,
    )
    key_t = nl.ndarray((head_dim, 512), dtype=keys.dtype, buffer=nl.sbuf)
    if paged:
        blocks_per_score_page = 512 // logical_slots_per_block
        for page_block in nl.static_range(blocks_per_score_page):
            column_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=column_sb,
                data=page_sb,
                op0=nl.multiply,
                operand0=blocks_per_score_page,
                op1=nl.add,
                operand1=page_block,
            )
            nisa.tensor_tensor(
                dst=column_sb,
                data1=column_sb,
                data2=block_table_row_base,
                op=nl.add,
            )
            column_reg = nisa.register_alloc()
            nisa.register_load(column_reg, column_sb)
            slot_dma_width = min(logical_slots_per_block, 128)
            for slot_chunk in nl.static_range(
                logical_slots_per_block // slot_dma_width
            ):
                slot = nl.ndarray(
                    (slot_dma_width, 1),
                    dtype=nl.uint32,
                    buffer=nl.sbuf,
                )
                nisa.dma_copy(
                    dst=slot,
                    src=block_table.ap(
                        pattern=[[0, slot_dma_width]],
                        scalar_offset=column_reg,
                        indirect_dim=0,
                    ),
                )
                nisa.tensor_scalar(
                    dst=slot,
                    data=slot,
                    op0=nl.multiply,
                    operand0=physical_page_stride,
                    op1=nl.add,
                    operand1=slot_chunk * slot_dma_width,
                )
                start = (
                    page_block * logical_slots_per_block
                    + slot_chunk * slot_dma_width
                )
                nisa.dma_transpose(
                    dst=key_t[:, start : start + slot_dma_width],
                    src=keys.ap(
                        pattern=[
                            [key_dim, slot_dma_width],
                            [1, key_dim],
                        ],
                        vector_offset=slot,
                        indirect_dim=0,
                    ),
                    oob_mode=nisa.oob_mode.skip,
                )
    else:
        nisa.dma_transpose(
            dst=key_t,
            src=keys.ap(
                pattern=[[128, 512], [1, 128]],
                scalar_offset=page_base,
                indirect_dim=0,
            ),
        )
    per_head_psum = nl.ndarray((heads, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(per_head_psum, q_t, key_t)
    per_head = nl.ndarray((heads, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(
        dst=per_head,
        op=nl.relu,
        data=per_head_psum,
        scale=1.0 / math.sqrt(128),
    )
    page_psum = nl.ndarray((1, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(page_psum, gate_t, per_head)
    page_scores = nl.ndarray((1, 512), dtype=nl.float32, buffer=nl.sbuf)
    if paged:
        blocks_per_score_page = 512 // logical_slots_per_block
        for page_block in nl.static_range(blocks_per_score_page):
            column_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=column_sb,
                data=page_sb,
                op0=nl.multiply,
                operand0=blocks_per_score_page,
                op1=nl.add,
                operand1=page_block,
            )
            nisa.tensor_tensor(
                dst=column_sb,
                data1=column_sb,
                data2=block_table_row_base,
                op=nl.add,
            )
            column_reg = nisa.register_alloc()
            nisa.register_load(column_reg, column_sb)
            upper = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            valid_block = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=valid_block,
                src=block_valid.ap(
                    pattern=[[1, 1]],
                    scalar_offset=column_reg,
                    indirect_dim=0,
                ),
            )
            nisa.tensor_tensor(
                dst=upper,
                data1=visible_sb,
                data2=page_base,
                op=nl.subtract,
            )
            nisa.tensor_scalar(
                dst=upper,
                data=upper,
                op0=nl.maximum,
                operand0=0,
            )
            nisa.tensor_scalar(
                dst=upper,
                data=upper,
                op0=nl.subtract,
                operand0=page_block * logical_slots_per_block,
                op1=nl.maximum,
                operand1=0,
            )
            nisa.tensor_scalar(
                dst=upper,
                data=upper,
                op0=nl.minimum,
                operand0=logical_slots_per_block,
            )
            nisa.tensor_tensor(
                dst=upper,
                data1=upper,
                data2=valid_block,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=used_count,
                data1=used_count,
                data2=upper,
                op=nl.add,
            )
            upper_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=upper_f32, src=upper)
            start = page_block * logical_slots_per_block
            nisa.range_select(
                page_scores[:, start : start + logical_slots_per_block],
                on_true_tile=page_psum[
                    :, start : start + logical_slots_per_block
                ],
                on_false_value=-3.4028234663852886e38,
                comp_op0=nl.greater_equal,
                comp_op1=nl.less,
                bound0=0.0,
                bound1=upper_f32,
                range_start=0,
            )
    else:
        page_upper = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=page_upper,
            data1=visible_sb,
            data2=page_base,
            op=nl.subtract,
        )
        nisa.tensor_scalar(
            dst=page_upper,
            data=page_upper,
            op0=nl.maximum,
            operand0=0,
        )
        nisa.tensor_scalar(
            dst=page_upper,
            data=page_upper,
            op0=nl.minimum,
            operand0=512,
        )
        page_upper_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=page_upper_f32, src=page_upper)
        nisa.range_select(
            page_scores,
            on_true_tile=page_psum,
            on_false_value=-3.4028234663852886e38,
            comp_op0=nl.greater_equal,
            comp_op1=nl.less,
            bound0=0.0,
            bound1=page_upper_f32,
            range_start=0,
        )
    nisa.tensor_scalar(
        dst=page_scores,
        data=page_scores,
        op0=nl.multiply,
        operand0=1.0 / math.sqrt(64),
    )

    merged_scores = nl.ndarray((1, 1024), dtype=nl.float32, buffer=nl.sbuf)
    merged_indices = nl.ndarray((1, 1024), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=merged_scores[:, :topk], src=running_scores)
    nisa.tensor_copy(dst=merged_scores[:, topk:], src=page_scores)
    nisa.tensor_copy(dst=merged_indices[:, :topk], src=running_indices)
    page_indices_f32 = nl.ndarray((1, 512), dtype=nl.float32, buffer=nl.sbuf)
    page_base_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=page_indices_f32, pattern=[[1, 512]], offset=0)
    nisa.tensor_copy(dst=page_base_f32, src=page_base)
    nisa.tensor_scalar(
        dst=page_indices_f32,
        data=page_indices_f32,
        op0=nl.add,
        operand0=page_base_f32,
    )
    nisa.tensor_copy(dst=merged_indices[:, topk:], src=page_indices_f32)
    next_scores, offsets = topk_core(merged_scores, topk)
    next_indices = nl.ndarray((1, topk), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nc_n_gather(
        dst=next_indices,
        data=merged_indices,
        indices=offsets,
    )
    nisa.tensor_copy(dst=running_scores, src=next_scores)
    nisa.tensor_copy(dst=running_indices, src=next_indices)


@nki.jit
def _projected_bf16_indexer_kernel(
    query,
    keys,
    gate,
    visible,
    visible_pages,
    topk: int,
    block_table=None,
    block_valid=None,
    token_to_request=None,
    physical_page_stride: int = 0,
    logical_slots_per_block: int = 0,
):
    """Score contiguous BF16 key pages without exposing scores outside NKI."""
    q_count, one, heads, head_dim = query.shape
    assert q_count in (
        1,
        8,
        16,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ), "unsupported indexer query bucket"
    entries, key_dim = keys.shape
    paged = block_table is not None
    if paged:
        assert block_table.shape == block_valid.shape
        assert token_to_request.shape == (q_count,)
        block_columns = block_table.shape[1]
        entries = block_columns * logical_slots_per_block
        flat_table_size = block_table.shape[0] * block_columns
        block_table = block_table.reshape((flat_table_size,))
        block_valid = block_valid.reshape((flat_table_size,))
    assert one == 1 and heads == 64 and head_dim == key_dim == 128
    assert entries % 512 == 0 and topk == 512
    one_page = entries == 512

    output = nl.ndarray((q_count, topk), dtype=nl.int32, buffer=nl.shared_hbm)
    valid_counts = nl.ndarray((q_count,), dtype=nl.int32, buffer=nl.shared_hbm)
    n_programs = nl.num_programs(0)
    program_id = nl.program_id(0)
    queries_per_program = q_count if q_count == 1 else q_count // n_programs
    assert q_count == 1 or q_count % n_programs == 0
    query_flat = query.reshape((q_count * heads * head_dim,))
    gate_flat = gate.reshape((q_count * heads,))
    visible_flat = visible.reshape((q_count,))
    visible_pages_flat = visible_pages.reshape((q_count,))
    output_flat = output.reshape((q_count * topk,))
    valid_counts_flat = valid_counts.reshape((q_count,))

    def process_query(local_q_idx):
        if q_count == 1:
            q_idx = 0
            q_idx_reg = None
        else:
            local_q_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            q_idx_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.register_store(local_q_sb, local_q_idx)
            nisa.tensor_scalar(
                dst=q_idx_sb,
                data=local_q_sb,
                op0=nl.add,
                operand0=program_id * queries_per_program,
            )
            q_idx_reg = nisa.register_alloc()
            nisa.register_load(q_idx_reg, q_idx_sb)
        q_t = nl.ndarray((head_dim, heads), dtype=query.dtype, buffer=nl.sbuf)
        gate_t = nl.ndarray((heads, 1), dtype=gate.dtype, buffer=nl.sbuf)
        if q_count == 1:
            nisa.dma_transpose(dst=q_t, src=query[0, 0, :, :])
            nisa.dma_transpose(dst=gate_t, src=gate[0, 0, :].reshape((1, heads)))
        else:
            query_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            gate_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=query_base,
                data=q_idx_sb,
                op0=nl.multiply,
                operand0=heads * head_dim,
            )
            nisa.tensor_scalar(
                dst=gate_base,
                data=q_idx_sb,
                op0=nl.multiply,
                operand0=heads,
            )
            query_base_reg = nisa.register_alloc()
            gate_base_reg = nisa.register_alloc()
            nisa.register_load(query_base_reg, query_base)
            nisa.register_load(gate_base_reg, gate_base)
            query_row = nl.ndarray(
                (heads, head_dim), dtype=query.dtype, buffer=nl.sbuf
            )
            gate_row = nl.ndarray((1, heads), dtype=gate.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=query_row,
                src=query_flat.ap(
                    pattern=[[head_dim, heads], [1, head_dim]],
                    scalar_offset=query_base_reg,
                    indirect_dim=0,
                ),
            )
            nisa.dma_copy(
                dst=gate_row,
                src=gate_flat.ap(
                    pattern=[[heads, 1], [1, heads]],
                    scalar_offset=gate_base_reg,
                    indirect_dim=0,
                ),
            )
            nisa.dma_transpose(dst=q_t, src=query_row)
            nisa.dma_transpose(dst=gate_t, src=gate_row)

        block_table_row_base = None
        if paged:
            owner_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            if q_count == 1:
                nisa.dma_copy(dst=owner_sb, src=token_to_request[0:1])
            else:
                nisa.dma_copy(
                    dst=owner_sb,
                    src=token_to_request.ap(
                        pattern=[[1, 1]],
                        scalar_offset=q_idx_reg,
                        indirect_dim=0,
                    ),
                )
            block_table_row_base = nl.ndarray(
                (1, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=block_table_row_base,
                data=owner_sb,
                op0=nl.multiply,
                operand0=block_columns,
            )

        running_scores = nl.ndarray((1, topk), dtype=nl.float32, buffer=nl.sbuf)
        running_indices = nl.ndarray((1, topk), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(running_scores, value=float("-inf"))
        nisa.memset(running_indices, value=-1)

        visible_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        visible_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        if q_count == 1:
            nisa.dma_copy(dst=visible_sb, src=visible[0:1])
        else:
            nisa.dma_copy(
                dst=visible_sb,
                src=visible_flat.ap(
                    pattern=[[1, 1]],
                    scalar_offset=q_idx_reg,
                    indirect_dim=0,
                ),
            )
        nisa.tensor_copy(dst=visible_f32, src=visible_sb)
        used_count = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(used_count, value=0)

        page_count_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        if q_count == 1:
            nisa.dma_copy(dst=page_count_sb, src=visible_pages[0:1])
        else:
            nisa.dma_copy(
                dst=page_count_sb,
                src=visible_pages_flat.ap(
                    pattern=[[1, 1]],
                    scalar_offset=q_idx_reg,
                    indirect_dim=0,
                ),
            )
        page_count_reg = nisa.register_alloc()
        nisa.register_load(page_count_reg, page_count_sb)

        def scan_page(page):
            # A one-page capacity pins the index to the constant 0 rather than
            # reading the loop register, which is what keeps the NKI compiler
            # from giving the two LNCs different basic-block counts in the
            # tiny Q8 graph.
            _emit_scan_page(
                0 if one_page else page,
                q_t,
                gate_t,
                keys,
                visible_sb,
                used_count,
                running_scores,
                running_indices,
                topk,
                heads,
                head_dim,
                key_dim,
                paged,
                one_page,
                block_table,
                block_valid,
                block_table_row_base,
                physical_page_stride,
                logical_slots_per_block,
            )

        if one_page:
            nl.fori_loop(0, 1, scan_page)
        else:
            nl.fori_loop(0, page_count_reg, scan_page)
        if q_count == 1:
            nisa.dma_copy(dst=output[0:1, :], src=running_indices)
        else:
            output_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=output_base,
                data=q_idx_sb,
                op0=nl.multiply,
                operand0=topk,
            )
            output_base_reg = nisa.register_alloc()
            nisa.register_load(output_base_reg, output_base)
            nisa.dma_copy(
                dst=output_flat.ap(
                    pattern=[[topk, 1], [1, topk]],
                    scalar_offset=output_base_reg,
                    indirect_dim=0,
                ),
                src=running_indices,
            )
        if not paged:
            nisa.tensor_scalar(
                dst=used_count,
                data=visible_sb,
                op0=nl.minimum,
                operand0=topk,
            )
        if q_count == 1:
            nisa.dma_copy(dst=valid_counts[0:1], src=used_count)
        else:
            nisa.dma_copy(
                dst=valid_counts_flat.ap(
                    pattern=[[1, 1]],
                    scalar_offset=q_idx_reg,
                    indirect_dim=0,
                ),
                src=used_count,
            )

    if queries_per_program > _MAX_RUNTIME_LOOP_TRIPS:
        # Runtime fori loops with 4096 trips stall on Trn2 even though the
        # identical Q4096 specialization (2048 trips per LNC) is healthy.
        # Keep one opaque call, but split each program's work into two bounded
        # loops so the outer graph has no repeated-call dependency to schedule.
        assert queries_per_program == 2 * _MAX_RUNTIME_LOOP_TRIPS

        nl.fori_loop(0, _MAX_RUNTIME_LOOP_TRIPS, process_query)
        nl.fori_loop(
            _MAX_RUNTIME_LOOP_TRIPS,
            2 * _MAX_RUNTIME_LOOP_TRIPS,
            process_query,
        )
    else:
        nl.fori_loop(0, queries_per_program, process_query)
    return output, valid_counts


@nki.jit
def _decode_indexer_kernel(
    query,
    keys,
    gate,
    visible,
    topk: int,
    max_pages: int,
    block_table=None,
    block_valid=None,
    token_to_request=None,
    physical_page_stride: int = 0,
    logical_slots_per_block: int = 0,
):
    """Fully static small-query indexer for batched decode.

    ``_projected_bf16_indexer_kernel`` bounds its page loop with a runtime
    register, which forces every launch to hold the two LNC programs in
    lockstep by data: no zero-trip loops, an identical trip count on every row,
    and padded decode rows carrying the real row's page count.  Decode instead
    unrolls both loops at trace time.  ``max_pages`` is the compiled capacity
    in 512-entry score pages, and scanning past ``visible`` is already inert --
    ``range_select`` fills those pages with -inf and the paged ``used_count``
    accumulation adds zero -- so the static bound changes work only, never
    selection.  The result contains no runtime control flow, so the two LNC
    programs cannot diverge.
    """
    q_count, one, heads, head_dim = query.shape
    assert q_count in (2, 4, 8), "decode indexer expects a small even bucket"
    entries, key_dim = keys.shape
    paged = block_table is not None
    if paged:
        assert block_table.shape == block_valid.shape
        assert token_to_request.shape == (q_count,)
        block_columns = block_table.shape[1]
        entries = block_columns * logical_slots_per_block
        flat_table_size = block_table.shape[0] * block_columns
        block_table = block_table.reshape((flat_table_size,))
        block_valid = block_valid.reshape((flat_table_size,))
    assert one == 1 and heads == 64 and head_dim == key_dim == 128
    assert entries % 512 == 0 and topk == 512
    assert max_pages == entries // 512

    output = nl.ndarray((q_count, topk), dtype=nl.int32, buffer=nl.shared_hbm)
    valid_counts = nl.ndarray((q_count,), dtype=nl.int32, buffer=nl.shared_hbm)
    n_programs = nl.num_programs(0)
    program_id = nl.program_id(0)
    assert q_count % n_programs == 0
    queries_per_program = q_count // n_programs
    assert queries_per_program * max_pages <= _MAX_STATIC_SCAN_BODIES

    for local_q_idx in nl.affine_range(queries_per_program):
        q_idx = program_id * queries_per_program + local_q_idx
        q_t = nl.ndarray((head_dim, heads), dtype=query.dtype, buffer=nl.sbuf)
        gate_t = nl.ndarray((heads, 1), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(dst=q_t, src=query[q_idx, 0, :, :])
        nisa.dma_transpose(dst=gate_t, src=gate[q_idx, 0, :].reshape((1, heads)))

        block_table_row_base = None
        if paged:
            owner_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=owner_sb,
                src=token_to_request[q_idx : q_idx + 1],
            )
            block_table_row_base = nl.ndarray(
                (1, 1), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=block_table_row_base,
                data=owner_sb,
                op0=nl.multiply,
                operand0=block_columns,
            )

        running_scores = nl.ndarray((1, topk), dtype=nl.float32, buffer=nl.sbuf)
        running_indices = nl.ndarray((1, topk), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(running_scores, value=float("-inf"))
        nisa.memset(running_indices, value=-1)

        visible_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=visible_sb, src=visible[q_idx : q_idx + 1])
        used_count = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(used_count, value=0)

        for page in nl.static_range(max_pages):
            _emit_scan_page(
                page,
                q_t,
                gate_t,
                keys,
                visible_sb,
                used_count,
                running_scores,
                running_indices,
                topk,
                heads,
                head_dim,
                key_dim,
                paged,
                True,
                block_table,
                block_valid,
                block_table_row_base,
                physical_page_stride,
                logical_slots_per_block,
            )

        nisa.dma_copy(dst=output[q_idx : q_idx + 1, :], src=running_indices)
        if not paged:
            nisa.tensor_scalar(
                dst=used_count,
                data=visible_sb,
                op0=nl.minimum,
                operand0=topk,
            )
        nisa.dma_copy(dst=valid_counts[q_idx : q_idx + 1], src=used_count)
    return output, valid_counts


_wrapped_projected_indexer = wrap_nki(_projected_bf16_indexer_kernel)
_wrapped_decode_indexer = wrap_nki(_decode_indexer_kernel)


def _query_tiles(q_count: int) -> tuple[tuple[int, int], ...]:
    if q_count not in _SCHEDULER_QUERY_BUCKETS:
        raise RuntimeError(
            "DeepSeek-V4 NKI indexer has unsupported query bucket "
            f"{q_count}; expected one of {sorted(_SCHEDULER_QUERY_BUCKETS)}"
        )
    return ((0, q_count),)


def _visible_page_counts(visible: torch.Tensor, capacity: int) -> torch.Tensor:
    """Per-row page counts for the runtime-loop prefill kernel.

    Both distortions below exist only because that kernel's page bound is a
    runtime register.  Decode no longer calls this: ``_decode_indexer_kernel``
    unrolls its page loop at trace time and takes no page-count tensor at all.
    """
    bounded = visible.to(torch.int32).clamp(min=0, max=capacity)
    # The LNC2 NKI ``fori_loop`` used by the indexer does not safely execute a
    # dynamic zero-trip loop: first-chunk CSA rows with visible==0 deadlock in
    # device execution.  Scan one fully masked page instead. ``visible`` still
    # bounds every score and the kernel still returns used_count==0, so no
    # logical entry becomes valid for those rows.
    counts = torch.div(bounded + 511, 512, rounding_mode="floor").clamp(min=1)
    counts = counts.to(torch.int32)
    if counts.numel() == 1:
        return counts
    # Both LNC programs execute the same NKI body.  A dynamic page loop whose
    # trip count differs between programs stalls when one LNC reaches the
    # top-k/synchronization body after the other has left it.  Scan through the
    # batch maximum in lockstep; each row's own ``visible`` value still masks
    # every extra page, so this changes work only, not selection semantics.
    return counts.max().expand_as(counts).contiguous()


def _pad_single_query_bucket(
    query: torch.Tensor,
    gate: torch.Tensor,
    visible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a Q1 decode launch up to the Q8 decode specialization.

    The launch grid is always 2 -- LNC1 is not a legal launch on this host --
    so a literal Q1 launch would hand both programs the same query and write
    the one output row twice.  Padding gives each program its own rows, and
    zero visibility makes the inactive ones semantically inert.

    The padded rows carry no page-count obligation: the decode kernel's page
    bound is a trace-time constant, so inactive rows cannot desynchronize the
    two programs the way they could under the runtime-loop prefill path.
    """
    if query.shape[0] != 1:
        return query, gate, visible
    pad = _INDEXER_QUERIES_PER_LNC - 1
    return (
        torch.cat((query, torch.zeros_like(query).expand(pad, -1, -1, -1))),
        torch.cat((gate, torch.zeros_like(gate).expand(pad, -1, -1))),
        torch.cat((visible, torch.zeros_like(visible).expand(pad))),
    )


def projected_bf16_indexer(
    query: torch.Tensor,
    keys: torch.Tensor,
    gate: torch.Tensor,
    visible: torch.Tensor,
) -> IndexerSelection:
    """Dispatch the projected official BF16 geometry through one opaque call."""
    if not can_run_kernel(query):
        return streaming_topk_compressed_entries(
            query,
            keys.unsqueeze(0).expand(query.shape[0], -1, -1),
            gate,
            visible[:, None],
            topk=512,
        )
    if query.shape[1:] != (1, 64, 128) or gate.shape != query.shape[:-1]:
        raise RuntimeError("DeepSeek-V4 NKI indexer requires [Q,1,64,128] query")
    if keys.ndim != 2 or keys.shape[1] != 128 or keys.shape[0] % 512:
        raise RuntimeError(
            "DeepSeek-V4 NKI indexer requires [N,128], N multiple of 512"
        )
    if (
        query.dtype != torch.bfloat16
        or keys.dtype != torch.bfloat16
        or gate.dtype != torch.bfloat16
    ):
        raise RuntimeError("DeepSeek-V4 NKI indexer requires BF16 inputs")
    visible_i32 = visible.to(torch.int32).clamp(min=0, max=keys.shape[0])
    parts = []
    for start, stop in _query_tiles(query.shape[0]):
        query_part, gate_part, visible_part = _pad_single_query_bucket(
            query[start:stop],
            gate[start:stop],
            visible_i32[start:stop],
        )
        if _use_static_decode_kernel(query_part.shape[0]):
            part = _wrapped_decode_indexer[2](
                query_part,
                keys,
                gate_part,
                visible_part,
                512,
                keys.shape[0] // 512,
            )
        else:
            visible_pages_part = _visible_page_counts(
                visible_part, keys.shape[0]
            )
            part = _wrapped_projected_indexer[2](
                query_part,
                keys,
                gate_part,
                visible_part,
                visible_pages_part,
                512,
            )
        parts.append((part[0][: stop - start], part[1][: stop - start]))
    indices = torch.cat([part[0] for part in parts], dim=0)
    used = torch.cat([part[1] for part in parts], dim=0)
    valid = torch.arange(512, device=query.device)[None, :] < used[:, None]
    return IndexerSelection(
        torch.where(valid, indices.to(torch.int32), torch.full_like(indices, -1)),
        valid,
    )


def paged_projected_bf16_indexer(
    query: torch.Tensor,
    gate: torch.Tensor,
    key_cache: torch.Tensor,
    block_tables: torch.Tensor,
    token_to_request: torch.Tensor,
    visible: torch.Tensor,
    *,
    logical_slots_per_block: int,
) -> IndexerSelection:
    """Opaque batched paged indexer with explicit request ownership."""
    if (
        query.shape[0] not in _SCHEDULER_QUERY_BUCKETS
        and query.shape[0] != _INDEXER_QUERY_TILE
    ):
        raise RuntimeError(
            "DeepSeek-V4 NKI indexer requires query bucket "
            f"{sorted(_SCHEDULER_QUERY_BUCKETS)} or the Q{_INDEXER_QUERY_TILE} tile; "
            f"got {query.shape[0]}"
        )
    if query.shape[1:] != (1, 64, 128) or gate.shape != query.shape[:-1]:
        raise RuntimeError("DeepSeek-V4 NKI indexer requires [Q,1,64,128] query")
    if key_cache.ndim != 4 or key_cache.shape[1] != 1 or key_cache.shape[-1] != 128:
        raise RuntimeError("DeepSeek-V4 NKI indexer cache must be [blocks,1,page,128]")
    if block_tables.ndim != 2 or block_tables.shape[0] < 1:
        raise RuntimeError(
            "DeepSeek-V4 NKI indexer block tables must be [requests,blocks]"
        )
    if token_to_request.shape != (query.shape[0],):
        raise RuntimeError(
            "DeepSeek-V4 NKI indexer requires one request id per query"
        )
    if visible.shape != (query.shape[0],):
        raise RuntimeError("DeepSeek-V4 NKI indexer visible counts must be [Q]")
    if logical_slots_per_block < 1 or 512 % logical_slots_per_block:
        raise RuntimeError("logical indexer page size must divide 512")
    if key_cache.shape[2] < logical_slots_per_block:
        raise RuntimeError("logical indexer page exceeds its physical cache page")
    if not can_run_kernel(query):
        raise RuntimeError("paged_projected_bf16_indexer is a Neuron-only opaque path")
    if any(t.dtype != torch.bfloat16 for t in (query, gate, key_cache)):
        raise RuntimeError("DeepSeek-V4 NKI indexer requires BF16 inputs")

    capacity = block_tables.shape[1] * logical_slots_per_block
    score_page_columns = 512 // logical_slots_per_block
    padded_columns = (
        math.ceil(block_tables.shape[1] / score_page_columns) * score_page_columns
    )
    if padded_columns != block_tables.shape[1]:
        padding = torch.full(
            (block_tables.shape[0], padded_columns - block_tables.shape[1]),
            -1,
            dtype=block_tables.dtype,
            device=block_tables.device,
        )
        kernel_block_table = torch.cat((block_tables, padding), dim=1)
    else:
        kernel_block_table = block_tables
    block_valid = (kernel_block_table >= 0) & (
        kernel_block_table < key_cache.shape[0]
    )
    safe_table = kernel_block_table.clamp(0, key_cache.shape[0] - 1).to(torch.int32)
    flat_cache = key_cache[:, 0].reshape(-1, 128)
    owner_valid = (token_to_request >= 0) & (
        token_to_request < block_tables.shape[0]
    )
    safe_owners = token_to_request.clamp(0, block_tables.shape[0] - 1).to(
        torch.int32
    )
    visible_i32 = torch.where(
        owner_valid,
        visible.to(torch.int32).clamp(min=0, max=capacity),
        torch.zeros((), dtype=torch.int32, device=visible.device),
    )
    block_valid_i32 = block_valid.to(torch.int32)
    # The kernel derives ``entries`` from the padded table, so the static page
    # bound must be derived the same way rather than from ``capacity``.
    max_pages = (padded_columns * logical_slots_per_block) // 512
    parts = []
    for start, stop in _query_tiles(query.shape[0]):
        query_part, gate_part, visible_part = _pad_single_query_bucket(
            query[start:stop],
            gate[start:stop],
            visible_i32[start:stop],
        )
        owners_part = safe_owners[start:stop]
        if owners_part.shape[0] != query_part.shape[0]:
            owners_part = torch.cat(
                (
                    owners_part,
                    torch.zeros(
                        query_part.shape[0] - owners_part.shape[0],
                        dtype=torch.int32,
                        device=owners_part.device,
                    ),
                )
            )
        if _use_static_decode_kernel(query_part.shape[0]):
            part = _wrapped_decode_indexer[2](
                query_part,
                flat_cache,
                gate_part,
                visible_part,
                512,
                max_pages,
                safe_table,
                block_valid_i32,
                owners_part,
                key_cache.shape[2],
                logical_slots_per_block,
            )
        else:
            visible_pages_part = _visible_page_counts(visible_part, capacity)
            part = _wrapped_projected_indexer[2](
                query_part,
                flat_cache,
                gate_part,
                visible_part,
                visible_pages_part,
                512,
                safe_table,
                block_valid_i32,
                owners_part,
                key_cache.shape[2],
                logical_slots_per_block,
            )
        parts.append((part[0][: stop - start], part[1][: stop - start]))
    indices = torch.cat([part[0] for part in parts], dim=0)
    used = torch.cat([part[1] for part in parts], dim=0)
    used = used.clamp(min=0, max=512)
    valid = torch.arange(512, device=query.device)[None, :] < used[:, None]
    return IndexerSelection(
        torch.where(valid, indices.to(torch.int32), torch.full_like(indices, -1)),
        valid,
    )
