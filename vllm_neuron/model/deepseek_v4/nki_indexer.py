# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B023
"""Opaque BF16 score/top-k kernel for DeepSeek-V4's projected CSA indexer."""

from __future__ import annotations

import math

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from nkilib.core.topk.rotational_topk_utils import topk_core
from torch_neuronx.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

from .indexer import IndexerSelection, streaming_topk_compressed_entries

_SCHEDULER_QUERY_BUCKETS = frozenset((1, 512, 1024, 2048, 4096))
_PREFILL_MICROCHUNK = 1024
_INDEXER_QUERIES_PER_LNC = 8
_INDEXER_QUERY_TILE = 2 * _INDEXER_QUERIES_PER_LNC


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
    physical_page_stride: int = 0,
    logical_slots_per_block: int = 0,
):
    """Score contiguous BF16 key pages without exposing scores outside NKI."""
    q_count, one, heads, head_dim = query.shape
    assert q_count in (1, 16), "indexer queries must be sliced to Q1 or Q16"
    entries, key_dim = keys.shape
    paged = block_table is not None
    if paged:
        entries = block_table.shape[0] * logical_slots_per_block
    assert one == 1 and heads == 64 and head_dim == key_dim == 128
    assert entries % 512 == 0 and topk == 512

    output = nl.ndarray((q_count, topk), dtype=nl.int32, buffer=nl.shared_hbm)
    valid_counts = nl.ndarray((q_count,), dtype=nl.int32, buffer=nl.shared_hbm)
    n_programs = nl.num_programs(0)
    program_id = nl.program_id(0)
    queries_per_program = q_count if q_count == 1 else _INDEXER_QUERIES_PER_LNC
    assert q_count == 1 or q_count % n_programs == 0
    for local_q_idx in nl.affine_range(queries_per_program):
        q_idx = (
            local_q_idx
            if q_count == 1
            else program_id * queries_per_program + local_q_idx
        )
        q_t = nl.ndarray((head_dim, heads), dtype=query.dtype, buffer=nl.sbuf)
        gate_t = nl.ndarray((heads, 1), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(dst=q_t, src=query[q_idx, 0, :, :])
        nisa.dma_transpose(dst=gate_t, src=gate[q_idx, 0, :].reshape((1, heads)))

        running_scores = nl.ndarray((1, topk), dtype=nl.float32, buffer=nl.sbuf)
        running_indices = nl.ndarray((1, topk), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(running_scores, value=float("-inf"))
        nisa.memset(running_indices, value=-1)

        visible_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        visible_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=visible_sb, src=visible[q_idx : q_idx + 1])
        nisa.tensor_copy(dst=visible_f32, src=visible_sb)
        used_count = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(used_count, value=0)

        page_count_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=page_count_sb, src=visible_pages[q_idx : q_idx + 1])
        page_count_reg = nisa.register_alloc()
        nisa.register_load(page_count_reg, page_count_sb)

        def scan_page(page):
            page_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            page_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
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
                    column_reg = nisa.register_alloc()
                    nisa.register_load(column_reg, column_sb)
                    slot = nl.ndarray(
                        (logical_slots_per_block, 1),
                        dtype=nl.uint32,
                        buffer=nl.sbuf,
                    )
                    nisa.dma_copy(
                        dst=slot,
                        src=block_table.ap(
                            pattern=[[0, logical_slots_per_block]],
                            scalar_offset=column_reg,
                            indirect_dim=0,
                        ),
                    )
                    nisa.tensor_scalar(
                        dst=slot,
                        data=slot,
                        op0=nl.multiply,
                        operand0=physical_page_stride,
                    )
                    nisa.dma_transpose(
                        dst=key_t[
                            :,
                            page_block * logical_slots_per_block : (page_block + 1)
                            * logical_slots_per_block,
                        ],
                        src=keys.ap(
                            pattern=[
                                [key_dim, logical_slots_per_block],
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

        nl.fori_loop(0, page_count_reg, scan_page)
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


def _query_tiles(q_count: int) -> tuple[tuple[int, int], ...]:
    if q_count == 1:
        return ((0, 1),)
    if q_count == _INDEXER_QUERY_TILE:
        return ((0, _INDEXER_QUERY_TILE),)
    if q_count not in _SCHEDULER_QUERY_BUCKETS:
        raise RuntimeError(
            "DeepSeek-V4 NKI indexer has unsupported query bucket "
            f"{q_count}; expected one of {sorted(_SCHEDULER_QUERY_BUCKETS)}"
        )
    if q_count % _INDEXER_QUERY_TILE:
        raise RuntimeError(
            f"DeepSeek-V4 NKI indexer Q{q_count} cannot be tiled by "
            f"Q{_INDEXER_QUERY_TILE}"
        )
    return tuple(
        (start, start + _INDEXER_QUERY_TILE)
        for start in range(0, q_count, _INDEXER_QUERY_TILE)
    )


def _visible_page_counts(visible: torch.Tensor, capacity: int) -> torch.Tensor:
    bounded = visible.to(torch.int32).clamp(min=0, max=capacity)
    return torch.div(bounded + 511, 512, rounding_mode="floor").to(torch.int32)


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
    visible_pages = _visible_page_counts(visible_i32, keys.shape[0])
    parts = []
    for start, stop in _query_tiles(query.shape[0]):
        lnc = 1 if stop - start == 1 else 2
        parts.append(
            _wrapped_projected_indexer[lnc](
                query[start:stop],
                keys,
                gate[start:stop],
                visible_i32[start:stop],
                visible_pages[start:stop],
                512,
            )
        )
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
    block_table: torch.Tensor,
    visible: torch.Tensor,
    *,
    logical_slots_per_block: int,
) -> IndexerSelection:
    """Opaque single-request paged indexer with safe indirect DMA targets."""
    if (
        query.shape[0] not in _SCHEDULER_QUERY_BUCKETS
        and query.shape[0] != _INDEXER_QUERY_TILE
    ):
        raise RuntimeError(
            "DeepSeek-V4 NKI indexer requires query bucket 1, 512, 1024, 2048, or 4096"
        )
    if query.shape[1:] != (1, 64, 128) or gate.shape != query.shape[:-1]:
        raise RuntimeError("DeepSeek-V4 NKI indexer requires [Q,1,64,128] query")
    if key_cache.ndim != 4 or key_cache.shape[1] != 1 or key_cache.shape[-1] != 128:
        raise RuntimeError("DeepSeek-V4 NKI indexer cache must be [blocks,1,page,128]")
    if block_table.ndim != 1:
        raise RuntimeError("DeepSeek-V4 NKI indexer milestone requires one request")
    if visible.shape != (query.shape[0],):
        raise RuntimeError("DeepSeek-V4 NKI indexer visible counts must be [Q]")
    if logical_slots_per_block < 1 or 512 % logical_slots_per_block:
        raise RuntimeError("logical indexer page size must divide 512")
    if block_table.numel() * logical_slots_per_block % 512:
        raise RuntimeError("logical indexer capacity must be a multiple of 512")
    if key_cache.shape[2] < logical_slots_per_block:
        raise RuntimeError("logical indexer page exceeds its physical cache page")
    if not can_run_kernel(query):
        raise RuntimeError("paged_projected_bf16_indexer is a Neuron-only opaque path")
    if any(t.dtype != torch.bfloat16 for t in (query, gate, key_cache)):
        raise RuntimeError("DeepSeek-V4 NKI indexer requires BF16 inputs")

    block_valid = (block_table >= 0) & (block_table < key_cache.shape[0])
    safe_table = block_table.clamp(0, key_cache.shape[0] - 1).to(torch.int32)
    flat_cache = key_cache[:, 0].reshape(-1, 128)
    capacity = block_table.numel() * logical_slots_per_block
    visible_i32 = visible.to(torch.int32).clamp(min=0, max=capacity)
    visible_pages = _visible_page_counts(visible_i32, capacity)
    block_valid_i32 = block_valid.to(torch.int32)
    parts = []
    for start, stop in _query_tiles(query.shape[0]):
        lnc = 1 if stop - start == 1 else 2
        parts.append(
            _wrapped_projected_indexer[lnc](
                query[start:stop],
                flat_cache,
                gate[start:stop],
                visible_i32[start:stop],
                visible_pages[start:stop],
                512,
                safe_table,
                block_valid_i32,
                key_cache.shape[2],
                logical_slots_per_block,
            )
        )
    indices = torch.cat([part[0] for part in parts], dim=0)
    used = torch.cat([part[1] for part in parts], dim=0)
    used = used.clamp(min=0, max=512)
    valid = torch.arange(512, device=query.device)[None, :] < used[:, None]
    return IndexerSelection(
        torch.where(valid, indices.to(torch.int32), torch.full_like(indices, -1)),
        valid,
    )
