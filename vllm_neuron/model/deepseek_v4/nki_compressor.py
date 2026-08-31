# SPDX-License-Identifier: Apache-2.0
"""Boundary-only paged gated reduction for DeepSeek-V4 compressors."""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch_neuronx.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

from .attention import logical_to_physical_slots_batched

_SUPPORTED_GEOMETRIES = frozenset(((128, 512), (4, 512), (4, 128)))


@nki.jit
def _paged_gated_compressor_kernel(
    state_cache,
    window_slots,
    window_mask,
    candidate_valid,
    position_bias,
    overlap: bool,
):
    """Gather and reduce only compressor-window completion candidates.

    ``window_mask`` is additive BF16 (zero for a real row, ``-inf`` for a
    missing row). Invalid candidates receive an all-zero mask and sanitized
    slot zero from the wrapper, so their reduction remains finite; the final
    ``candidate_valid`` multiplication writes an exact zero.
    """
    candidate_count, window = window_slots.shape
    ratio, width = position_bias.shape
    state_width = state_cache.shape[-1]
    coff = 2 if overlap else 1
    head_dim = width // coff
    assert (ratio, head_dim) in ((128, 512), (4, 512), (4, 128))
    assert window == coff * ratio
    assert width == coff * head_dim
    assert state_width == 2 * width
    assert head_dim % 128 == 0
    assert candidate_count == 1 or candidate_count % 2 == 0
    reduction_window = 32 if overlap else window

    output = nl.ndarray(
        (candidate_count, head_dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    flat_cache = state_cache.reshape(
        (state_cache.shape[0] * state_cache.shape[2], state_width)
    )
    flat_slots = window_slots.reshape((candidate_count * window,))
    flat_mask = window_mask.reshape((candidate_count * window,))
    flat_output = output.reshape((candidate_count * head_dim,))
    bias_internal = nl.ndarray(
        (ratio, width), dtype=position_bias.dtype, buffer=nl.sbuf
    )
    nisa.dma_copy(dst=bias_internal, src=position_bias)

    n_programs = nl.num_programs(0)
    program_id = nl.program_id(0)
    candidates_per_program = (
        candidate_count if candidate_count == 1 else candidate_count // n_programs
    )
    program_start = 0 if candidate_count == 1 else program_id * candidates_per_program

    def reduce_candidate(local_candidate):
        local_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        candidate_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.register_store(local_sb, local_candidate)
        nisa.tensor_scalar(
            dst=candidate_sb,
            data=local_sb,
            op0=nl.add,
            operand0=program_start,
        )

        window_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=window_base,
            data=candidate_sb,
            op0=nl.multiply,
            operand0=window,
        )
        window_base_reg = nisa.register_alloc()
        nisa.register_load(window_base_reg, window_base)
        if overlap:
            current_mask_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=current_mask_base,
                data=window_base,
                op0=nl.add,
                operand0=ratio,
            )
            current_mask_base_reg = nisa.register_alloc()
            nisa.register_load(current_mask_base_reg, current_mask_base)

        offsets = nl.ndarray((window, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=offsets,
            src=flat_slots.ap(
                pattern=[[1, window]],
                scalar_offset=window_base_reg,
                indirect_dim=0,
            ),
        )
        state = nl.ndarray(
            (window, state_width), dtype=state_cache.dtype, buffer=nl.sbuf
        )
        nisa.memset(state, value=0.0)
        nisa.dma_copy(
            dst=state,
            src=flat_cache.ap(
                pattern=[[state_width, window], [1, state_width]],
                vector_offset=offsets,
                indirect_dim=0,
            ),
            oob_mode=nisa.oob_mode.skip,
        )

        valid_tile = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        candidate_reg = nisa.register_alloc()
        nisa.register_load(candidate_reg, candidate_sb)
        nisa.dma_copy(
            dst=valid_tile,
            src=candidate_valid.ap(
                pattern=[[0, 128], [1, 1]],
                scalar_offset=candidate_reg,
                indirect_dim=0,
            ),
        )

        for dim_start in nl.static_range(0, head_dim, 128):
            kv_t = nl.ndarray(
                (128, reduction_window), dtype=state_cache.dtype, buffer=nl.sbuf
            )
            gate_t = nl.ndarray(
                (128, reduction_window), dtype=state_cache.dtype, buffer=nl.sbuf
            )
            bias_t = nl.ndarray(
                (128, reduction_window), dtype=position_bias.dtype, buffer=nl.sbuf
            )
            nisa.memset(kv_t, value=0.0)
            nisa.memset(gate_t, value=0.0)
            nisa.memset(bias_t, value=0.0)
            if overlap:
                nisa.dma_transpose(
                    dst=kv_t[:, :ratio],
                    src=state[:ratio, dim_start : dim_start + 128],
                )
                nisa.dma_transpose(
                    dst=kv_t[:, 16 : 16 + ratio],
                    src=state[
                        ratio:,
                        head_dim + dim_start : head_dim + dim_start + 128,
                    ],
                )
                nisa.dma_transpose(
                    dst=gate_t[:, :ratio],
                    src=state[
                        :ratio,
                        width + dim_start : width + dim_start + 128,
                    ],
                )
                nisa.dma_transpose(
                    dst=gate_t[:, 16 : 16 + ratio],
                    src=state[
                        ratio:,
                        width + head_dim + dim_start : width
                        + head_dim
                        + dim_start
                        + 128,
                    ],
                )
                nisa.dma_transpose(
                    dst=bias_t[:, :ratio],
                    src=bias_internal[:, dim_start : dim_start + 128],
                )
                nisa.dma_transpose(
                    dst=bias_t[:, 16 : 16 + ratio],
                    src=bias_internal[
                        :, head_dim + dim_start : head_dim + dim_start + 128
                    ],
                )
            else:
                nisa.dma_transpose(
                    dst=kv_t,
                    src=state[:, dim_start : dim_start + 128],
                )
                nisa.dma_transpose(
                    dst=gate_t,
                    src=state[:, width + dim_start : width + dim_start + 128],
                )
                nisa.dma_transpose(
                    dst=bias_t,
                    src=bias_internal[:, dim_start : dim_start + 128],
                )

            logits = nl.ndarray(
                (128, reduction_window), dtype=nl.bfloat16, buffer=nl.sbuf
            )
            nisa.tensor_tensor(dst=logits, data1=gate_t, data2=bias_t, op=nl.add)
            mask_t = nl.ndarray(
                (128, reduction_window), dtype=nl.bfloat16, buffer=nl.sbuf
            )
            if overlap:
                nisa.memset(mask_t, value=float("-inf"))
                nisa.dma_copy(
                    dst=mask_t[:, :ratio],
                    src=flat_mask.ap(
                        pattern=[[0, 128], [1, ratio]],
                        scalar_offset=window_base_reg,
                        indirect_dim=0,
                    ),
                )
                nisa.dma_copy(
                    dst=mask_t[:, 16 : 16 + ratio],
                    src=flat_mask.ap(
                        pattern=[[0, 128], [1, ratio]],
                        scalar_offset=current_mask_base_reg,
                        indirect_dim=0,
                    ),
                )
            else:
                nisa.dma_copy(
                    dst=mask_t,
                    src=flat_mask.ap(
                        pattern=[[0, 128], [1, window]],
                        scalar_offset=window_base_reg,
                        indirect_dim=0,
                    ),
                )
            nisa.tensor_tensor(dst=logits, data1=logits, data2=mask_t, op=nl.add)

            neg_max = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=neg_max,
                op=nl.maximum,
                data=logits,
                axis=1,
                negate=True,
            )
            weights = nl.ndarray(
                (128, reduction_window), dtype=nl.bfloat16, buffer=nl.sbuf
            )
            denominator = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=weights,
                op=nl.exp,
                data=logits,
                bias=neg_max,
                reduce_op=nl.add,
                reduce_res=denominator,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )
            weighted = nl.ndarray(
                (128, reduction_window), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(dst=weighted, data1=kv_t, data2=weights, op=nl.multiply)
            numerator = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=numerator, op=nl.add, data=weighted, axis=1)
            reciprocal = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=reciprocal, data=denominator)
            reduced = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=reduced, data1=numerator, data2=reciprocal, op=nl.multiply
            )
            nisa.tensor_tensor(
                dst=reduced, data1=reduced, data2=valid_tile, op=nl.multiply
            )

            output_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=output_base,
                data=candidate_sb,
                op0=nl.multiply,
                operand0=head_dim,
                op1=nl.add,
                operand1=dim_start,
            )
            output_base_reg = nisa.register_alloc()
            nisa.register_load(output_base_reg, output_base)
            nisa.dma_copy(
                dst=flat_output.ap(
                    pattern=[[1, 128]],
                    scalar_offset=output_base_reg,
                    indirect_dim=0,
                ),
                src=reduced,
            )

    nl.fori_loop(0, candidates_per_program, reduce_candidate)
    return output


_wrapped_paged_gated_compressor = wrap_nki(_paged_gated_compressor_kernel)


def _completion_candidates(
    positions: torch.Tensor,
    token_to_request: torch.Tensor,
    output_slot_mapping: torch.Tensor,
    ratio: int,
    num_requests: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed-size per-request completion candidates.

    vLLM packs every request into an equally sized, padded segment. Building a
    fixed ``ceil(segment / ratio)`` candidate grid per segment preserves the
    boundary-only kernel size while allowing ragged prefill and one-token
    decode rows from different requests to share a launch.
    """
    if positions.ndim != 1 or token_to_request.shape != positions.shape:
        raise ValueError("positions and token_to_request must be one-dimensional")
    if output_slot_mapping.shape != positions.shape:
        raise ValueError("output_slot_mapping must contain one slot per token")
    if positions.numel() < 1 or ratio < 1 or num_requests < 1:
        raise ValueError(
            "a non-empty packed query, positive ratio, and requests are required"
        )

    query_count = positions.shape[0]
    if query_count % num_requests:
        raise ValueError("packed query rows must divide evenly across requests")
    tokens_per_request = query_count // num_requests
    candidates_per_request = (tokens_per_request + ratio - 1) // ratio
    request_starts = (
        torch.arange(num_requests, device=positions.device, dtype=torch.long)
        * tokens_per_request
    )
    first_positions = positions[request_starts].long()
    first_completions = ratio - 1 - (first_positions % ratio)
    local_candidates = first_completions[:, None] + (
        torch.arange(
            candidates_per_request, device=positions.device, dtype=torch.long
        )[None, :]
        * ratio
    )
    candidate_indices = (request_starts[:, None] + local_candidates).reshape(-1)
    in_range = (local_candidates < tokens_per_request).reshape(-1)
    safe_indices = candidate_indices.clamp(max=query_count - 1)
    actual_positions = positions[safe_indices].long()
    expected_positions = (
        first_positions[:, None] + local_candidates
    ).reshape(-1)
    expected_owners = token_to_request[request_starts].long()[:, None].expand(
        -1, candidates_per_request
    ).reshape(-1)
    valid = in_range & (actual_positions == expected_positions)
    valid &= token_to_request[safe_indices].long() == expected_owners
    slots = output_slot_mapping[safe_indices].long()
    valid &= slots >= 0
    rope_positions = torch.div(expected_positions, ratio, rounding_mode="floor") * ratio
    return candidate_indices, rope_positions, slots, valid


def _paged_candidate_windows(
    state_cache: torch.Tensor,
    positions: torch.Tensor,
    token_to_request: torch.Tensor,
    state_block_tables: torch.Tensor,
    output_slot_mapping: torch.Tensor,
    *,
    ratio: int,
    overlap: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build only ``[candidate_count, coff*ratio]`` raw paged windows."""
    candidate_indices, rope_positions, output_slots, candidate_valid = (
        _completion_candidates(
            positions,
            token_to_request,
            output_slot_mapping,
            ratio,
            state_block_tables.shape[0],
        )
    )
    safe_indices = candidate_indices.clamp(max=positions.shape[0] - 1)
    completion_positions = positions[safe_indices].long()
    candidate_owners = token_to_request[safe_indices].long()
    window = (2 if overlap else 1) * ratio
    offsets = torch.arange(window, device=positions.device, dtype=torch.long)
    logical = completion_positions[:, None] + 1 - window + offsets[None, :]
    requested = logical >= 0
    slots, row_valid = logical_to_physical_slots_batched(
        logical,
        requested,
        state_block_tables,
        candidate_owners,
        logical_slots_per_block=state_cache.shape[2],
        physical_page_stride=state_cache.shape[2],
        cache_blocks=state_cache.shape[0],
    )
    candidate_valid &= row_valid.any(dim=1)
    # Invalid candidates must remain finite inside the kernel so multiplying by
    # candidate_valid produces an exact zero instead of NaN * 0.
    effective_valid = torch.where(
        candidate_valid[:, None], row_valid, torch.ones_like(row_valid)
    )
    safe_slots = torch.where(row_valid, slots, torch.zeros_like(slots)).to(torch.int32)
    zero = torch.zeros((), dtype=torch.bfloat16, device=positions.device)
    neg_inf = torch.full(
        (), float("-inf"), dtype=torch.bfloat16, device=positions.device
    )
    mask = torch.where(effective_valid, zero, neg_inf)
    output_slots = torch.where(
        candidate_valid, output_slots, torch.full_like(output_slots, -1)
    )
    return safe_slots, mask, rope_positions, output_slots, candidate_valid


def paged_gated_compressor(
    state_cache: torch.Tensor,
    positions: torch.Tensor,
    token_to_request: torch.Tensor,
    state_block_tables: torch.Tensor,
    output_slot_mapping: torch.Tensor,
    position_bias: torch.Tensor,
    *,
    ratio: int,
    overlap: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce packed compressor boundaries without a per-query state gather.

    Returns FP32 reduced entries, their compressor-RoPE positions, selected
    compressed-cache slots, and candidate validity. Each request occupies an
    equally sized padded segment; invalid/padded candidates are retained at the
    fixed shape and returned as exact zero rows.
    """
    head_dim = position_bias.shape[1] // (2 if overlap else 1)
    if (ratio, head_dim) not in _SUPPORTED_GEOMETRIES:
        raise RuntimeError(
            f"unsupported DeepSeek-V4 compressor geometry ratio={ratio}, "
            f"head_dim={head_dim}"
        )
    if overlap != (ratio == 4):
        raise RuntimeError("only ratio-4 CSA may use overlapping compressor windows")
    if state_cache.ndim != 4 or state_cache.shape[1] != 1:
        raise RuntimeError("state cache must have shape [blocks,1,page,state_width]")
    width = (2 if overlap else 1) * head_dim
    if state_cache.shape[-1] != 2 * width or position_bias.shape != (ratio, width):
        raise RuntimeError("state cache and position bias do not match the compressor")
    if state_cache.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("DeepSeek-V4 NKI compressor requires BF16 or FP32 state")
    if position_bias.dtype != torch.bfloat16:
        raise RuntimeError("DeepSeek-V4 NKI compressor requires BF16 position bias")
    if not can_run_kernel(state_cache):
        raise RuntimeError("paged_gated_compressor is a Neuron-only opaque path")

    slots, mask, candidate_positions, output_slots, valid = _paged_candidate_windows(
        state_cache,
        positions,
        token_to_request,
        state_block_tables,
        output_slot_mapping,
        ratio=ratio,
        overlap=overlap,
    )
    candidate_count = slots.shape[0]
    kernel_slots = slots
    kernel_mask = mask
    kernel_valid = valid
    padding = 4 - candidate_count if candidate_count < 4 else candidate_count % 2
    if padding:
        # LNC2 gives both programs the same runtime-loop bound. Keep every
        # small launch at a minimum of four candidates: the compiler's PSUM
        # spill pass asserts for two, while a one-candidate LNC1 runtime loop
        # produces mismatched basic blocks when linked into the LNC2 model.
        # Larger odd shapes need only one inert row. Restore the public
        # candidate count after the opaque call in every case.
        kernel_slots = torch.cat(
            (slots, torch.zeros_like(slots[:padding])), dim=0
        )
        kernel_mask = torch.cat((mask, torch.zeros_like(mask[:padding])), dim=0)
        kernel_valid = torch.cat(
            (valid, torch.zeros_like(valid[:padding])), dim=0
        )
    lnc = 2
    reduced = _wrapped_paged_gated_compressor[lnc](
        state_cache,
        kernel_slots,
        kernel_mask,
        kernel_valid.to(torch.float32),
        position_bias,
        overlap,
    )[:candidate_count]
    return reduced, candidate_positions, output_slots, valid
