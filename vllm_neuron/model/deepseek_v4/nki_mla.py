# SPDX-License-Identifier: Apache-2.0
"""NKI prototype for 512-wide materialized latent attention."""

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
_DIRECT_QUERY_BUCKETS = frozenset((1, 512, 1024))
_SCHEDULER_QUERY_BUCKETS = frozenset((1, 512, 1024, 2048, 4096))
_PREFILL_MICROCHUNK = 1024
_HISTORY_LIMITS = frozenset((128, 160, 384, 512, 640, 1024, 1152))


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


@nki.jit
def _materialize_paged_latent_stage(
    sliding_cache,
    sliding_slots,
    sliding_mask,
    compressed_cache,
    compressed_slots,
    compressed_mask,
):
    """Gather two bounded physical-slot streams without exposing them to FX."""
    q_count, sliding_count = sliding_slots.shape
    assert q_count in (1, 512, 1024), "prefill queries must be sliced to <=1024"
    compressed_count = compressed_slots.shape[1]
    latent_dim = sliding_cache.shape[-1]
    history = compressed_count + sliding_count
    sliding_flat = sliding_cache.reshape(
        (sliding_cache.shape[0] * sliding_cache.shape[2], latent_dim)
    )
    compressed_flat = compressed_cache.reshape(
        (compressed_cache.shape[0] * compressed_cache.shape[2], latent_dim)
    )
    latent = nl.ndarray(
        (q_count, history, latent_dim),
        dtype=sliding_cache.dtype,
        buffer=nl.shared_hbm,
    )
    attention_mask = nl.ndarray(
        (q_count, history), dtype=nl.bfloat16, buffer=nl.shared_hbm
    )
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
        for start in nl.static_range(0, compressed_count, 128):
            count = min(128, compressed_count - start)
            offsets = nl.ndarray((count, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=offsets,
                src=compressed_slots[q_idx, start : start + count].reshape((count, 1)),
            )
            gathered = nl.ndarray(
                (count, latent_dim),
                dtype=compressed_cache.dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=gathered,
                src=compressed_flat.ap(
                    pattern=[[latent_dim, count], [1, latent_dim]],
                    vector_offset=offsets,
                    indirect_dim=0,
                ),
                oob_mode=nisa.oob_mode.skip,
            )
            nisa.dma_copy(dst=latent[q_idx, start : start + count, :], src=gathered)
        for start in nl.static_range(0, sliding_count, 128):
            count = min(128, sliding_count - start)
            offsets = nl.ndarray((count, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=offsets,
                src=sliding_slots[q_idx, start : start + count].reshape((count, 1)),
            )
            gathered = nl.ndarray(
                (count, latent_dim), dtype=sliding_cache.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=gathered,
                src=sliding_flat.ap(
                    pattern=[[latent_dim, count], [1, latent_dim]],
                    vector_offset=offsets,
                    indirect_dim=0,
                ),
                oob_mode=nisa.oob_mode.skip,
            )
            nisa.dma_copy(
                dst=latent[
                    q_idx,
                    compressed_count + start : compressed_count + start + count,
                    :,
                ],
                src=gathered,
            )
        nisa.dma_copy(
            dst=attention_mask[q_idx, :compressed_count],
            src=compressed_mask[q_idx, :],
        )
        nisa.dma_copy(
            dst=attention_mask[q_idx, compressed_count:],
            src=sliding_mask[q_idx, :],
        )
    return latent, attention_mask


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
):
    latent, attention_mask = _materialize_paged_latent_stage(
        sliding_cache,
        sliding_slots,
        sliding_mask,
        compressed_cache,
        compressed_slots,
        compressed_mask,
    )
    probs, recip = _manual_qk_softmax_stage(query, latent, attention_mask, sinks)
    return _manual_pv_stage(query, latent, probs, recip)


@nki.jit
def _paged_sliding_latent_mla_kernel(
    query, sliding_cache, sliding_slots, sliding_mask, sinks
):
    q_count, history = sliding_slots.shape
    latent_dim = sliding_cache.shape[-1]
    sliding_flat = sliding_cache.reshape(
        (sliding_cache.shape[0] * sliding_cache.shape[2], latent_dim)
    )
    latent = nl.ndarray(
        (q_count, history, latent_dim),
        dtype=sliding_cache.dtype,
        buffer=nl.shared_hbm,
    )
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
        offsets = nl.ndarray((history, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=offsets,
            src=sliding_slots[q_idx, :].reshape((history, 1)),
        )
        gathered = nl.ndarray(
            (history, latent_dim), dtype=sliding_cache.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=gathered,
            src=sliding_flat.ap(
                pattern=[[latent_dim, history], [1, latent_dim]],
                vector_offset=offsets,
                indirect_dim=0,
            ),
            oob_mode=nisa.oob_mode.skip,
        )
        nisa.dma_copy(dst=latent[q_idx, :, :], src=gathered)
    probs, recip = _manual_qk_softmax_stage(query, latent, sliding_mask, sinks)
    return _manual_pv_stage(query, latent, probs, recip)


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

    q_count = query.shape[0]
    if can_run_kernel(query) and q_count > _PREFILL_MICROCHUNK:
        if q_count not in _SCHEDULER_QUERY_BUCKETS or q_count % _PREFILL_MICROCHUNK:
            raise RuntimeError(
                "DeepSeek-V4 paged MLA has unsupported query bucket "
                f"{q_count}; expected one of {sorted(_SCHEDULER_QUERY_BUCKETS)}"
            )
        # Caches are shared and were completely updated by _forward_packed
        # before this dispatcher is entered.  Only query-derived rows are
        # sliced, preserving token order and each row's causal visibility.
        from .attention import SharedLatentMLAInputs

        outputs = []
        for start in range(0, q_count, _PREFILL_MICROCHUNK):
            stop = start + _PREFILL_MICROCHUNK
            outputs.append(
                paged_shared_latent_mla(
                    SharedLatentMLAInputs(
                        query=query[start:stop],
                        sliding_cache=inputs.sliding_cache,
                        sliding_slots=inputs.sliding_slots[start:stop],
                        sliding_valid=inputs.sliding_valid[start:stop],
                        compressed_cache=inputs.compressed_cache,
                        compressed_slots=(
                            inputs.compressed_slots[start:stop]
                            if has_compressed
                            else None
                        ),
                        compressed_valid=(
                            inputs.compressed_valid[start:stop]
                            if has_compressed
                            else None
                        ),
                        sinks=inputs.sinks,
                    )
                )
            )
        return torch.cat(outputs, dim=0)

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
        query.shape[0] not in _DIRECT_QUERY_BUCKETS
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
    if not has_compressed:
        return _wrapped_paged_sliding_latent_mla[2](
            query,
            inputs.sliding_cache,
            sliding_slots,
            sliding_mask,
            inputs.sinks.to(torch.bfloat16),
        )
    compressed_slots, compressed_valid = safe_slots(
        compressed_cache, compressed_slots, compressed_valid
    )
    return _wrapped_paged_shared_latent_mla[2](
        query,
        inputs.sliding_cache,
        sliding_slots,
        sliding_mask,
        compressed_cache,
        compressed_slots,
        torch.where(compressed_valid, zero, neg_inf),
        inputs.sinks.to(torch.bfloat16),
    )


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
