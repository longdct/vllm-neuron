# SPDX-License-Identifier: Apache-2.0
"""Portable DeepSeek-V4 MLA reference operations used by T0 bring-up."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply interleaved rotary embedding to the last dimension."""
    if x.shape[-1] % 2:
        raise ValueError("rotary dimension must be even")
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def apply_partial_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rope_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate only the trailing RoPE channels, optionally applying the inverse."""
    if rope_dim < 0 or rope_dim > x.shape[-1] or rope_dim % 2:
        raise ValueError("rope_dim must be even and within the head dimension")
    if rope_dim == 0:
        return x
    prefix, rotary = x[..., :-rope_dim], x[..., -rope_dim:]
    if inverse:
        sin = -sin
    return torch.cat((prefix, apply_rotary(rotary, cos, sin)), dim=-1)


def gather_paged_latent(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    """Gather native-token order from ``[blocks,heads,slots,latent]`` storage."""
    if cache.ndim != 4 or block_table.ndim != 1:
        raise ValueError("cache must be 4-D and block_table must be 1-D")
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    slots_per_block = cache.shape[2]
    required = math.ceil(sequence_length / slots_per_block) if sequence_length else 0
    if required > block_table.numel():
        raise ValueError("block table is too short for sequence_length")
    blocks = block_table[:required].long()
    # Pure input-validation guard, skipped while torch.compile is tracing --
    # see the matching comment on mhc.py::sinkhorn_positive.
    if not torch.compiler.is_compiling() and (
        (blocks < 0).any() or (blocks >= cache.shape[0]).any()
    ):
        raise ValueError("block table contains an invalid physical block")
    gathered = cache[blocks].permute(0, 2, 1, 3).reshape(-1, cache.shape[1], cache.shape[3])
    return gathered[:sequence_length]


def scatter_paged_latent(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Write ``values`` into ``[blocks,heads,slots,latent]`` storage in place.

    ``slot_mapping`` is a flat cache-relative slot index per row of ``values``
    (``blk_idx = slot // storage_block_size``, ``pos_idx = slot %
    storage_block_size``, matching the convention every other cache write in
    this plugin uses -- see ``llama3/model.py``'s ``_write_kv_cache``).
    Entries with ``slot_mapping == -1`` are padding and are not written,
    exactly like every other paged cache write in this plugin.
    """
    if cache.ndim != 4:
        raise ValueError("cache must be 4-D: [blocks, heads, slots, latent]")
    if slot_mapping.ndim != 1 or values.ndim != 2:
        raise ValueError("slot_mapping must be 1-D and values must be 2-D")
    if slot_mapping.shape[0] != values.shape[0]:
        raise ValueError("slot_mapping and values must have the same row count")
    storage_block_size = cache.shape[2]
    valid = slot_mapping >= 0
    # Padding rows are dropped rather than redirected to a dummy slot: an
    # earlier version wrote padding rows back to slot 0 with their
    # pre-existing value to keep this a plain unconditional index_put_, but
    # index_put_ makes no ordering guarantee between rows that collide on the
    # same physical index -- whenever a real row's slot happened to be 0 too,
    # the padding row's "restore" write could nondeterministically clobber
    # it. Filtering to the valid rows costs a data-dependent shape (fine here
    # -- this pass is eager/CPU-mode; see the module docstring's scope note)
    # but is unconditionally correct.
    slot_mapping = slot_mapping[valid]
    values = values[valid]
    if slot_mapping.numel() == 0:
        return
    blk_idx = torch.div(slot_mapping, storage_block_size, rounding_mode="floor")
    pos_idx = slot_mapping % storage_block_size
    cache.index_put_(
        (blk_idx.long(), torch.zeros_like(blk_idx.long()), pos_idx.long()),
        values.to(cache.dtype),
    )


def compressed_entry_slot_mapping(
    raw_slot_mapping: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Map a per-token raw-cache slot to its compressed-entry storage slot.

    Ported directly from vLLM 0.24's own DeepSeek-V4 GPU backend
    (``vllm/v1/attention/backends/mla/sparse_swa.py::
    _compressed_slot_mapping_kernel``): a raw token completes a compressed
    window exactly when ``(pos + 1) % compress_ratio == 0``, where ``pos`` is
    its absolute sequence position. Because the runner computes
    ``raw_slot_mapping`` the same way for every cache group (physical block
    from ``block_table``, offset ``pos % block_size``) and this group's
    ``block_size`` is always a multiple of ``compress_ratio`` (enforced by
    ``kv_spec_conversion.layer_spec_to_vllm_spec``), ``raw_slot_mapping``'s
    low bits already equal ``pos``'s low bits mod ``compress_ratio`` -- so
    the same test and the same floor-division apply directly to the slot
    value, with no need to separately track absolute position here.

    Returns the compressed storage slot (``-1`` for padding and for raw
    tokens that do not complete a window -- both mean "do not write").
    """
    if compress_ratio < 1:
        raise ValueError("compress_ratio must be positive")
    valid = (raw_slot_mapping >= 0) & ((raw_slot_mapping + 1) % compress_ratio == 0)
    entry_slot = torch.div(raw_slot_mapping, compress_ratio, rounding_mode="floor")
    return torch.where(valid, entry_slot, torch.full_like(raw_slot_mapping, -1))


def compose_swa_and_compressed_history(
    local: torch.Tensor,
    compressed: torch.Tensor,
    *,
    sliding_window: int,
) -> torch.Tensor:
    """Place long-range compressed entries before the retained local suffix."""
    if local.ndim != compressed.ndim or local.shape[0] != compressed.shape[0]:
        raise ValueError("local and compressed histories must have compatible batches")
    if sliding_window < 1:
        raise ValueError("sliding_window must be positive")
    return torch.cat((compressed, local[:, -sliding_window:]), dim=1)


@dataclass(frozen=True)
class MLABucket:
    batch_size: int
    query_length: int
    context_length: int
    head_dim: int = 512

    def __post_init__(self) -> None:
        if min(self.batch_size, self.query_length, self.context_length, self.head_dim) < 1:
            raise ValueError("bucket dimensions must be positive")
        if self.head_dim != 512:
            raise ValueError("DeepSeek-V4 MLA buckets require head_dim=512")


P2_REPRESENTATIVE_BUCKETS = (
    MLABucket(1, 32, 32),
    MLABucket(1, 128, 128),
    MLABucket(1, 1, 128),
    MLABucket(4, 1, 512),
)


def mla_attention_reference(
    query: torch.Tensor,
    latent: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    attention_sinks: torch.Tensor | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Materialized fp32 MLA oracle supporting prefill and one-token decode.

    ``query`` is ``[B,T,H,Dq]``, ``latent`` is ``[B,S,L]``, projection weights
    are ``[H,L,Dq]`` and ``[H,L,Dv]``. Optional RoPE features are concatenated
    to Q/K after the latent projection. The implementation is deliberately
    straightforward and never used as a production kernel.
    """
    if latent.shape[-1] != key_weight.shape[1] or key_weight.shape[:2] != value_weight.shape[:2]:
        raise ValueError("latent/projection dimensions do not agree")
    q = query.float()
    k = torch.einsum("bsl,hld->bshd", latent.float(), key_weight.float())
    v = torch.einsum("bsl,hlv->bshv", latent.float(), value_weight.float())
    if (query_rope is None) != (key_rope is None):
        raise ValueError("query_rope and key_rope must be provided together")
    if query_rope is not None:
        q = torch.cat((q, query_rope.float()), dim=-1)
        k = torch.cat((k, key_rope.float()), dim=-1)
    scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    t, s = query.shape[1], latent.shape[1]
    if causal:
        # Decode (T=1) sees the whole supplied cache; prefill aligns its last
        # query with the last key when S >= T.
        qpos = torch.arange(s - t, s, device=scores.device)[:, None]
        kpos = torch.arange(s, device=scores.device)[None, :]
        allowed = kpos <= qpos
        if sliding_window is not None:
            allowed &= kpos > (qpos - sliding_window)
        scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    if attention_sinks is not None:
        if attention_sinks.shape != (query.shape[2],):
            raise ValueError("attention_sinks must have shape [num_heads]")
        sink = attention_sinks.float()[None, :, None, None].expand(
            query.shape[0], -1, t, 1
        )
        weights = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :s]
    else:
        weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bshv->bthv", weights, v).to(query.dtype)
