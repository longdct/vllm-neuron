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
    rotary = x[..., -rope_dim:]
    if inverse:
        sin = -sin
    rotated = apply_rotary(rotary, cos, sin)
    rotary_indices = torch.arange(
        x.shape[-1] - rope_dim, x.shape[-1], device=x.device
    )
    # A functional overwrite avoids Neuron lowering the small rotated suffix
    # as a dead/zero concat operand (observed for rank-4 query tensors with a
    # two-channel rotary suffix).  The indices and output shape are entirely
    # static for a compiled model.
    return torch.index_copy(x, -1, rotary_indices, rotated)


def gather_paged_latent(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_length: int,
    *,
    start_token: int = 0,
    logical_slots_per_block: int | None = None,
    return_validity: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Gather native-token order from ``[blocks,heads,slots,latent]`` storage.

    ``start_token`` is the absolute token offset the read begins at --
    default 0 reads from column 0, exactly as before. Any caller reading
    from an *evicting* sliding-window cache group must pass the true live
    window's start (``max(0, cached_seq_len - window)``), not rely on the
    default: real vLLM's ``SlidingWindowManager`` never compacts a block
    table on eviction, it remaps only the columns that have fallen entirely
    out of the window to a shared null block, in place, while the live
    window's real data keeps living at ever-higher column indices as
    generation continues. Reading columns ``[0:required]`` unconditionally
    (the old, and still the default, behavior) is therefore only correct
    while nothing has been evicted yet -- see
    ``docs/model-dev/deepseek-v4-swa-null-block-bug.md`` for the full account
    of the bug this parameter fixes.
    """
    if cache.ndim != 4 or block_table.ndim != 1:
        raise ValueError("cache must be 4-D and block_table must be 1-D")
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if start_token < 0:
        raise ValueError("start_token must be non-negative")
    if sequence_length == 0:
        empty = cache.new_zeros((0, cache.shape[1], cache.shape[3]))
        validity = torch.zeros(0, dtype=torch.bool, device=cache.device)
        return (empty, validity) if return_validity else empty
    physical_stride = cache.shape[2]
    slots_per_block = physical_stride if logical_slots_per_block is None else logical_slots_per_block
    if slots_per_block < 1 or slots_per_block > physical_stride:
        raise ValueError("logical_slots_per_block must fit within the physical page stride")
    if block_table.numel() == 0 or cache.shape[0] == 0:
        raise ValueError("cache and block table must contain storage")
    positions = start_token + torch.arange(sequence_length, device=block_table.device)
    columns = torch.div(positions, slots_per_block, rounding_mode="floor")
    offsets = positions % slots_per_block
    column_valid = columns < block_table.numel()
    safe_columns = columns.clamp(min=0, max=block_table.numel() - 1)
    candidate_blocks = block_table[safe_columns].long()
    block_valid = (candidate_blocks >= 0) & (candidate_blocks < cache.shape[0])
    valid = column_valid & block_valid
    safe_blocks = candidate_blocks.clamp(min=0, max=cache.shape[0] - 1)
    gathered = cache[safe_blocks, :, offsets, :]
    gathered = torch.where(valid[:, None, None], gathered, torch.zeros_like(gathered))
    return (gathered, valid) if return_validity else gathered


def gather_recent_window(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    window: int,
    end_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamo-shape-static counterpart to ``gather_paged_latent``: always
    returns exactly ``window`` rows -- ``window`` is a compile-time-constant
    Python int, never derived from a traced value -- covering absolute token
    positions ``[end_position + 1 - window, end_position]`` (inclusive of
    ``end_position``), plus a ``[window]`` bool mask marking which of those
    rows are real content.

    ``end_position`` is a tensor (0-D or broadcastable), so the *offset* into
    the block table is fully tensor-derived; the fixed ``window`` size then
    comes from ordinary advanced indexing (real proxied ops), not a
    Python-int-length slice -- the combination Dynamo can trace without
    tripping "Could not guard on data-dependent expression" (see
    ``docs/model-dev/deepseek-v4-swa-null-block-bug.md``'s "Suggested fix
    direction" item 2). By construction the window's last row is always
    ``end_position`` itself, so every row satisfies the causal
    ``kpos <= qpos`` check on its own; the only rows that can be invalid are
    *leading* ones with a negative absolute position -- generation hasn't
    produced that much history yet (only possible when ``end_position + 1 <
    window``). Callers combine the returned mask with
    ``mla_attention_reference``'s ``key_valid`` to exclude those rows.

    Unlike ``gather_paged_latent``, this never reads a null-remapped column:
    real vLLM's eviction only nulls columns strictly *before* the live
    window, and this always reads columns covering
    ``[end_position + 1 - window, end_position]`` -- exactly the live
    window, by the same reasoning as ``gather_paged_latent``'s
    ``start_token`` parameter.
    """
    if cache.ndim != 4 or block_table.ndim != 1:
        raise ValueError("cache must be 4-D and block_table must be 1-D")
    if window < 1:
        raise ValueError("window must be positive")
    slots_per_block = cache.shape[2]
    start = end_position.view(()).long() + (1 - window)
    positions = start + torch.arange(window, device=block_table.device, dtype=start.dtype)
    valid = positions >= 0
    positions = positions.clamp(min=0)
    columns = torch.div(positions, slots_per_block, rounding_mode="floor")
    slot_offsets = positions % slots_per_block
    if block_table.numel() == 0 or cache.shape[0] == 0:
        raise ValueError("cache and block table must contain storage")
    column_valid = columns < block_table.numel()
    columns = columns.clamp(max=block_table.numel() - 1)
    candidate_blocks = block_table[columns].long()
    block_valid = (candidate_blocks >= 0) & (candidate_blocks < cache.shape[0])
    valid = valid & column_valid & block_valid
    blocks = candidate_blocks.clamp(min=0, max=cache.shape[0] - 1)
    gathered = cache[blocks, :, slot_offsets, :]
    gathered = torch.where(valid[:, None, None], gathered, torch.zeros_like(gathered))
    return gathered, valid


def scatter_paged_latent(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
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
    if cache.shape[1] != 1:
        raise ValueError("DeepSeek latent cache must have exactly one head")
    if slot_mapping.ndim != 1 or values.ndim != 2:
        raise ValueError("slot_mapping must be 1-D and values must be 2-D")
    if slot_mapping.shape[0] != values.shape[0]:
        raise ValueError("slot_mapping and values must have the same row count")
    storage_block_size = cache.shape[2]
    capacity = cache.shape[0] * storage_block_size
    if capacity == 0:
        raise ValueError("cache must contain storage")
    if not torch.compiler.is_compiling() and (
        (slot_mapping >= capacity).any()
    ):
        raise ValueError("slot_mapping contains an out-of-range positive slot")
    valid = (slot_mapping >= 0) & (slot_mapping < capacity)
    # Keep every intermediate shape independent of the number of valid rows.
    # Invalid rows target scratch slot zero but contribute neither an update
    # nor an update count, so they cannot race with a genuine write to slot 0.
    safe_slots = torch.where(valid, slot_mapping, torch.zeros_like(slot_mapping)).long()
    storage_slots = torch.arange(capacity, device=cache.device)
    if values.shape[0] == 1:
        update_mask = (storage_slots == safe_slots[0])[:, None] & valid[0]
        updates = values[0].to(cache.dtype).expand(capacity, -1)
    else:
        row_targets = safe_slots[:, None] == storage_slots[None, :]
        row_targets = row_targets & valid[:, None]
        flat_counts = row_targets.sum(dim=0, keepdim=False).unsqueeze(-1)
        flat_updates = (
            row_targets.to(cache.dtype).transpose(0, 1) @ values.to(cache.dtype)
        )
        update_mask = flat_counts > 0
        # Slot mappings are expected to be unique. Averaging duplicates keeps
        # the operation deterministic without changing the normal case.
        updates = flat_updates / flat_counts.clamp(min=1)
    cache_head = cache[:, 0, :, :].reshape(capacity, cache.shape[3])
    updated_head = torch.where(update_mask, updates, cache_head).reshape(
        cache.shape[0], storage_block_size, cache.shape[3]
    )
    updated_cache = updated_head.unsqueeze(1)
    cache.copy_(updated_cache)
    return updated_cache


def compressed_entry_slot_mapping(
    raw_slot_mapping: torch.Tensor,
    compress_ratio: int,
    raw_block_size: int,
    physical_page_stride: int,
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
    if raw_block_size < 1 or raw_block_size % compress_ratio:
        raise ValueError("raw_block_size must be a positive multiple of compress_ratio")
    logical_entries = raw_block_size // compress_ratio
    if physical_page_stride < logical_entries:
        raise ValueError("physical_page_stride is smaller than the logical compressed page")
    safe_raw = raw_slot_mapping.clamp(min=0)
    physical_block = torch.div(safe_raw, raw_block_size, rounding_mode="floor")
    raw_offset = safe_raw % raw_block_size
    valid = (raw_slot_mapping >= 0) & ((raw_offset + 1) % compress_ratio == 0)
    logical_offset = torch.div(raw_offset, compress_ratio, rounding_mode="floor")
    entry_slot = physical_block * physical_page_stride + logical_offset
    return torch.where(valid, entry_slot, torch.full_like(raw_slot_mapping, -1))


def visible_compressed_entries(
    position: torch.Tensor, compress_ratio: int
) -> torch.Tensor:
    """Compressed entries a query at *position* may attend to.

    The read-side counterpart of :func:`compressed_entry_slot_mapping`, and
    deliberately its neighbour: the two must agree on when an entry exists.
    A token completes a window when ``(position + 1) % compress_ratio == 0``,
    so once it has been compressed there are ``(position + 1) //
    compress_ratio`` entries -- *including* the one this very token just
    completed. The reference gates visibility identically
    (``causal_threshold = (position_ids + 1) // compress_rate`` in
    ``DeepseekV4CSACompressor``).

    Counting ``position // compress_ratio`` instead hides each new entry from
    the query that completes it. That is invisible at most positions and wrong
    at exactly ``position % compress_ratio == compress_ratio - 1``.
    """
    if compress_ratio < 1:
        raise ValueError("compress_ratio must be positive")
    return torch.div(position + 1, compress_ratio, rounding_mode="floor")


def read_compressed_history(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    compress_ratio: int,
    raw_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All compressed entries visible at *position_ids*, Dynamo-shape-static.

    The cache stores one physical row per *compressed entry*, not per raw token
    (``storage_block_size = block_size // compress_ratio`` --
    ``kv_spec_conversion.py``), while positions are raw-token-scaled like every
    other group (see :func:`compressed_entry_slot_mapping`). The two address
    spaces differ by exactly ``compress_ratio``, matching the write side's
    ``raw_slot // compress_ratio``.

    Unlike the sliding-window readers, this group never evicts -- entries are
    addressed from 0 and simply accumulate, so "the first ``num_entries``
    columns" is a fixed, growing *prefix*, not a moving window. So rather than
    branch on the live count, gather the entire block-table-addressable
    capacity (``max_entries``, a plain Python int from real tensor shapes) and
    mask off what is not yet real. Trades throughput for compilability, the
    same tradeoff ``DeepseekV4MoE.forward``'s always-compute redesign
    documents.

    Returns ``(entries, valid)``: ``[max_entries, head_dim]`` with invalid rows
    zeroed, and the ``[max_entries]`` mask saying which are real.

    Shared by the dense attention path and the lightning indexer, which reads
    its own parallel cache at ``index_head_dim``. Sharing it is deliberate:
    the two must agree on which entries exist, and
    :func:`visible_compressed_entries` is where that is decided.
    """
    logical_slots_per_block = raw_block_size // compress_ratio
    max_entries = block_table_row.shape[0] * logical_slots_per_block
    gathered = gather_paged_latent(
        cache,
        block_table_row,
        max_entries,
        logical_slots_per_block=logical_slots_per_block,
    ).squeeze(1)
    num_entries = visible_compressed_entries(
        position_ids.view(()).long(), compress_ratio
    )
    valid = torch.arange(max_entries, device=gathered.device) < num_entries
    # Invalid physical rows may hold allocator garbage or a non-finite candidate
    # from a compressor call that did not complete a logical window. Masking
    # logits alone is insufficient: the value path still evaluates ``0 * NaN``.
    gathered = torch.where(valid[:, None], gathered, torch.zeros_like(gathered))
    return gathered, valid


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
    key_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialized fp32 MLA oracle supporting prefill and one-token decode.

    ``query`` is ``[B,T,H,Dq]``, ``latent`` is ``[B,S,L]``, projection weights
    are ``[H,L,Dq]`` and ``[H,L,Dv]``. Optional RoPE features are concatenated
    to Q/K after the latent projection. The implementation is deliberately
    straightforward and never used as a production kernel.

    ``key_valid`` (``[S]``, ``causal=True`` only) marks rows of ``latent``
    that are real content vs. structural padding -- needed once a caller
    supplies a *fixed-size* ``latent`` that can include rows the local
    ``kpos<=qpos`` ordering check alone can't see are invalid (e.g.
    ``gather_recent_window``'s leading rows before generation has produced
    that much history yet). ``None`` (default) applies no extra masking,
    exactly like before this parameter existed.
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
        if key_valid is not None:
            if key_valid.shape != (s,):
                raise ValueError("key_valid must have shape [S]")
            allowed = allowed & key_valid[None, :]
        scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    elif key_valid is not None:
        raise ValueError("key_valid requires causal=True")
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
