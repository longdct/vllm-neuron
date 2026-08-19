# SPDX-License-Identifier: Apache-2.0
"""Chunk-invariant strided compressor reference with explicit carry state."""

from dataclasses import dataclass

import torch

from .attention import apply_partial_rotary


@dataclass(frozen=True)
class CompressorState:
    carry: torch.Tensor
    total_tokens: int = 0


@dataclass(frozen=True)
class GatedCompressorState:
    """Unconsumed projections and the prior c4 window across scheduler chunks."""

    kv_carry: torch.Tensor
    gate_carry: torch.Tensor
    overlap_kv: torch.Tensor | None = None
    overlap_gate: torch.Tensor | None = None
    total_tokens: int = 0


def carry_gather_length(cached_seq_len: int, ratio: int, *, needs_overlap: bool) -> int:
    """How many trailing raw-token rows to replay from the state cache.

    The paged compressor-state cache (``CacheKind.COMPRESSOR_STATE``) retains
    raw per-token ``[kv, gate]`` projections under an ordinary sliding-window
    lifecycle -- see ``deepseek-v4-carry-cache-design.md``. Rather than
    deserializing a :class:`GatedCompressorState` from the cache, the device
    path gathers exactly the rows the state-based functions would have kept
    as carry and replays them through ``compress_hca_chunk``/
    ``compress_csa_chunk`` with ``state=None``; ``_join_gated_carry``'s own
    ``torch.cat((state.kv_carry, kv), dim=1)`` makes the two paths produce
    identical windows for identical inputs.

    * HCA (``needs_overlap=False``): the only carry ``compress_hca_chunk``
      keeps is the unconsumed suffix, ``cached_seq_len % ratio`` rows (always
      < ``ratio``, so it can never itself contain a complete window -- no
      already-emitted output to discard).
    * CSA (``needs_overlap=True``): additionally needs the previous complete
      window's raw rows, purely to re-derive ``overlap_kv``/``overlap_gate``
      (``compress_csa_chunk`` computes those as ``windows[:, -1, ...]`` --
      i.e. from raw rows, not a separately-stored reduction). That prepends
      one full window whose output *was* already emitted; the caller must
      discard exactly that many leading rows of ``compress_csa_chunk``'s
      output (see :func:`carry_replay_already_emitted`).
    """
    # Pure input-validation guard, skipped while torch.compile is tracing --
    # cached_seq_len genuinely can never be negative at runtime (it's a
    # monotonically-increasing live KV-cache length), but under Dynamo it
    # arrives as a symbolic (not plain Python) int derived from a traced
    # tensor value, and branching on it trips "Could not guard on
    # data-dependent expression" -- same family of fix as
    # attention.py::gather_paged_latent's bounds check and
    # mhc.py::sinkhorn_positive. ``ratio`` never needs this: it's always a
    # compile-time Python int (a config constant), never traced.
    if not torch.compiler.is_compiling() and cached_seq_len < 0:
        raise ValueError("cached_seq_len must be non-negative")
    if ratio < 1:
        raise ValueError("ratio must be positive")
    unconsumed = cached_seq_len % ratio
    if needs_overlap and cached_seq_len >= ratio:
        return min(cached_seq_len, unconsumed + ratio)
    return unconsumed


def carry_replay_already_emitted(cached_seq_len: int, ratio: int, *, needs_overlap: bool) -> int:
    """Leading output windows to drop after replaying :func:`carry_gather_length` rows.

    Only CSA's overlap-reconstruction prepends a previously-completed
    window, and only when one exists yet (``cached_seq_len >= ratio``); that
    window's output was already emitted by an earlier chunk and must not be
    written to the compressed cache a second time.
    """
    return 1 if (needs_overlap and cached_seq_len >= ratio) else 0


def finalize_compressed_entries(
    compressed: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply V4's weighted RMSNorm and trailing interleaved compressor RoPE."""
    if compressed.ndim != 3 or norm_weight.shape != (compressed.shape[-1],):
        raise ValueError("compressed entries must be [batch, entries, head_dim]")
    if cos.shape != sin.shape or cos.shape[:-1] != compressed.shape[:-1]:
        raise ValueError("compressor cos/sin shapes do not agree with entries")
    normalized = compressed.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    normalized = (normalized.to(compressed.dtype) * norm_weight).to(compressed.dtype)
    return apply_partial_rotary(
        normalized,
        cos,
        sin,
        rope_dim=2 * cos.shape[-1],
    )


def _join_gated_carry(
    kv: torch.Tensor,
    gate: torch.Tensor,
    state: GatedCompressorState | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv.ndim != 3 or gate.shape != kv.shape:
        raise ValueError("kv and gate must have identical [batch, sequence, width] shapes")
    if state is None:
        return kv, gate
    if (
        state.kv_carry.ndim != 3
        or state.kv_carry.shape != state.gate_carry.shape
        or state.kv_carry.shape[0] != kv.shape[0]
        or state.kv_carry.shape[2] != kv.shape[2]
    ):
        raise ValueError("compressor carry and new projections do not agree")
    return (
        torch.cat((state.kv_carry, kv), dim=1),
        torch.cat((state.gate_carry, gate), dim=1),
    )


def compress_hca_chunk(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    state: GatedCompressorState | None = None,
) -> tuple[torch.Tensor, GatedCompressorState]:
    """Transformers-equivalent c128 gated reduction before RMSNorm and RoPE."""
    joined_kv, joined_gate = _join_gated_carry(kv, gate, state)
    ratio = position_bias.shape[0]
    if ratio < 1 or position_bias.shape != (ratio, kv.shape[-1]):
        raise ValueError("HCA position_bias must have shape [ratio, head_dim]")
    complete = joined_kv.shape[1] // ratio
    used = complete * ratio
    if complete:
        windows = joined_kv[:, :used].view(kv.shape[0], complete, ratio, -1)
        logits = joined_gate[:, :used].view_as(windows) + position_bias
        weights = logits.softmax(dim=2, dtype=torch.float32).to(kv.dtype)
        output = (windows * weights).sum(dim=2)
    else:
        output = kv.new_empty((kv.shape[0], 0, kv.shape[-1]))
    prior_tokens = 0 if state is None else state.total_tokens
    return output, GatedCompressorState(
        joined_kv[:, used:],
        joined_gate[:, used:],
        total_tokens=prior_tokens + kv.shape[1],
    )


def compress_csa_chunk(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    state: GatedCompressorState | None = None,
) -> tuple[torch.Tensor, GatedCompressorState]:
    """Transformers-equivalent c4 overlapping reduction before RMSNorm and RoPE."""
    joined_kv, joined_gate = _join_gated_carry(kv, gate, state)
    ratio, double_width = position_bias.shape
    if ratio < 1 or double_width % 2 or kv.shape[-1] != double_width:
        raise ValueError("CSA projections and position_bias must have shape [*, *, 2*head_dim]")
    head_dim = double_width // 2
    complete = joined_kv.shape[1] // ratio
    used = complete * ratio
    if not complete:
        prior_tokens = 0 if state is None else state.total_tokens
        return kv.new_empty((kv.shape[0], 0, head_dim)), GatedCompressorState(
            joined_kv,
            joined_gate,
            None if state is None else state.overlap_kv,
            None if state is None else state.overlap_gate,
            prior_tokens + kv.shape[1],
        )

    windows = joined_kv[:, :used].view(kv.shape[0], complete, ratio, double_width)
    logits = joined_gate[:, :used].view_as(windows) + position_bias
    combined_kv = kv.new_zeros((kv.shape[0], complete, 2 * ratio, head_dim))
    combined_gate = gate.new_full(
        (kv.shape[0], complete, 2 * ratio, head_dim), float("-inf")
    )
    combined_kv[:, :, ratio:] = windows[..., head_dim:]
    combined_gate[:, :, ratio:] = logits[..., head_dim:]
    if complete > 1:
        combined_kv[:, 1:, :ratio] = windows[:, :-1, :, :head_dim]
        combined_gate[:, 1:, :ratio] = logits[:, :-1, :, :head_dim]
    if state is not None and state.overlap_kv is not None:
        if state.overlap_kv.shape != (kv.shape[0], ratio, head_dim):
            raise ValueError("CSA overlap state has the wrong shape")
        combined_kv[:, 0, :ratio] = state.overlap_kv
        combined_gate[:, 0, :ratio] = state.overlap_gate

    weights = combined_gate.softmax(dim=2, dtype=torch.float32).to(kv.dtype)
    output = (combined_kv * weights).sum(dim=2)
    prior_tokens = 0 if state is None else state.total_tokens
    return output, GatedCompressorState(
        joined_kv[:, used:],
        joined_gate[:, used:],
        windows[:, -1, :, :head_dim],
        logits[:, -1, :, :head_dim],
        prior_tokens + kv.shape[1],
    )


def compress_chunk(
    hidden: torch.Tensor,
    ape: torch.Tensor,
    stride: int,
    state: CompressorState | None = None,
) -> tuple[torch.Tensor, CompressorState]:
    """Reduce complete windows and retain the incomplete suffix."""
    if hidden.ndim != 2 or ape.ndim != 1 or stride < 1 or ape.numel() < stride:
        raise ValueError("invalid compressor shape or stride")
    carry = hidden.new_empty((0, hidden.shape[-1])) if state is None else state.carry
    if carry.ndim != 2 or carry.shape[-1] != hidden.shape[-1]:
        raise ValueError("carry and hidden dimensions do not agree")
    joined = torch.cat((carry, hidden), dim=0)
    complete = joined.shape[0] // stride
    used = complete * stride
    if complete:
        windows = joined[:used].view(complete, stride, hidden.shape[-1])
        output = (windows * ape[:stride].to(hidden)[:, None]).sum(dim=1)
    else:
        output = hidden.new_empty((0, hidden.shape[-1]))
    prior = 0 if state is None else state.total_tokens
    return output, CompressorState(joined[used:], prior + hidden.shape[0])
