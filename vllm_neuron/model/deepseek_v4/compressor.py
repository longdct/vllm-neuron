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
