# SPDX-License-Identifier: Apache-2.0
"""Chunk-invariant strided compressor reference with explicit carry state."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompressorState:
    carry: torch.Tensor
    total_tokens: int = 0


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

