# SPDX-License-Identifier: Apache-2.0
"""Shard-at-a-time conversion into final DeepSeek destination tensors."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StreamingLoadStats:
    source_bytes: int
    destination_bytes: int
    peak_temporary_bytes: int
    tensors_loaded: int


def stream_into_final_tensors(
    sources: Iterable[tuple[str, torch.Tensor]],
    destinations: Mapping[str, torch.Tensor],
    *,
    convert: Callable[[torch.Tensor, torch.dtype], torch.Tensor] | None = None,
) -> StreamingLoadStats:
    """Convert and copy one source tensor at a time into final storage.

    ``sources`` may lazily read individual safetensors shards. The function
    never accumulates source tensors and never constructs a second full model.
    The returned temporary peak is the largest simultaneous source+converted
    tensor, suitable for feeding the P7 analytical budget.
    """
    seen: set[str] = set()
    source_bytes = destination_bytes = peak = count = 0
    for name, source in sources:
        if name in seen:
            raise ValueError(f"duplicate source tensor {name!r}")
        seen.add(name)
        if name not in destinations:
            raise KeyError(f"no final destination for {name!r}")
        destination = destinations[name]
        source_nbytes = source.numel() * source.element_size()
        source_bytes += source_nbytes
        if convert is None:
            converted = source.to(destination.dtype)
        else:
            converted = convert(source, destination.dtype)
        if converted.shape != destination.shape:
            raise ValueError(
                f"shape mismatch for {name!r}: converted {tuple(converted.shape)}, "
                f"destination {tuple(destination.shape)}"
            )
        converted_nbytes = converted.numel() * converted.element_size()
        peak = max(peak, source_nbytes + converted_nbytes)
        destination.copy_(converted)
        destination_bytes += destination.numel() * destination.element_size()
        count += 1
        del converted
        del source
    missing = set(destinations) - seen
    if missing:
        raise KeyError(f"destinations were not loaded: {sorted(missing)}")
    return StreamingLoadStats(source_bytes, destination_bytes, peak, count)


def dequantize_symmetric(
    packed: torch.Tensor,
    destination_dtype: torch.dtype,
    *,
    scale: float,
) -> torch.Tensor:
    """Small-scale reference conversion used to validate streaming mechanics."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    return (packed.float() * scale).to(destination_dtype)
