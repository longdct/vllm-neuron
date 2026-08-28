#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dequantize official DeepSeek-V4 weights to BF16.

The published checkpoint mixes two quantized formats, distinguished here by the
*stored dtype* rather than by config, so a config drift cannot silently pick the
wrong unpacking:

``float8_e4m3fn``
    Attention projections and shared experts. Block-scaled with one
    ``float8_e8m0fnu`` scale per ``[128, 128]`` tile.

``int8``
    Routed experts, holding **two** ``float4_e2m1fn`` values per byte packed
    along the input dimension, with one ``float8_e8m0fnu`` scale per 32 input
    elements (MXFP4). The stored shape is therefore ``[out, in // 2]``.

Semantics follow the official ``inference/convert.py`` and ``inference/kernel.py``
from the checkpoint repository -- in particular the FP4 code table and the
low-nibble-first packing order, both of which are silent-corruption traps if
guessed.
"""

from __future__ import annotations

import torch

#: Official ``float4_e2m1fn`` code table (``inference/convert.py``). Index is the
#: 4-bit code; the top bit is the sign, so entries 8..15 mirror 0..7.
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

#: Tile height/width of an FP8 weight scale.
FP8_BLOCK = 128

#: Number of input elements sharing one FP4 scale.
FP4_BLOCK = 32


def _expand_scale(scale: torch.Tensor, rows: int, cols: int,
                  row_block: int, col_block: int) -> torch.Tensor:
    """Broadcast a block scale up to full ``[rows, cols]``.

    ``repeat_interleave`` then crop, rather than reshape tricks: the trailing
    block is allowed to be partial and cropping keeps that case correct.
    """
    expanded = scale.float()
    if row_block > 1:
        expanded = expanded.repeat_interleave(row_block, dim=0)
    if col_block > 1:
        expanded = expanded.repeat_interleave(col_block, dim=1)
    if expanded.shape[0] < rows or expanded.shape[1] < cols:
        raise ValueError(
            f"scale {tuple(scale.shape)} expands to {tuple(expanded.shape)}, "
            f"too small for weight [{rows}, {cols}]"
        )
    return expanded[:rows, :cols]


def dequantize_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``float8_e4m3fn`` + one scale per ``[128, 128]`` tile -> float32."""
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D fp8 weight, got {tuple(weight.shape)}")
    rows, cols = weight.shape
    expected = ((rows + FP8_BLOCK - 1) // FP8_BLOCK,
                (cols + FP8_BLOCK - 1) // FP8_BLOCK)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"fp8 scale {tuple(scale.shape)} does not match weight "
            f"{tuple(weight.shape)} (expected {expected})"
        )
    return weight.float() * _expand_scale(scale, rows, cols, FP8_BLOCK, FP8_BLOCK)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """``int8`` holding two FP4 codes per byte -> float32 ``[out, in]``.

    Low nibble first: output column ``2j`` is the low nibble of byte ``j`` and
    column ``2j + 1`` is the high nibble. Reversing this silently produces
    plausible-looking garbage.
    """
    if packed.ndim != 2:
        raise ValueError(f"expected a 2-D packed weight, got {tuple(packed.shape)}")
    raw = packed.view(torch.uint8)
    low = FP4_TABLE.to(raw.device)[(raw & 0x0F).long()]
    high = FP4_TABLE.to(raw.device)[((raw >> 4) & 0x0F).long()]
    rows, half = raw.shape
    return torch.stack([low, high], dim=-1).reshape(rows, half * 2)


def dequantize_fp4_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 (``int8``-packed) + one scale per 32 input elements -> float32."""
    values = unpack_fp4(weight)
    rows, cols = values.shape
    expected = (rows, (cols + FP4_BLOCK - 1) // FP4_BLOCK)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"fp4 scale {tuple(scale.shape)} does not match unpacked weight "
            f"[{rows}, {cols}] (expected {expected})"
        )
    return values * _expand_scale(scale, rows, cols, 1, FP4_BLOCK)


def dequantize(weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """Dispatch on the stored dtype and return BF16.

    A tensor with no companion scale is already in a real dtype (norms,
    ``attn_sink``, the mHC parameters, the router) and passes through untouched
    apart from the BF16 cast.
    """
    if scale is None:
        return weight if weight.dtype == torch.bfloat16 else weight.to(torch.bfloat16)
    if weight.dtype == torch.int8:
        return dequantize_fp4_blockwise(weight, scale).to(torch.bfloat16)
    if weight.dtype == torch.float8_e4m3fn:
        return dequantize_fp8_blockwise(weight, scale).to(torch.bfloat16)
    raise ValueError(f"unexpected quantized dtype {weight.dtype} alongside a scale")
