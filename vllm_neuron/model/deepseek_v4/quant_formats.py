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


#: TRN2 implements ``float8_e4m3`` (max finite 240), not OCP ``e4m3fn``
#: (max 448). torch has no dtype for the former, so weights are carried as
#: ``float8_e4m3fn`` and the NKI tracer reinterprets them under
#: ``UNSAFE_FP8FNCAST=1``. Quantizing against 240 is what makes that
#: reinterpretation lossless rather than merely "unsafe".
FP8_E4M3_MAX_TRN2 = 240.0

#: Widest per-output-channel E8M0 exponent spread that a re-block can absorb
#: while every FP4 magnitude stays inside e4m3's normal range. FP4 spans
#: ``2^-1..2^2.585``; e4m3 normals bottom out at ``2^-6``; so a downshift of 5
#: binades still lands at ``2^-6``, and 6 would go subnormal and truncate.
#: Measured spread on the real shards is at most 3 (layers 0/2/3).
MAX_EXACT_CHANNEL_SPREAD = 5


def requantize_fp4_to_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    strict: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 ``[out, in//2]`` + per-32 E8M0 -> FP8 ``[out, in]`` + per-channel.

    TRN2 cannot execute FP4, so the routed experts have to be widened. The
    conversion is *exact*, not approximate, and the reason is worth stating
    because it is what licenses asserting bitwise equality downstream:

      * every ``float4_e2m1fn`` code is representable in ``e4m3`` -- one
        mantissa bit fits in three, two exponent bits fit in four;
      * both the source and destination scales are powers of two, so folding
        one into the other only shifts an exponent and never touches a
        mantissa.

    The single failure mode is range. Collapsing the per-32 scales of one
    output channel onto that channel's largest exponent shifts the other
    blocks *down* by their exponent difference. FP4 magnitudes start at
    ``2^-1``, so a spread wider than ``MAX_EXACT_CHANNEL_SPREAD`` pushes the
    smallest values below e4m3's ``2^-6`` normal floor, where they truncate.
    ``strict`` refuses that case rather than silently degrading -- the shards
    on disk peak at a spread of 3, but 40 of 43 layers have never been seen.

    Returns ``(fp8_weight, channel_scale)`` where ``channel_scale`` is float32
    of shape ``[out]`` and ``fp8_weight[r, c] * channel_scale[r]`` reproduces
    the MXFP4 value exactly.
    """
    if scale.ndim != 2:
        raise ValueError(f"expected a 2-D fp4 scale, got {tuple(scale.shape)}")
    values = unpack_fp4(weight)
    rows, cols = values.shape
    expected = (rows, (cols + FP4_BLOCK - 1) // FP4_BLOCK)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"fp4 scale {tuple(scale.shape)} does not match unpacked weight "
            f"[{rows}, {cols}] (expected {expected})"
        )

    # ``scale`` arrives decoded, matching ``dequantize_fp4_blockwise``: reading
    # a checkpoint's ``float8_e8m0fnu`` scale yields the power of two itself,
    # not the biased exponent byte. Recover the exponent by log2, which is
    # exact for a power of two, and do the folding in exponent space so no
    # step can round.
    decoded = scale.float()
    if not torch.all(decoded > 0):
        raise ValueError("fp4 scales must be positive powers of two")
    exponent = torch.log2(decoded).round().to(torch.int32)
    if not torch.equal(torch.exp2(exponent.float()), decoded):
        raise ValueError("fp4 scales must be exact powers of two")

    channel_exponent = exponent.amax(dim=1)
    spread = int((channel_exponent[:, None] - exponent).amax())
    if strict and spread > MAX_EXACT_CHANNEL_SPREAD:
        raise ValueError(
            f"per-output-channel exponent spread {spread} exceeds "
            f"{MAX_EXACT_CHANNEL_SPREAD}; folding it into e4m3 would push "
            "values subnormal and lose mantissa bits. Re-quantizing this "
            "tensor is not exact -- handle it explicitly rather than "
            "silently degrading."
        )

    # Shift each 32-block down onto its channel's exponent, then expand to
    # full width. exp2 of a non-positive integer is exact.
    relative = _expand_scale(
        torch.exp2((exponent - channel_exponent[:, None]).float()),
        rows,
        cols,
        1,
        FP4_BLOCK,
    )
    elements = (values * relative).to(torch.float8_e4m3fn)
    channel_scale = torch.exp2(channel_exponent.float())
    return elements, channel_scale


def dequantize_fp8_per_channel(
    weight: torch.Tensor, channel_scale: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`requantize_fp4_to_fp8`, for verification."""
    if channel_scale.ndim != 1 or channel_scale.shape[0] != weight.shape[0]:
        raise ValueError(
            f"channel scale {tuple(channel_scale.shape)} does not match "
            f"weight {tuple(weight.shape)}"
        )
    return weight.float() * channel_scale[:, None]
