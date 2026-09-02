# SPDX-License-Identifier: Apache-2.0
"""Quantizing Qwen3.5 linear weights to per-channel FP8 as they load.

The Neuron side of this is ``QuantizationType.ROW``: weights carry one fp32
dequant scale per output channel, and *activation* scales are computed on
device at runtime. That choice is what makes this file possible at all --
the alternative, ``QuantizationType.STATIC``, needs calibrated ``input_scale``
tensors, and neither the BF16 checkpoint nor the official FP8 release
(``activation_scheme="dynamic"``) ships any.

Because ROW derives its weight scales from the weights themselves, no
calibration set and no offline artifact is needed: the BF16 checkpoint already
on disk is quantized here, on the loader thread, one shard at a time.

Layout
------
Every weight this file quantizes is stored ``[contraction, output_channel]``
-- ``NF.mlp`` takes ``gate/up [H, I]`` and ``down [I, H]``, ``NF.qkv_proj``
takes ``[H, q_gate + 2*kv]``, and ``NF.o_proj`` takes ``[N*D, H]``. So the
reduction that produces a row scale is always over dim 0, and the scale vector
is always as long as dim 1. nkilib wants it broadcast across the 128
partitions: ``(128, N)`` fp32 (``mlp_parameters.py:222-234``).

Sharding
--------
A rank quantizes only the shard it loaded, and never needs to agree with any
other rank:

* ``gate``/``up``/``qkv`` are sharded along the **output** dim, so each rank
  owns whole output channels and its amax over dim 0 is the true amax.
* ``down``/``o_proj`` are sharded along the **contraction** dim, so each rank
  holds a partial column. Its amax is therefore only over its own rows -- and
  that is correct, not an approximation: the rank dequantizes its own partial
  product ``x_r @ W_r`` before the all-reduce, so ``W_r`` is reconstructed
  exactly from ``W_r_fp8 * scale_r`` whatever the other ranks chose. A tighter
  per-rank scale is strictly better than a shared one here.

Range
-----
Scales divide by :data:`~vllm_neuron.utils.dtype_utils.FP8_CLAMP_MAX`, which
is **240.0** on Trn2 and 448.0 on Trn3. Trn2's tensor engine speaks legacy
``nl.float8_e4m3`` (max finite 240, inf/NaN reserved), not OCP
``float8_e4m3fn`` (max finite 448); the two encodings agree on every code
below 240 and differ only in the 14 codes above it, so clamping to 240 keeps
the values exactly representable on both. Hardcoding 448 here would silently
produce values Trn2 reads as inf or NaN.
"""

import torch

from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

#: Storage dtype. Torch has no legacy-e4m3 dtype, and it does not need one:
#: the bytes below 240 are identical in both encodings, so ``float8_e4m3fn``
#: is a faithful container for values the Trn2 kernel will read as e4m3.
FP8_DTYPE = torch.float8_e4m3fn

#: Partition dim the kernels broadcast a scale across (``nl.tile_size.pmax``).
_PMAX = 128

#: Guards against a division by zero for an all-zero output channel. Far below
#: any real weight magnitude, so it never perturbs a channel that has content.
_MIN_SCALE = 1e-12


def row_scale(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-channel dequant scale for a ``[contraction, out]`` weight.

    Returned as fp32 of shape ``[out]``. Computed in fp32 regardless of the
    input dtype: a bf16 amax reduction over thousands of rows loses precision
    exactly where it matters, at the top of the range that sets the scale.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"expected a 2-D [contraction, out] weight, got {tuple(weight.shape)}"
        )
    amax = weight.float().abs().amax(dim=0)
    return (amax / FP8_CLAMP_MAX).clamp_min(_MIN_SCALE)


def quantize_row_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[contraction, out]`` bf16 -> ``(fp8 weight, fp32 [out] scale)``.

    The clamp is not belt-and-braces: dividing by ``amax / FP8_CLAMP_MAX``
    puts the largest magnitude exactly at ``FP8_CLAMP_MAX``, where rounding to
    the nearest representable value can land one ulp above it. On Trn2 that
    code is an inf.
    """
    scale = row_scale(weight)
    scaled = (weight.float() / scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
    return scaled.to(FP8_DTYPE), scale


def dequantize_row_fp8(
    weight_fp8: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`quantize_row_fp8`, for CPU parity checks."""
    return weight_fp8.float() * scale.reshape(1, -1)


def broadcast_row_scale(scale: torch.Tensor) -> torch.Tensor:
    """``[out]`` fp32 -> the ``(128, out)`` fp32 the nkilib kernels assert.

    Expanded into a real allocation rather than a stride-0 view: the tensor is
    handed to a compiled kernel, and a broadcast view is not contiguous.
    """
    return scale.reshape(1, -1).expand(_PMAX, -1).contiguous().float()


def fp8_weight_loader(base: SafetensorsWeightLoader) -> SafetensorsWeightLoader:
    """Wrap a sharding loader so its result lands as FP8.

    Composed around the model's existing loader rather than replacing it, so
    the sharding, transposition and fusion logic stays in one place and this
    file only changes the dtype. Same idiom as llama3's
    ``_wrap_with_fp8_downscale``.
    """
    base_transform = base.transform or (lambda slices, rank: slices[0][:])

    def transform(slices, rank):
        return quantize_row_fp8(base_transform(slices, rank))[0]

    return SafetensorsWeightLoader(transform=transform)


def fp8_scale_loader(base: SafetensorsWeightLoader) -> SafetensorsWeightLoader:
    """The matching ``(128, out)`` scale for :func:`fp8_weight_loader`.

    Reads the same checkpoint slice as the weight loader and recomputes the
    amax. That is a deliberate second read: the two parameters are loaded
    independently by a thread pool, and threading a cache between them would
    add cross-parameter state to the one place in the loader that is
    concurrent. The read hits the page cache the weight load just warmed.
    """
    base_transform = base.transform or (lambda slices, rank: slices[0][:])

    def transform(slices, rank):
        return broadcast_row_scale(row_scale(base_transform(slices, rank)))

    return SafetensorsWeightLoader(transform=transform)
