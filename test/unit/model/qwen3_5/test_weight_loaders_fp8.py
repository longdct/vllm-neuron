# SPDX-License-Identifier: Apache-2.0
"""Per-channel FP8 quantization of Qwen3.5 linear weights at load time."""

import pytest
import torch

from vllm_neuron.model.qwen3_5.weight_loaders_fp8 import (
    FP8_DTYPE,
    broadcast_row_scale,
    dequantize_row_fp8,
    fp8_scale_loader,
    fp8_weight_loader,
    quantize_row_fp8,
    row_scale,
)
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

_PMAX = 128


def _weight(k=256, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, n, generator=g, dtype=torch.float32) * 0.02


# ---------------------------------------------------------------------------
# The scale itself
# ---------------------------------------------------------------------------


def test_scale_is_one_value_per_output_channel():
    w = _weight(k=256, n=64)
    assert row_scale(w).shape == (64,)


def test_scale_puts_the_largest_magnitude_at_the_clamp():
    """The whole point of the scale: use the full FP8 range, no more."""
    w = _weight()
    scaled = w.float() / row_scale(w)
    assert torch.isclose(
        scaled.abs().amax(), torch.tensor(FP8_CLAMP_MAX), rtol=1e-5
    )


def test_a_1d_weight_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        row_scale(torch.zeros(8))


def test_an_all_zero_channel_does_not_divide_by_zero():
    w = _weight()
    w[:, 3] = 0.0
    scale = row_scale(w)
    assert torch.isfinite(scale).all()
    assert scale[3] > 0
    q, _ = quantize_row_fp8(w)
    assert torch.isfinite(q.float()).all()
    assert (q.float()[:, 3] == 0).all()


def test_scale_is_computed_in_fp32_even_for_a_bf16_weight():
    """A bf16 amax reduction loses precision exactly at the top of the range,
    which is the value that sets the scale."""
    w = _weight()
    assert row_scale(w.to(torch.bfloat16)).dtype == torch.float32


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_quantize_produces_fp8_and_a_matching_scale():
    w = _weight(k=256, n=64)
    q, s = quantize_row_fp8(w)
    assert q.dtype is FP8_DTYPE
    assert q.shape == w.shape
    assert s.shape == (64,)


def test_no_value_exceeds_the_platform_clamp():
    """Above FP8_CLAMP_MAX, Trn2 reads the code as inf or NaN. Rounding at the
    boundary can push a value one ulp over, so this is a real risk, not a
    theoretical one."""
    for seed in range(8):
        q, _ = quantize_row_fp8(_weight(seed=seed))
        assert q.float().abs().amax() <= FP8_CLAMP_MAX
        assert torch.isfinite(q.float()).all()


def test_round_trip_error_matches_what_was_measured_on_the_real_checkpoint():
    """~2.65% relative Frobenius error is the intrinsic cost of e4m3's 3-bit
    mantissa at a 240 clamp -- measured across layers 3/31/59 of the real 27B.
    A large jump means the scale is wrong; a large drop means this test is not
    exercising the quantizer."""
    w = _weight(k=1024, n=256)
    q, s = quantize_row_fp8(w)
    err = (dequantize_row_fp8(q, s) - w).norm() / w.norm()
    assert 0.01 < err < 0.05, err


def test_a_channel_with_a_much_smaller_range_keeps_its_precision():
    """The reason for per-channel rather than per-tensor scales: a quiet
    channel must not be quantized against a loud channel's amax."""
    w = _weight(k=256, n=8)
    w[:, 0] *= 1e-3
    q, s = quantize_row_fp8(w)
    recon = dequantize_row_fp8(q, s)
    quiet = (recon[:, 0] - w[:, 0]).norm() / w[:, 0].norm()
    loud = (recon[:, 1] - w[:, 1]).norm() / w[:, 1].norm()
    assert quiet < 2 * loud


# ---------------------------------------------------------------------------
# Kernel-facing scale shape
# ---------------------------------------------------------------------------


def test_scale_is_broadcast_to_the_shape_the_kernel_asserts():
    s = row_scale(_weight(n=64))
    b = broadcast_row_scale(s)
    assert b.shape == (_PMAX, 64)
    assert b.dtype is torch.float32
    assert torch.equal(b[0], s)
    assert torch.equal(b[_PMAX - 1], s)


def test_the_broadcast_scale_is_materialized_not_a_stride_zero_view():
    """It is handed to a compiled kernel, which cannot take a non-contiguous
    broadcast view."""
    b = broadcast_row_scale(row_scale(_weight(n=64)))
    assert b.is_contiguous()
    assert b.stride(0) != 0


# ---------------------------------------------------------------------------
# Loader composition and sharding
# ---------------------------------------------------------------------------


class _FakeSlice:
    """Stands in for a PySafeSlice: indexable, and reports its shape."""

    def __init__(self, tensor):
        self._t = tensor

    def __getitem__(self, key):
        return self._t[key]

    def get_shape(self):
        return list(self._t.shape)


def _column_shard_loader(shard_size, num_shards):
    """Shards along dim 1 (output channels), like gate/up."""
    del num_shards

    def transform(slices, rank):
        return slices[0][:, rank * shard_size : (rank + 1) * shard_size]

    return SafetensorsWeightLoader(transform=transform)


def test_weight_loader_wraps_the_shard_and_returns_fp8():
    w = _weight(k=128, n=64)
    base = _column_shard_loader(shard_size=16, num_shards=4)
    out = fp8_weight_loader(base).load([_FakeSlice(w)], rank=2)
    assert out.dtype is FP8_DTYPE
    assert out.shape == (128, 16)


def test_scale_loader_matches_the_weight_loader_shard_for_shard():
    w = _weight(k=128, n=64)
    base = _column_shard_loader(shard_size=16, num_shards=4)
    for rank in range(4):
        q = fp8_weight_loader(base).load([_FakeSlice(w)], rank)
        s = fp8_scale_loader(base).load([_FakeSlice(w)], rank)
        assert s.shape == (_PMAX, 16)
        recon = dequantize_row_fp8(q, s[0])
        expected = w[:, rank * 16 : (rank + 1) * 16]
        assert (recon - expected).norm() / expected.norm() < 0.05


def test_output_sharded_ranks_reproduce_the_unsharded_quantization():
    """gate/up/qkv own whole output channels, so sharding must not change a
    single value -- the amax over the contraction dim is complete on each rank."""
    w = _weight(k=128, n=64)
    whole_q, whole_s = quantize_row_fp8(w)
    base = _column_shard_loader(shard_size=16, num_shards=4)
    for rank in range(4):
        q = fp8_weight_loader(base).load([_FakeSlice(w)], rank)
        s = fp8_scale_loader(base).load([_FakeSlice(w)], rank)
        cols = slice(rank * 16, (rank + 1) * 16)
        assert torch.equal(q.float(), whole_q.float()[:, cols])
        assert torch.allclose(s[0], whole_s[cols])


def test_contraction_sharded_ranks_may_disagree_and_that_is_correct():
    """down/o_proj split the contraction dim, so each rank's amax covers only
    its own rows. The scales differ between ranks by design: each rank
    dequantizes its own partial product before the all-reduce, so the sum
    still reconstructs the full matmul."""
    w = _weight(k=128, n=32)

    def row_shard(slices, rank):
        return slices[0][rank * 64 : (rank + 1) * 64, :]

    base = SafetensorsWeightLoader(transform=row_shard)
    x = torch.randn(4, 128, generator=torch.Generator().manual_seed(7))

    partial = torch.zeros(4, 32)
    for rank in range(2):
        q = fp8_weight_loader(base).load([_FakeSlice(w)], rank)
        s = fp8_scale_loader(base).load([_FakeSlice(w)], rank)
        rows = slice(rank * 64, (rank + 1) * 64)
        partial += x[:, rows] @ dequantize_row_fp8(q, s[0])

    reference = x @ w
    assert (partial - reference).norm() / reference.norm() < 0.05


def test_a_loader_without_a_transform_still_works():
    """The base loader's transform is optional; the wrapper must not assume it."""
    w = _weight(k=64, n=16)
    out = fp8_weight_loader(SafetensorsWeightLoader()).load([_FakeSlice(w)], rank=0)
    assert out.dtype is FP8_DTYPE
    assert out.shape == (64, 16)
