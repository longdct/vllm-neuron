# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet tensor-parallel shard invariance.

The property under test is the one that makes TP correct at all: **summing
every rank's output must reproduce the unsharded layer, per element.** The
layer's ``out_proj`` is column-sharded, so at runtime that sum is the exit
all-reduce; here it is done explicitly, in one process, with no process group.

Three regimes are covered, using a tiny config whose 2 key heads and 6 value
heads reach each of them at a size a test can hold:

  tp=2   pure head sharding                  (the shipped 27B at tp<=16)
  tp=4   each value head's dim split in two  (the shipped 27B at tp=32)
  tp=8   split four ways                     (beyond what the 27B needs)

Once the value dimension is split, the gated RMSNorm becomes the one piece that
is *not* separable -- it normalizes over the full ``linear_value_head_dim``, and
a rank sees only part of it. That is what ``_all_reduce_head_sums`` exists for,
and the accumulator below stands in for the collective.

Everything runs in fp32 on CPU and is compared per element, never in aggregate:
the DeepSeek-V4 record is explicit that an aggregate metric read an exact
off-by-one at 2 of 8 positions as floating-point noise
(docs/model-dev/deepseek-v4-real-weight-validation.md:38-53).
"""

import pytest
import torch

from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig
from vllm_neuron.model.qwen3_5.gated_deltanet import Qwen3_5GatedDeltaNet
from vllm_neuron.model.qwen3_5.parallel import resolve_sharding
from vllm_neuron.model.qwen3_5.weight_loaders import (
    gdn_conv1d_weight_loader,
    gdn_gated_norm_loader,
    gdn_head_vector_loader,
    gdn_out_proj_weight_loader,
    gdn_qkv_weight_loader,
    gdn_row_weight_loader,
)

#: Degrees the tiny config admits. 1 is included so the harness itself is
#: pinned against the path it is meant to reproduce.
TP_DEGREES = [1, 2, 4, 8]

#: fp32 on CPU through ~10 matmuls and an rsqrt. Tight enough that a misplaced
#: head or a half-width norm cannot hide under it -- those are O(1) errors.
ATOL = 1e-5
RTOL = 1e-5


def _config(**overrides):
    base = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        # 8 rather than 6: these tests sweep tp=4, and 6 query heads over 4
        # ranks gives 2 per rank against a GQA group of 3, so rank 1 would own
        # heads 2 and 3 -- one from each group. resolve_sharding rejects that.
        # Attention geometry is incidental here; the suite is about the GDN.
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        torch_dtype=torch.float32,
    )
    base.update(overrides)
    return Qwen3_5TextConfig(**base)


class HeadSumAccumulator:
    """Stands in for the cross-rank sum-of-squares all-reduce.

    ``forward`` is pure, so each shard can simply be run twice: pass 1 records
    every rank's local per-head sums into a shared buffer, pass 2 hands each
    rank the accumulated totals for its own columns. Doing it this way keeps
    the production path free of a test-only injection seam -- the only thing
    replaced is the collective itself.
    """

    def __init__(self, num_v_heads: int):
        self.num_v_heads = num_v_heads
        self.buffer = None
        self.recording = True

    def attach(self, layer):
        def reduce(local_sums):
            cols = torch.tensor(layer.global_v_heads, dtype=torch.long)
            if self.recording:
                if self.buffer is None:
                    self.buffer = torch.zeros(local_sums.shape[0], self.num_v_heads)
                self.buffer.index_add_(1, cols, local_sums)
                # Pass 1's output is discarded; only the accumulation matters.
                return local_sums
            return self.buffer.index_select(1, cols)

        layer._all_reduce_head_sums = reduce


def _build_reference(config, seed=0):
    torch.manual_seed(seed)
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval().float()
    with torch.no_grad():
        # Default init leaves A_log at zero and dt_bias at one, which makes the
        # decay identical for every head and would hide a head permutation.
        for param in layer.parameters():
            param.copy_(torch.randn_like(param) * 0.2)
    return layer


def _build_shard(config, policy, rank, reference):
    """A rank's layer, loaded from the reference through the real loaders.

    Going through the production loaders rather than slicing by hand is
    deliberate: it proves the loader partition and the module sizes agree, so
    a shard that is merely self-consistent cannot pass.
    """
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0, policy=policy).eval().float()
    ref = reference

    pairs = [
        (layer.in_proj_qkv.weight, ref.in_proj_qkv.weight, gdn_qkv_weight_loader),
        (layer.conv1d.weight, ref.conv1d.weight, gdn_conv1d_weight_loader),
        (layer.in_proj_z.weight, ref.in_proj_z.weight, gdn_row_weight_loader),
        (layer.out_proj.weight, ref.out_proj.weight, gdn_out_proj_weight_loader),
        (layer.in_proj_b.weight, ref.in_proj_b.weight, gdn_head_vector_loader),
        (layer.in_proj_a.weight, ref.in_proj_a.weight, gdn_head_vector_loader),
        (layer.dt_bias, ref.dt_bias, gdn_head_vector_loader),
        (layer.A_log, ref.A_log, gdn_head_vector_loader),
        (layer.norm.weight, ref.norm.weight, gdn_gated_norm_loader),
    ]
    with torch.no_grad():
        for dest, source, make_loader in pairs:
            shard = make_loader(policy).load([source.detach()], rank=rank)
            assert shard.shape == dest.shape, (dest.shape, shard.shape)
            dest.copy_(shard)

    # A real rank learns who it is from the process group. There is none here,
    # so every simulated shard would otherwise believe it is rank 0 -- claiming
    # the same value-head columns as its peers and summing four half-widths
    # into one head instead of two.
    layer.rank = rank
    layer.global_v_heads = policy.v_head_indices(rank)
    layer.needs_norm_allreduce = policy.gated_norm_needs_allreduce
    return layer


def _run_shards(config, tp, reference, hidden, states=None, use_recurrent=False):
    """Every rank's ``forward``, with the gated-norm reduction simulated.

    Returns ``(outputs, states)``, one entry per rank.
    """
    policy = resolve_sharding(config, tp)
    shards = [_build_shard(config, policy, r, reference) for r in range(tp)]

    accumulator = HeadSumAccumulator(policy.num_v_heads)
    for layer in shards:
        accumulator.attach(layer)

    def sweep():
        results = []
        for rank, layer in enumerate(shards):
            conv, rec = (None, None) if states is None else states[rank]
            results.append(
                layer(
                    hidden,
                    conv_state=conv,
                    recurrent_state=rec,
                    use_recurrent=use_recurrent,
                )
            )
        return results

    if policy.gated_norm_needs_allreduce:
        sweep()  # pass 1: fill the accumulator
        accumulator.recording = False
    results = sweep()

    return [r[0] for r in results], [(r[1], r[2]) for r in results]


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp", TP_DEGREES)
@pytest.mark.parametrize("seq_len", [128, 8])
def test_summed_shards_reproduce_the_unsharded_prefill(tp, seq_len):
    """seq_len 128 crosses a chunk boundary; 8 is shorter than chunk_size."""
    config = _config()
    reference = _build_reference(config)
    hidden = torch.randn(2, seq_len, config.hidden_size)

    expected = reference(hidden)[0]
    outputs, _ = _run_shards(config, tp, reference, hidden)

    total = torch.stack(outputs).sum(0)
    torch.testing.assert_close(total, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("tp", TP_DEGREES)
def test_summed_shards_reproduce_the_unsharded_decode(tp):
    config = _config()
    reference = _build_reference(config)
    hidden = torch.randn(3, 1, config.hidden_size)

    expected = reference(hidden, use_recurrent=True)[0]
    outputs, _ = _run_shards(config, tp, reference, hidden, use_recurrent=True)

    total = torch.stack(outputs).sum(0)
    torch.testing.assert_close(total, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("tp", TP_DEGREES)
def test_state_handoff_survives_sharding(tp):
    """Prefill, then decode from the state each rank just produced.

    A shard whose state layout disagreed with its own projections would still
    pass the single-shot tests above; only carrying the state forward catches
    it. This is the degenerate regime the plan's section 4.3 calls out.
    """
    config = _config()
    reference = _build_reference(config)
    prompt = torch.randn(2, 96, config.hidden_size)
    step = torch.randn(2, 1, config.hidden_size)

    _, ref_conv, ref_rec = reference(prompt)
    expected = reference(
        step, conv_state=ref_conv, recurrent_state=ref_rec, use_recurrent=True
    )[0]

    _, states = _run_shards(config, tp, reference, prompt)
    outputs, _ = _run_shards(
        config, tp, reference, step, states=states, use_recurrent=True
    )

    total = torch.stack(outputs).sum(0)
    torch.testing.assert_close(total, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("tp", TP_DEGREES)
def test_shard_state_shapes_match_the_policy(tp):
    """The state each rank emits must be exactly what get_kv_spec sizes for."""
    config = _config()
    policy = resolve_sharding(config, tp)
    reference = _build_reference(config)
    hidden = torch.randn(2, 32, config.hidden_size)

    _, states = _run_shards(config, tp, reference, hidden)

    for rank, (conv, rec) in enumerate(states):
        assert conv.shape == (
            2,
            policy.conv_dim_per_rank,
            config.linear_conv_kernel_dim - 1,
        ), rank
        assert rec.shape == (
            2,
            policy.v_heads_per_rank,
            config.linear_key_head_dim,
            policy.v_dim_per_rank,
        ), rank


def test_gated_norm_is_wrong_without_the_cross_rank_sum():
    """Guard the guard: the reduction must actually be load-bearing at tp=4.

    If this ever passes, the value dimension is no longer being split and the
    tp=32 path has silently degenerated into something else.
    """
    config = _config()
    tp = 4
    policy = resolve_sharding(config, tp)
    assert policy.gated_norm_needs_allreduce

    reference = _build_reference(config)
    hidden = torch.randn(2, 32, config.hidden_size)
    expected = reference(hidden)[0]

    # Same shards, but each normalizing over its own quarter-width only.
    shards = [_build_shard(config, policy, r, reference) for r in range(tp)]
    for layer in shards:
        layer.needs_norm_allreduce = False
    total = torch.stack([layer(hidden)[0] for layer in shards]).sum(0)

    assert not torch.allclose(total, expected, atol=ATOL, rtol=RTOL)
