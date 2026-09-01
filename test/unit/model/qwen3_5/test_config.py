# SPDX-License-Identifier: Apache-2.0
"""Config normalization and TP sharding policy for the Qwen3.5 family.

These are pure-CPU tests: no torch device, no NKI, no vLLM engine. They pin the
geometry that every later phase depends on, and they pin the *rejections* --
a config that quietly defaults is a config that quietly narrows the gate.
"""

import collections

import pytest

from vllm_neuron.model.qwen3_5.config import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen3_5TextConfig,
)
from vllm_neuron.model.qwen3_5.parallel import resolve_sharding


# ---------------------------------------------------------------------------
# Geometry of the shipped 27B checkpoint
# ---------------------------------------------------------------------------


def test_default_matches_qwen38_27b():
    """Defaults reproduce the Qwen3.8-27B / Qwen3.6-27B text_config."""
    c = Qwen3_5TextConfig()

    assert c.num_hidden_layers == 64
    assert len(c.linear_layer_indices) == 48
    assert len(c.full_layer_indices) == 16
    # 3:1 schedule, full attention on every 4th layer (0-indexed: 3, 7, 11, ...).
    assert c.full_layer_indices[:4] == [3, 7, 11, 15]
    assert c.layer_types[0] == LINEAR_ATTENTION
    assert c.layer_types[3] == FULL_ATTENTION

    assert c.hidden_size == 5120
    assert c.intermediate_size == 17408
    assert c.vocab_size == 248320
    assert c.head_dim == 256


def test_derived_gdn_geometry():
    c = Qwen3_5TextConfig()
    assert c.key_dim == 16 * 128 == 2048
    assert c.value_dim == 48 * 128 == 6144
    # conv operates on [q | k | v] stacked
    assert c.conv_dim == 2 * 2048 + 6144 == 10240
    assert c.num_v_per_k == 3


def test_partial_rope_and_mrope_band():
    c = Qwen3_5TextConfig()
    # Only the leading 64 of 256 channels rotate.
    assert c.rotary_dim == 64
    # Interleaved mRoPE splits the half-rotary band three ways.
    assert sum(c.mrope_section) == c.rotary_dim // 2 == 32
    assert c.mrope_interleaved is True
    assert c.rope_theta == 10_000_000.0


def test_derived_mrope_section_reproduces_the_checkpoint_value():
    """The balanced split must agree with HuggingFace's shipped [11, 11, 10]."""
    assert Qwen3_5TextConfig().mrope_section == [11, 11, 10]


def test_derived_mrope_section_tracks_a_smaller_head_dim():
    """A tiny fixture must not inherit the 27B band and fail validation."""
    c = Qwen3_5TextConfig(head_dim=64)
    assert c.rotary_dim == 16
    assert sum(c.mrope_section) == 8
    assert c.mrope_section == [3, 3, 2]


def test_explicit_mrope_section_wins_over_derivation():
    params = dict(Qwen3_5TextConfig().rope_parameters, mrope_section=[12, 10, 10])
    assert Qwen3_5TextConfig(rope_parameters=params).mrope_section == [12, 10, 10]


def test_uses_mrope_mirrors_vllm_and_keys_off_the_raw_config():
    """vLLM checks the raw dict, so a derived section must not claim mRoPE."""
    # Shipped checkpoints state it explicitly -> mandatory SupportsMRoPE.
    params = dict(Qwen3_5TextConfig().rope_parameters, mrope_section=[11, 11, 10])
    assert Qwen3_5TextConfig(rope_parameters=params).uses_mrope is True
    # A fixture that omits it -> vLLM says no, and so must we.
    assert Qwen3_5TextConfig().uses_mrope is False


def test_head_dim_256_forces_single_shot_prefill():
    """head_dim 256 exceeds the segmented-attention kernel's SBUF partition bound."""
    assert Qwen3_5TextConfig().needs_single_shot_prefill is True
    assert Qwen3_5TextConfig(head_dim=128).needs_single_shot_prefill is False


# ---------------------------------------------------------------------------
# Layer-type normalization: derive, but never guess
# ---------------------------------------------------------------------------


def test_layer_types_derived_from_interval_matches_explicit():
    explicit = Qwen3_5TextConfig()
    derived = Qwen3_5TextConfig(layer_types=None, full_attention_interval=4)
    assert derived.layer_types == explicit.layer_types


def test_explicit_layer_types_are_honoured():
    types = [LINEAR_ATTENTION, FULL_ATTENTION] * 2
    c = Qwen3_5TextConfig(num_hidden_layers=4, layer_types=list(types))
    assert c.layer_types == types
    assert c.full_layer_indices == [1, 3]


def test_rejects_layer_types_length_mismatch():
    with pytest.raises(ValueError, match="entries but num_hidden_layers"):
        Qwen3_5TextConfig(num_hidden_layers=4, layer_types=[LINEAR_ATTENTION] * 3)


def test_rejects_unknown_layer_type():
    with pytest.raises(ValueError, match="Unsupported layer_types"):
        Qwen3_5TextConfig(
            num_hidden_layers=2, layer_types=[LINEAR_ATTENTION, "mamba2"]
        )


def test_rejects_all_linear_stack():
    with pytest.raises(ValueError, match="no full_attention layer"):
        Qwen3_5TextConfig(num_hidden_layers=2, layer_types=[LINEAR_ATTENTION] * 2)


def test_rejects_dense_stack_pointing_at_qwen3():
    with pytest.raises(ValueError, match="no linear_attention layer"):
        Qwen3_5TextConfig(num_hidden_layers=2, layer_types=[FULL_ATTENTION] * 2)


def test_rejects_mrope_section_that_does_not_span_the_band():
    bad = {
        "rope_type": "default",
        "rope_theta": 1e7,
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 11],  # sums to 33, not 32
        "partial_rotary_factor": 0.25,
    }
    with pytest.raises(ValueError, match="sums to 33"):
        Qwen3_5TextConfig(rope_parameters=bad)


def test_rejects_mrope_section_with_wrong_axis_count():
    bad = dict(Qwen3_5TextConfig().rope_parameters, mrope_section=[16, 16])
    with pytest.raises(ValueError, match="exactly 3 entries"):
        Qwen3_5TextConfig(rope_parameters=bad)


def test_rejects_value_heads_not_multiple_of_key_heads():
    with pytest.raises(ValueError, match="must be a multiple of"):
        Qwen3_5TextConfig(linear_num_key_heads=16, linear_num_value_heads=40)


# ---------------------------------------------------------------------------
# Nested config extraction
# ---------------------------------------------------------------------------


def test_from_configs_unwraps_text_config():
    """The shipped checkpoint nests the decoder under text_config."""
    raw = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 128,
            "dtype": "bfloat16",
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 6,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
        },
        "vision_config": {"depth": 27},
    }
    c = Qwen3_5TextConfig.from_configs(raw)

    assert c.hidden_size == 256
    assert c.num_hidden_layers == 4
    # "dtype" in json maps onto the dataclass's torch_dtype
    import torch

    assert c.torch_dtype is torch.bfloat16
    # vision_config is ignored entirely
    assert not hasattr(c, "depth")


def test_from_configs_accepts_bare_text_config():
    c = Qwen3_5TextConfig.from_configs({"hidden_size": 5120})
    assert c.hidden_size == 5120


# ---------------------------------------------------------------------------
# TP sharding policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tp,q_per_rank,pad,kv_per_rank,replicas,k_per_rank,v_per_rank,v_dim,splits",
    [
        (1, 24, 0, 4, 1, 16, 48, 128, 1),
        (2, 12, 0, 2, 1, 8, 24, 128, 1),
        (4, 6, 0, 1, 1, 4, 12, 128, 1),
        (8, 3, 0, 1, 2, 2, 6, 128, 1),
        # 24 Q heads do not divide 16 -> pad to 32. GDN is exact.
        (16, 2, 8, 1, 4, 1, 3, 128, 1),
        # Beyond 16 key heads, split the value dimension instead.
        (32, 1, 8, 1, 8, 1, 3, 64, 2),
    ],
)
def test_sharding_policy(
    tp, q_per_rank, pad, kv_per_rank, replicas, k_per_rank, v_per_rank, v_dim, splits
):
    p = resolve_sharding(Qwen3_5TextConfig(), tp)

    assert p.q_heads_per_rank == q_per_rank
    assert p.q_head_padding == pad
    assert p.kv_heads_per_rank == kv_per_rank
    assert p.num_kv_replicas == replicas
    assert p.k_heads_per_rank == k_per_rank
    assert p.v_heads_per_rank == v_per_rank
    assert p.v_dim_per_rank == v_dim
    assert p.v_dim_shards == splits
    assert p.intermediate_per_rank == 17408 // tp


def test_padded_q_heads_cover_every_rank():
    """Padding must reach a whole multiple of tp, never truncate a real head."""
    c = Qwen3_5TextConfig()
    for tp in (1, 2, 4, 8, 16, 32):
        p = resolve_sharding(c, tp)
        assert p.padded_q_heads % tp == 0
        assert p.padded_q_heads >= c.num_attention_heads
        assert p.padded_q_heads - c.num_attention_heads < tp


def test_gated_norm_allreduce_only_when_value_dim_split():
    c = Qwen3_5TextConfig()
    assert resolve_sharding(c, 16).gated_norm_needs_allreduce is False
    assert resolve_sharding(c, 32).gated_norm_needs_allreduce is True


def test_rejects_tp_that_straddles_key_heads():
    """tp must divide the key heads or be a multiple of them, never straddle.

    The shipped 27B cannot exercise this: hidden_size and intermediate_size
    share a gcd of 1024, so every admissible tp is a power of two, and every
    power of two either divides 16 or is a multiple of it. A config with a
    non-power-of-two key-head count is needed to reach the branch.
    """
    c = Qwen3_5TextConfig(linear_num_key_heads=3, linear_num_value_heads=6)
    with pytest.raises(ValueError, match="not supported for Gated DeltaNet"):
        resolve_sharding(c, 2)


def test_real_27b_admits_every_power_of_two_up_to_32():
    """No supported degree for the shipped config should be rejected."""
    c = Qwen3_5TextConfig()
    for tp in (1, 2, 4, 8, 16, 32):
        resolve_sharding(c, tp)  # must not raise


def test_rejects_tp_that_splits_kv_heads_unevenly():
    c = Qwen3_5TextConfig(num_key_value_heads=3)
    with pytest.raises(ValueError, match="must be a multiple of"):
        resolve_sharding(c, 8)


# ---------------------------------------------------------------------------
# Full-attention head partition
#
# The rank -> KV head mapping used to be `rank // num_kv_replicas`, computed
# inline in the weight loader. That is right only while the query heads are
# unpadded, so it is right for the 0.8B at every degree and wrong for the
# shipped 27B as soon as padding appears. Nothing asserted which global heads a
# rank owns, which is why it survived.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16, 32])
def test_every_rank_query_heads_share_its_kv_head_group(tp):
    """The invariant that matters: a rank's query heads and its KV head agree.

    Checks the shipped 27B geometry (24 Q / 4 KV, GQA group 6). At tp=16 the 24
    heads pad to 32, and the old `rank // num_kv_replicas` gives rank 3 KV head
    0 while it owns query heads 6-7, which belong to KV head 1.
    """
    c = Qwen3_5TextConfig()
    p = resolve_sharding(c, tp)
    if p.kv_heads_per_rank != 1:
        pytest.skip("KV heads are split, not replicated, at this degree")

    for rank in range(tp):
        real = [h for h in p.q_head_indices(rank) if h < p.num_q_heads]
        if not real:
            # Padding-only rank: its output is annihilated by the zeroed
            # o_proj columns, but the index must still be in range.
            assert 0 <= p.kv_head_index(rank) < p.num_kv_heads
            continue
        wanted = {h // p.gqa_group_size for h in real}
        assert len(wanted) == 1, (
            f"tp={tp} rank={rank} owns heads {real} spanning KV groups {wanted}"
        )
        assert p.kv_head_index(rank) == wanted.pop(), (
            f"tp={tp} rank={rank} heads={real} got KV head "
            f"{p.kv_head_index(rank)}"
        )


def test_kv_head_index_is_not_the_rank_over_replicas_shortcut():
    """Pin the specific 27B ranks the old expression got wrong.

    Without this, a regression back to `rank // num_kv_replicas` passes every
    other test in the suite.
    """
    p = resolve_sharding(Qwen3_5TextConfig(), 16)
    assert p.padded_q_heads == 32 and p.q_heads_per_rank == 2
    shortcut = [rank // p.num_kv_replicas for rank in range(16)]
    correct = [p.kv_head_index(rank) for rank in range(16)]
    assert correct != shortcut
    # Rank 3 owns query heads 6-7; 6 // 6 == KV head 1, not 0.
    assert p.q_head_indices(3) == [6, 7]
    assert p.kv_head_index(3) == 1
    assert shortcut[3] == 0


def test_rejects_a_degree_whose_ranks_straddle_kv_groups():
    """6 Q / 2 KV at tp=4 gives 2 heads per rank against a group of 3.

    Rank 1 would own heads 2 and 3 -- one from each group -- and a single KV
    head cannot serve both. Reject rather than shard silently wrong.
    """
    c = Qwen3_5TextConfig(num_attention_heads=6, num_key_value_heads=2)
    with pytest.raises(ValueError, match="span two GQA groups"):
        resolve_sharding(c, 4)


# ---------------------------------------------------------------------------
# Gated DeltaNet row partition
#
# Every GDN weight loader and the state-cache spec derive their widths from
# these helpers, so a wrong partition here is wrong everywhere at once.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16, 32])
def test_conv_rows_partition_the_value_block_exactly_once(tp):
    """Value rows are split; no row may be dropped or handed to two ranks."""
    c = Qwen3_5TextConfig()
    p = resolve_sharding(c, tp)

    covered = collections.Counter()
    for rank in range(tp):
        for lo, hi in p.value_row_ranges(rank):
            covered.update(range(lo, hi))

    assert set(covered) == set(range(p.value_dim))
    assert set(covered.values()) == {1}


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16, 32])
def test_query_and_key_rows_are_replicated_across_value_dim_shards(tp):
    """q and k are *not* split when the value dim is.

    A rank sharing a key head with another needs the whole query and key for
    that head -- only the value columns of its state are its own. This is what
    makes conv_dim_per_rank exceed conv_dim // tp at tp=32.
    """
    c = Qwen3_5TextConfig()
    p = resolve_sharding(c, tp)

    covered = collections.Counter()
    for rank in range(tp):
        for lo, hi in p.key_row_ranges(rank):
            covered.update(range(lo, hi))

    assert set(covered) == set(range(p.key_dim))
    assert set(covered.values()) == {p.v_dim_shards}


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16, 32])
def test_conv_row_ranges_are_the_three_blocks_in_order(tp):
    """conv_row_ranges must be q, then k, then v -- the checkpoint's order."""
    c = Qwen3_5TextConfig()
    p = resolve_sharding(c, tp)

    for rank in range(tp):
        ranges = p.conv_row_ranges(rank)
        width = sum(hi - lo for lo, hi in ranges)
        assert width == p.conv_dim_per_rank

        n_k = len(p.key_row_ranges(rank))
        q_block = ranges[:n_k]
        k_block = ranges[n_k : 2 * n_k]
        v_block = ranges[2 * n_k :]
        assert all(hi <= p.key_dim for _, hi in q_block)
        assert all(p.key_dim < hi <= 2 * p.key_dim for _, hi in k_block)
        assert all(hi > 2 * p.key_dim for _, hi in v_block)
        # k is the same rows as q, shifted by one key_dim block.
        assert [(lo - p.key_dim, hi - p.key_dim) for lo, hi in k_block] == q_block


def test_conv_dim_per_rank_is_not_the_naive_quotient_at_tp32():
    """Pin the trap: the naive form is right at every degree but the target one.

    ``conv_dim // tp`` agrees up to tp=16 and undersizes the conv state cache by
    28% at tp=32, which would corrupt the convolution window rather than fail.
    """
    c = Qwen3_5TextConfig()
    for tp in (1, 2, 4, 8, 16):
        assert resolve_sharding(c, tp).conv_dim_per_rank == c.conv_dim // tp

    p32 = resolve_sharding(c, 32)
    assert p32.conv_dim_per_rank == 2 * 128 + 3 * 64 == 448
    assert c.conv_dim // 32 == 320


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16, 32])
def test_value_heads_follow_their_key_head(tp):
    """Value head j belongs to key head j // num_v_per_k, on every rank."""
    c = Qwen3_5TextConfig()
    p = resolve_sharding(c, tp)

    for rank in range(tp):
        k_heads = p.k_head_indices(rank)
        v_heads = p.v_head_indices(rank)
        assert len(v_heads) == p.v_heads_per_rank
        assert {v // c.num_v_per_k for v in v_heads} == set(k_heads)


def test_partner_ranks_at_tp32_share_heads_but_split_the_dimension():
    """The two ranks on one key head must cover its value dim between them."""
    p = resolve_sharding(Qwen3_5TextConfig(), 32)

    for pair in range(16):
        a, b = 2 * pair, 2 * pair + 1
        assert p.k_head_indices(a) == p.k_head_indices(b) == [pair]
        assert p.v_head_indices(a) == p.v_head_indices(b)
        assert p.v_dim_offset(a) == 0
        assert p.v_dim_offset(b) == 64
        assert p.value_row_ranges(a) != p.value_row_ranges(b)
