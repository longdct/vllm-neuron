# SPDX-License-Identifier: Apache-2.0
"""Checkpoint -> parameter mapping for Qwen3.5.

Covers the two traps that fail silently: the RMSNorm ``+1`` fold (and the one
norm that must *not* be folded), and the per-head query/gate interleave that a
flat column split would scramble.
"""

import pytest
import torch

from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig
from vllm_neuron.model.qwen3_5.parallel import resolve_sharding
from vllm_neuron.model.qwen3_5.weight_loaders import (
    TEXT_PREFIX,
    VISION_PREFIX,
    gated_o_proj_weight_loader,
    gated_qkv_weight_loader,
    gdn_conv1d_weight_loader,
    gdn_gated_norm_loader,
    gdn_head_vector_loader,
    gdn_out_proj_weight_loader,
    gdn_qkv_weight_loader,
    gdn_row_weight_loader,
    needs_plus_one_fold,
    norm_plus_one_loader,
    plain_loader,
    text_weight_mappings,
)


class FakeSlice:
    """Minimal stand-in for a safetensors PySafeSlice."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def __getitem__(self, key):
        return self._tensor[key]

    @property
    def shape(self):
        return self._tensor.shape


def _config(**overrides):
    base = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=6,
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


# ---------------------------------------------------------------------------
# The +1 fold
# ---------------------------------------------------------------------------


def test_norm_loader_folds_the_plus_one():
    raw = torch.tensor([0.0, 0.5, -0.25], dtype=torch.float32)
    loaded = norm_plus_one_loader().load([FakeSlice(raw)], rank=0)
    torch.testing.assert_close(loaded, torch.tensor([1.0, 1.5, 0.75]))


def test_zero_checkpoint_norm_becomes_the_identity():
    """HF initializes these to zeros, which must mean 'scale by 1'."""
    raw = torch.zeros(8, dtype=torch.float32)
    loaded = norm_plus_one_loader().load([FakeSlice(raw)], rank=0)
    torch.testing.assert_close(loaded, torch.ones(8))


def test_plain_loader_does_not_fold():
    raw = torch.tensor([1.0, 0.5], dtype=torch.float32)
    torch.testing.assert_close(
        plain_loader().load([FakeSlice(raw)], rank=0), raw
    )


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.input_layernorm.weight",
        "model.layers.3.post_attention_layernorm.weight",
        "model.layers.3.self_attn.q_norm.weight",
        "model.layers.3.self_attn.k_norm.weight",
        "model.norm.weight",
    ],
)
def test_hf_rmsnorm_weights_are_folded(name):
    assert needs_plus_one_fold(name)


def test_gated_norm_is_not_folded():
    """linear_attn.norm is HF's *gated* RMSNorm: ordinary convention, ones-init.

    Folding it would be exactly as wrong as failing to fold the others.
    """
    assert not needs_plus_one_fold("model.layers.0.linear_attn.norm.weight")


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.gate_proj_weight",
        "model.layers.0.self_attn.qkv_proj_weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
    ],
)
def test_non_norm_weights_are_not_folded(name):
    assert not needs_plus_one_fold(name)


# ---------------------------------------------------------------------------
# Gated QKV fusion and sharding
# ---------------------------------------------------------------------------


def _qkv_slices(config):
    """Checkpoint tensors, [out, hidden], with recognisable values per head."""
    head_dim = config.head_dim
    hidden = config.hidden_size

    # Row block h*2*head_dim .. +head_dim is head h's query; the next is its gate.
    q = torch.zeros(config.num_attention_heads * 2 * head_dim, hidden)
    for h in range(config.num_attention_heads):
        q[h * 2 * head_dim : h * 2 * head_dim + head_dim, :] = float(h + 1)
        q[h * 2 * head_dim + head_dim : (h + 1) * 2 * head_dim, :] = -float(h + 1)

    k = torch.zeros(config.num_key_value_heads * head_dim, hidden)
    v = torch.zeros(config.num_key_value_heads * head_dim, hidden)
    for h in range(config.num_key_value_heads):
        k[h * head_dim : (h + 1) * head_dim, :] = 10.0 + h
        v[h * head_dim : (h + 1) * head_dim, :] = 20.0 + h

    return [FakeSlice(q), FakeSlice(k), FakeSlice(v)]


def test_qkv_loader_keeps_the_per_head_query_gate_interleave():
    config = _config()
    policy = resolve_sharding(config, 1)
    fused = gated_qkv_weight_loader(config, policy).load(_qkv_slices(config), rank=0)

    head_dim = config.head_dim
    n_heads = config.num_attention_heads
    # Parameter is [hidden, fused]; transpose back for inspection.
    fused_t = fused.t()

    q_block = fused_t[: n_heads * 2 * head_dim]
    per_head = q_block.view(n_heads, 2 * head_dim, -1)
    for h in range(n_heads):
        assert torch.all(per_head[h, :head_dim] == float(h + 1)), h
        assert torch.all(per_head[h, head_dim:] == -float(h + 1)), h


def test_qkv_loader_round_trips_through_split_query_and_gate():
    """The fused layout must be exactly what the runtime split expects."""
    from vllm_neuron.model.qwen3_5.attention import split_query_and_gate

    config = _config()
    policy = resolve_sharding(config, 1)
    fused = gated_qkv_weight_loader(config, policy).load(_qkv_slices(config), rank=0)

    hidden = config.hidden_size
    head_dim = config.head_dim
    n_heads = config.num_attention_heads
    q_gate_size = n_heads * 2 * head_dim

    x = torch.eye(hidden)
    projected = x @ fused  # [hidden, fused]
    qg = projected[..., :q_gate_size]

    query, gate = split_query_and_gate(qg, n_heads, head_dim)
    for h in range(n_heads):
        assert torch.all(query[:, h] == float(h + 1)), h
        assert torch.all(gate[:, h] == -float(h + 1)), h


def test_qkv_loader_shards_heads_across_ranks_without_overlap():
    config = _config()
    policy = resolve_sharding(config, 2)
    head_dim, n_heads = config.head_dim, config.num_attention_heads

    seen = []
    for rank in range(2):
        fused_t = (
            gated_qkv_weight_loader(config, policy)
            .load(_qkv_slices(config), rank=rank)
            .t()
        )
        q_block = fused_t[: policy.q_heads_per_rank * 2 * head_dim]
        per_head = q_block.view(policy.q_heads_per_rank, 2 * head_dim, -1)
        seen.extend(int(per_head[h, 0, 0].item()) for h in range(policy.q_heads_per_rank))

    assert sorted(seen) == list(range(1, n_heads + 1))


def test_padded_query_heads_are_zero():
    """6 heads over tp=4 pads to 8; the two extra heads must be zeros."""
    config = _config()
    policy = resolve_sharding(config, 4)
    assert policy.padded_q_heads == 8
    assert policy.q_head_padding == 2

    head_dim = config.head_dim
    last_rank = 3
    fused_t = (
        gated_qkv_weight_loader(config, policy)
        .load(_qkv_slices(config), rank=last_rank)
        .t()
    )
    q_block = fused_t[: policy.q_heads_per_rank * 2 * head_dim]
    # Rank 3 owns heads 6 and 7, both beyond the real 6.
    assert torch.all(q_block == 0)


def test_kv_heads_replicate_when_ranks_outnumber_them():
    config = _config()
    policy = resolve_sharding(config, 4)
    assert policy.num_kv_replicas == 2

    head_dim = config.head_dim
    values = []
    for rank in range(4):
        fused_t = (
            gated_qkv_weight_loader(config, policy)
            .load(_qkv_slices(config), rank=rank)
            .t()
        )
        k_start = policy.q_heads_per_rank * 2 * head_dim
        values.append(float(fused_t[k_start, 0].item()))

    # Ranks 0,1 share KV head 0; ranks 2,3 share KV head 1.
    assert values == [10.0, 10.0, 11.0, 11.0]


def test_o_proj_zeroes_the_padded_head_columns():
    config = _config()
    policy = resolve_sharding(config, 4)
    hidden, head_dim = config.hidden_size, config.head_dim

    o = torch.arange(
        hidden * config.num_attention_heads * head_dim, dtype=torch.float32
    ).reshape(hidden, config.num_attention_heads * head_dim)

    loaded = gated_o_proj_weight_loader(config, policy).load([FakeSlice(o)], rank=3)
    # Parameter is [heads_per_rank * head_dim, hidden]; rank 3 is all padding.
    assert torch.all(loaded == 0)


def test_o_proj_preserves_real_head_columns():
    config = _config()
    policy = resolve_sharding(config, 1)
    hidden, head_dim = config.hidden_size, config.head_dim
    o = torch.randn(hidden, config.num_attention_heads * head_dim)

    loaded = gated_o_proj_weight_loader(config, policy).load([FakeSlice(o)], rank=0)
    torch.testing.assert_close(loaded, o.t())


# ---------------------------------------------------------------------------
# Key mapping
# ---------------------------------------------------------------------------


def test_mappings_target_the_text_decoder_subtree():
    config = _config()
    mappings = text_weight_mappings(config)
    assert mappings["model.embed_tokens.weight"] == f"{TEXT_PREFIX}.embed_tokens.weight"
    assert mappings["lm_head.weight"] == "lm_head.weight"


def test_mappings_never_reference_the_vision_tower():
    config = _config()
    flat = []
    for value in text_weight_mappings(config).values():
        flat.extend(value if isinstance(value, list) else [value])
    assert not any(key.startswith(VISION_PREFIX) for key in flat)


def test_mappings_follow_the_layer_schedule():
    config = _config()
    mappings = text_weight_mappings(config)

    for i, kind in enumerate(config.layer_types):
        if kind == "full_attention":
            assert f"model.layers.{i}.self_attn.qkv_proj_weight" in mappings
            assert f"model.layers.{i}.linear_attn.conv1d.weight" not in mappings
        else:
            assert f"model.layers.{i}.linear_attn.in_proj_qkv.weight" in mappings
            assert f"model.layers.{i}.self_attn.qkv_proj_weight" not in mappings


def test_gdn_layers_map_four_separate_input_projections():
    """Qwen3.5 keeps them separate; Qwen3-Next fuses into qkvz/ba pairs."""
    config = _config()
    mappings = text_weight_mappings(config)
    gdn_index = config.linear_layer_indices[0]
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        key = f"model.layers.{gdn_index}.linear_attn.{name}.weight"
        assert mappings[key] == f"{TEXT_PREFIX}.layers.{gdn_index}.linear_attn.{name}.weight"


def test_every_full_attention_layer_maps_three_qkv_sources():
    config = _config()
    mappings = text_weight_mappings(config)
    for i in config.full_layer_indices:
        sources = mappings[f"model.layers.{i}.self_attn.qkv_proj_weight"]
        assert len(sources) == 3
        assert sources[0].endswith("q_proj.weight")


# ---------------------------------------------------------------------------
# Gated DeltaNet sharding
#
# The tiny config below has 2 key heads and 6 value heads, so tp=2 is pure head
# sharding, tp=4 splits each value head in two and tp=8 splits it four ways --
# structurally the same three regimes the shipped 27B hits at tp<=16, 32 and
# (hypothetically) 64, at a size a test can hold.
# ---------------------------------------------------------------------------

GDN_TP_DEGREES = [1, 2, 4, 8]


def _row_ids(rows: int, cols: int) -> torch.Tensor:
    """``[rows, cols]`` where every element of row ``i`` is ``i``.

    Loading it back tells you exactly which global rows a rank was handed,
    which is the only thing these loaders are responsible for.
    """
    ids = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    return ids.expand(rows, cols).contiguous()


def _loaded_row_ids(tensor: torch.Tensor, dim: int = 0) -> list[int]:
    ids = tensor if dim == 0 else tensor.t()
    return [int(v) for v in ids[:, 0].tolist()]


@pytest.mark.parametrize("tp", GDN_TP_DEGREES)
def test_gdn_qkv_loader_replicates_qk_and_splits_v(tp):
    config = _config()
    policy = resolve_sharding(config, tp)
    ckpt = FakeSlice(_row_ids(config.conv_dim, config.hidden_size))
    loader = gdn_qkv_weight_loader(policy)

    kd = config.key_dim
    d_k = config.linear_key_head_dim
    d_v = config.linear_value_head_dim
    width = policy.v_dim_per_rank

    v_rows_seen = []
    for rank in range(tp):
        shard = loader.load([ckpt], rank=rank)
        assert shard.shape == (policy.conv_dim_per_rank, config.hidden_size)
        got = _loaded_row_ids(shard)

        # Derived here from head arithmetic alone, independent of the policy.
        if policy.v_dim_shards > 1:
            k_heads = [rank // policy.v_dim_shards]
            offset = (rank % policy.v_dim_shards) * width
        else:
            k_heads = list(
                range(
                    rank * policy.k_heads_per_rank,
                    (rank + 1) * policy.k_heads_per_rank,
                )
            )
            offset = 0

        expect_q = [h * d_k + i for h in k_heads for i in range(d_k)]
        expect_k = [kd + r for r in expect_q]
        expect_v = [
            2 * kd + v * d_v + offset + i
            for h in k_heads
            for v in range(h * config.num_v_per_k, (h + 1) * config.num_v_per_k)
            for i in range(width)
        ]
        assert got == expect_q + expect_k + expect_v, rank
        v_rows_seen.extend(expect_v)

    # Value rows partition the block; q/k rows are replicated v_dim_shards ways.
    assert sorted(v_rows_seen) == list(range(2 * kd, config.conv_dim))


def test_gdn_qkv_partner_ranks_get_identical_query_and_key():
    """The pair sharing a key head must both hold its whole q and k."""
    config = _config()
    policy = resolve_sharding(config, 4)  # v_dim_shards == 2
    ckpt = FakeSlice(_row_ids(config.conv_dim, config.hidden_size))
    loader = gdn_qkv_weight_loader(policy)

    qk = 2 * policy.key_dim_per_rank
    for pair in range(2):
        a = loader.load([ckpt], rank=2 * pair)
        b = loader.load([ckpt], rank=2 * pair + 1)
        assert torch.equal(a[:qk], b[:qk]), pair
        assert not torch.equal(a[qk:], b[qk:]), pair


@pytest.mark.parametrize("tp", GDN_TP_DEGREES)
def test_gdn_conv1d_loader_matches_the_qkv_channel_partition(tp):
    """Depthwise: one filter per channel, so the two partitions must agree."""
    config = _config()
    policy = resolve_sharding(config, tp)
    kernel = config.linear_conv_kernel_dim

    conv = _row_ids(config.conv_dim, kernel).unsqueeze(1)  # [conv_dim, 1, K]
    qkv = _row_ids(config.conv_dim, config.hidden_size)

    for rank in range(tp):
        conv_shard = gdn_conv1d_weight_loader(policy).load([FakeSlice(conv)], rank=rank)
        qkv_shard = gdn_qkv_weight_loader(policy).load([FakeSlice(qkv)], rank=rank)
        assert conv_shard.shape == (policy.conv_dim_per_rank, 1, kernel)
        assert _loaded_row_ids(conv_shard[:, 0]) == _loaded_row_ids(qkv_shard), rank


@pytest.mark.parametrize("tp", GDN_TP_DEGREES)
def test_gdn_z_and_out_proj_cover_the_value_dim_exactly_once(tp):
    """in_proj_z rows and out_proj columns are the same partition, transposed."""
    config = _config()
    policy = resolve_sharding(config, tp)

    z = FakeSlice(_row_ids(config.value_dim, config.hidden_size))
    o = FakeSlice(_row_ids(config.value_dim, config.hidden_size).t().contiguous())

    seen = []
    for rank in range(tp):
        z_shard = gdn_row_weight_loader(policy).load([z], rank=rank)
        o_shard = gdn_out_proj_weight_loader(policy).load([o], rank=rank)
        assert z_shard.shape == (policy.value_dim_per_rank, config.hidden_size)
        assert o_shard.shape == (config.hidden_size, policy.value_dim_per_rank)
        rows = _loaded_row_ids(z_shard)
        assert rows == _loaded_row_ids(o_shard, dim=1), rank
        seen.extend(rows)

    assert sorted(seen) == list(range(config.value_dim))


@pytest.mark.parametrize("tp", GDN_TP_DEGREES)
def test_gdn_head_vector_loader_handles_both_ranks_of_tensor(tp):
    """dt_bias/A_log are 1-D; in_proj_b/a are 2-D. Same head partition."""
    config = _config()
    policy = resolve_sharding(config, tp)

    matrix = FakeSlice(_row_ids(config.linear_num_value_heads, config.hidden_size))
    vector = FakeSlice(torch.arange(config.linear_num_value_heads, dtype=torch.float32))

    seen = []
    for rank in range(tp):
        m = gdn_head_vector_loader(policy).load([matrix], rank=rank)
        v = gdn_head_vector_loader(policy).load([vector], rank=rank)
        assert m.shape == (policy.v_heads_per_rank, config.hidden_size)
        assert v.shape == (policy.v_heads_per_rank,)
        assert _loaded_row_ids(m) == [int(x) for x in v.tolist()], rank
        seen.extend(int(x) for x in v.tolist())

    # Value heads are replicated, not split, when only the value *dim* splits.
    expected = list(range(config.linear_num_value_heads)) * policy.v_dim_shards
    assert sorted(seen) == sorted(expected)


@pytest.mark.parametrize("tp", GDN_TP_DEGREES)
def test_gdn_gated_norm_loader_slices_to_this_rank_width(tp):
    """The norm weight spans one value head's dim, which tp may have split."""
    config = _config()
    policy = resolve_sharding(config, tp)
    weight = FakeSlice(torch.arange(config.linear_value_head_dim, dtype=torch.float32))

    for rank in range(tp):
        shard = gdn_gated_norm_loader(policy).load([weight], rank=rank)
        assert shard.shape == (policy.v_dim_per_rank,)
        lo = (rank % policy.v_dim_shards) * policy.v_dim_per_rank
        assert shard.tolist() == list(range(lo, lo + policy.v_dim_per_rank)), rank

    # No +1 fold here: this is HF's gated norm, already in the ordinary form.
    assert needs_plus_one_fold("model.layers.0.linear_attn.norm.weight") is False
