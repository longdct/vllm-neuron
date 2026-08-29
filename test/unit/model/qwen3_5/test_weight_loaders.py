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
