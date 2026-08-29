# SPDX-License-Identifier: Apache-2.0
"""Config normalization and TP sharding policy for the Qwen3.5 family.

These are pure-CPU tests: no torch device, no NKI, no vLLM engine. They pin the
geometry that every later phase depends on, and they pin the *rejections* --
a config that quietly defaults is a config that quietly narrows the gate.
"""

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
