# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_neuron.model.deepseek_v4.weight_loaders import (
    StackedShard,
    map_checkpoint_name,
    require_weight_shape,
    resolve_stacked_shard,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("embed.weight", "model.embed_tokens.weight"),
        ("head.weight", "lm_head.weight"),
        ("norm.weight", "model.norm.weight"),
        ("hc_head_fn", "model.hc_head_fn"),
        ("layers.3.attn.wq_a.weight", "model.layers.3.attn.wq_a.weight"),
        (
            "layers.3.ffn.gate.bias",
            "model.layers.3.ffn.gate.e_score_correction_bias",
        ),
        (
            "layers.3.ffn.shared_experts.w2.weight",
            "model.layers.3.ffn.shared_experts.down_proj.weight",
        ),
        ("mtp.layers.0.norm.weight", "model.mtp.layers.0.norm.weight"),
    ],
)
def test_checkpoint_names_match_vllm_026_mapper(source, expected):
    assert map_checkpoint_name(source) == expected


def test_quantized_scale_names_distinguish_fp4_experts():
    source = "layers.2.ffn.experts.7.w1.scale"
    assert map_checkpoint_name(source, "fp4").endswith("w1.weight_scale")
    assert map_checkpoint_name(source, "fp8").endswith("w1.weight_scale_inv")
    shared = "layers.2.ffn.shared_experts.w2.scale"
    assert map_checkpoint_name(shared, "fp4").endswith(
        "shared_experts.down_proj.weight_scale_inv"
    )


@pytest.mark.parametrize(
    ("source", "target", "shard_id"),
    [
        ("layers.1.ffn.w1.weight", "model.layers.1.ffn.gate_up_proj.weight", 0),
        ("layers.1.ffn.w3.weight", "model.layers.1.ffn.gate_up_proj.weight", 1),
        (
            "layers.1.attn.wq_a.weight",
            "model.layers.1.attn.fused_wqa_wkv.weight",
            0,
        ),
        (
            "layers.1.attn.wkv.weight",
            "model.layers.1.attn.fused_wqa_wkv.weight",
            1,
        ),
        (
            "layers.1.attn.compressor.wgate.weight",
            "model.layers.1.attn.compressor.fused_wkv_wgate.weight",
            1,
        ),
    ],
)
def test_stacked_parameter_contract(source, target, shard_id):
    assert resolve_stacked_shard(map_checkpoint_name(source)) == StackedShard(
        target, shard_id
    )


def test_expert_weights_are_not_consumed_by_generic_stacking():
    mapped = map_checkpoint_name("layers.1.ffn.experts.2.w1.weight")
    assert resolve_stacked_shard(mapped) is None


def test_shape_drift_fails_before_copy():
    require_weight_shape("head.weight", (32, 16), (32, 16))
    with pytest.raises(ValueError, match=r"head\.weight.*\(31, 16\).+\(32, 16\)"):
        require_weight_shape("head.weight", (31, 16), (32, 16))
