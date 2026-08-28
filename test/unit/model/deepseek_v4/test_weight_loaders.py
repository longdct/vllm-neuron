# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.weight_loaders import (
    StackedShard,
    load_checkpoint_weights,
    is_unsupported_checkpoint_name,
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
        # mHC parameters live on real submodules, so the checkpoint's flattened
        # ``hc_<site>_<param>`` spelling becomes a dotted path. ``fn``/``base``
        # drop the ``hc_`` prefix while ``hc_scale`` keeps it, matching the
        # module definitions in model.py.
        ("hc_head_fn", "model.hc_head.fn"),
        ("hc_head_base", "model.hc_head.base"),
        ("hc_head_scale", "model.hc_head.hc_scale"),
        ("layers.3.hc_attn_fn", "model.layers.3.attn_hc.fn"),
        ("layers.3.hc_attn_scale", "model.layers.3.attn_hc.hc_scale"),
        ("layers.3.hc_ffn_base", "model.layers.3.ffn_hc.base"),
        # The decoder layer holds ``attention``/``moe`` submodules and
        # ``input_layernorm``/``post_attention_layernorm``.
        ("layers.3.attn.wq_a.weight", "model.layers.3.attention.q_a_proj.weight"),
        ("layers.3.attn.wkv.weight", "model.layers.3.attention.kv_proj.weight"),
        ("layers.3.attn.wq_b.weight", "model.layers.3.attention.q_b_proj.weight"),
        ("layers.3.attn.q_norm.weight", "model.layers.3.attention.q_a_norm.weight"),
        ("layers.3.attn.kv_norm.weight", "model.layers.3.attention.kv_norm.weight"),
        ("layers.3.attn.wo_a.weight", "model.layers.3.attention.o_a_proj.weight"),
        ("layers.3.attn.wo_b.weight", "model.layers.3.attention.o_b_proj.weight"),
        ("layers.3.attn.attn_sink", "model.layers.3.attention.sinks"),
        ("layers.3.attn_norm.weight", "model.layers.3.input_layernorm.weight"),
        ("layers.3.ffn_norm.weight", "model.layers.3.post_attention_layernorm.weight"),
        (
            "layers.3.attn.compressor.norm.weight",
            "model.layers.3.attention.compressor.norm_weight",
        ),
        ("layers.3.attn.compressor.ape", "model.layers.3.attention.compressor.ape"),
        # Router state sits on the MoE block itself, not on the gate submodule.
        ("layers.3.ffn.gate.bias", "model.layers.3.moe.correction_bias"),
        ("layers.3.ffn.gate.weight", "model.layers.3.moe.gate.weight"),
        # Hash-routing table: a registered buffer, and one that real checkpoints
        # actually provide -- the config-derived fallback must not win.
        ("layers.3.ffn.gate.tid2eid", "model.layers.3.moe.tid2eid"),
        # Expert projections are bare parameters, so ``.weight`` is dropped.
        (
            "layers.3.ffn.shared_experts.w2.weight",
            "model.layers.3.moe.shared_experts.down_proj",
        ),
        ("mtp.layers.0.norm.weight", "model.mtp.layers.0.norm.weight"),
    ],
)
def test_checkpoint_names_map_onto_plugin_parameters(source, expected):
    assert map_checkpoint_name(source) == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "layers.2.attn.indexer.wq_b.weight",
            "model.layers.2.attention.indexer.q_b_proj.weight",
        ),
        (
            "layers.2.attn.indexer.weights_proj.weight",
            "model.layers.2.attention.indexer.weights_proj.weight",
        ),
        (
            "layers.2.attn.indexer.compressor.ape",
            "model.layers.2.attention.indexer.compressor.ape",
        ),
        (
            "layers.2.attn.indexer.compressor.norm.weight",
            "model.layers.2.attention.indexer.compressor.norm_weight",
        ),
    ],
)
def test_indexer_tensors_map_onto_plugin_parameters(source, expected):
    """Every name the real checkpoint ships for the indexer, verified against
    ``ds-v4-flash-shards/model.safetensors.index.json``. These were skipped
    outright until the indexer existed."""
    assert map_checkpoint_name(source) == expected


@pytest.mark.parametrize("leaf,shard", [("wkv", 0), ("wgate", 1)])
def test_indexer_compressor_projections_fuse_separately_from_the_outer_one(leaf, shard):
    """The indexer's compressor and the layer's own must not collide.

    Both are ``DeepseekV4Compressor``s with a ``fused_wkv_wgate``, distinguished
    only by the ``indexer.`` segment. ``resolve_stacked_shard`` matches on a
    bare substring, so this pins that the more specific name wins.
    """
    outer = resolve_stacked_shard(
        map_checkpoint_name(f"layers.2.attn.compressor.{leaf}.weight")
    )
    inner = resolve_stacked_shard(
        map_checkpoint_name(f"layers.2.attn.indexer.compressor.{leaf}.weight")
    )
    assert outer.shard_id == inner.shard_id == shard
    assert outer.parameter_name == (
        "model.layers.2.attention.compressor.fused_wkv_wgate.weight"
    )
    assert inner.parameter_name == (
        "model.layers.2.attention.indexer.compressor.fused_wkv_wgate.weight"
    )
    assert outer.parameter_name != inner.parameter_name


def test_no_subtree_is_skipped_now_that_the_indexer_is_modelled():
    assert not is_unsupported_checkpoint_name("layers.2.attn.indexer.wq_b.weight")
    assert not is_unsupported_checkpoint_name("layers.2.attn.compressor.ape")


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
        ("layers.1.ffn.w1.weight", "model.layers.1.moe.gate_up_proj.weight", 0),
        ("layers.1.ffn.w3.weight", "model.layers.1.moe.gate_up_proj.weight", 1),
        (
            "layers.1.attn.compressor.wgate.weight",
            "model.layers.1.attention.compressor.fused_wkv_wgate.weight",
            1,
        ),
    ],
)
def test_stacked_parameter_contract(source, target, shard_id):
    assert resolve_stacked_shard(map_checkpoint_name(source)) == StackedShard(
        target, shard_id
    )


def test_attention_weights_are_plain_renames_not_stacked_shards():
    """``wq_a``/``wkv`` used to be two shards merged into one fused
    ``fused_wqa_wkv`` parameter (matching vLLM's own real GPU DeepSeek-V4
    backend). This plugin's ``DeepseekV4Attention`` keeps them separate
    instead (matching the ``transformers`` reference module it is
    cross-validated against), so each checkpoint tensor is now a plain
    rename onto its own standalone parameter, not a fused shard.
    """
    for source in (
        "layers.1.attn.wq_a.weight",
        "layers.1.attn.wkv.weight",
        "layers.1.attn.wq_b.weight",
        "layers.1.attn.q_norm.weight",
        "layers.1.attn.wo_a.weight",
        "layers.1.attn.wo_b.weight",
        "layers.1.attn.attn_sink",
    ):
        assert resolve_stacked_shard(map_checkpoint_name(source)) is None


def test_routed_expert_weights_target_persistent_grouped_tensors():
    assert resolve_stacked_shard(
        map_checkpoint_name("layers.1.ffn.experts.2.w1.weight")
    ) == StackedShard("model.layers.1.moe.routed_gate_up", 0, 2, True)
    assert resolve_stacked_shard(
        map_checkpoint_name("layers.1.ffn.experts.2.w3.weight")
    ) == StackedShard("model.layers.1.moe.routed_gate_up", 1, 2, True)
    mapped = map_checkpoint_name("layers.1.ffn.experts.2.w2.weight")
    assert resolve_stacked_shard(mapped) == StackedShard(
        "model.layers.1.moe.routed_down", 0, 2, True
    )


def test_loader_transposes_official_routed_experts_into_grouped_storage():
    class Grouped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.routed_gate_up = torch.nn.Parameter(torch.zeros(3, 4, 2, 5))
            self.routed_down = torch.nn.Parameter(torch.zeros(3, 5, 4))

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = Grouped()

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    model = Model()
    w1 = torch.arange(20.0).view(5, 4)
    w3 = w1 + 100
    w2 = torch.arange(20.0).view(4, 5) + 200
    loaded = load_checkpoint_weights(
        model,
        [
            ("layers.0.ffn.experts.2.w1.weight", w1),
            ("layers.0.ffn.experts.2.w3.weight", w3),
            ("layers.0.ffn.experts.2.w2.weight", w2),
        ],
    )
    assert loaded == {
        "model.layers.0.moe.routed_gate_up",
        "model.layers.0.moe.routed_down",
    }
    torch.testing.assert_close(model.model.layers[0].moe.routed_gate_up[2, :, 0], w1.T)
    torch.testing.assert_close(model.model.layers[0].moe.routed_gate_up[2, :, 1], w3.T)
    torch.testing.assert_close(model.model.layers[0].moe.routed_down[2], w2.T)


def test_shape_drift_fails_before_copy():
    require_weight_shape("head.weight", (32, 16), (32, 16))
    with pytest.raises(ValueError, match=r"head\.weight.*\(31, 16\).+\(32, 16\)"):
        require_weight_shape("head.weight", (31, 16), (32, 16))


class FakeCompressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_wkv_wgate = torch.nn.Linear(3, 4, bias=False)
        parameter = self.fused_wkv_wgate.weight

        def load_shard(target, source, shard_id):
            target[shard_id * 2 : shard_id * 2 + 2].copy_(source)

        parameter.weight_loader = load_shard


class FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # q_a_proj/kv_proj are plain renames now (see
        # weight_loaders.py::_ATTENTION_RENAMES) -- no weight_loader, exactly
        # like every other non-fused parameter.
        self.q_a_proj = torch.nn.Linear(3, 4, bias=False)
        self.kv_proj = torch.nn.Linear(3, 2, bias=False)
        # compressor.fused_wkv_wgate is still a real fused shard.
        self.compressor = FakeCompressor()


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FakeAttention()


class FakeInnerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(5, 3)
        self.layers = torch.nn.ModuleList([FakeLayer()])


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeInnerModel()
        self.lm_head = torch.nn.Linear(3, 5, bias=False)


def test_loader_copies_plain_and_dispatches_fused_shards():
    model = FakeModel()
    embed = torch.arange(15, dtype=torch.float32).view(5, 3)
    q_a = torch.full((4, 3), 2.0)
    kv = torch.full((4, 3), 3.0)
    loaded = load_checkpoint_weights(
        model,
        [
            ("embed.weight", embed),
            ("layers.0.attn.wq_a.weight", q_a),
            ("layers.0.attn.compressor.wkv.weight", kv[:2]),
            ("layers.0.attn.compressor.wgate.weight", kv[2:]),
        ],
    )
    assert loaded == {
        "model.embed_tokens.weight",
        "model.layers.0.attention.q_a_proj.weight",
        "model.layers.0.attention.compressor.fused_wkv_wgate.weight",
    }
    torch.testing.assert_close(model.model.embed_tokens.weight, embed)
    torch.testing.assert_close(model.model.layers[0].attention.q_a_proj.weight, q_a)
    torch.testing.assert_close(
        model.model.layers[0].attention.compressor.fused_wkv_wgate.weight[:2], kv[:2]
    )
    torch.testing.assert_close(
        model.model.layers[0].attention.compressor.fused_wkv_wgate.weight[2:], kv[2:]
    )


def test_loader_rejects_missing_target_shape_drift_and_duplicate_sources():
    model = FakeModel()
    with pytest.raises(ValueError, match="maps to missing parameter"):
        load_checkpoint_weights(model, [("norm.weight", torch.ones(3))])
    with pytest.raises(ValueError, match="has shape"):
        load_checkpoint_weights(model, [("head.weight", torch.ones(4, 3))])
    with pytest.raises(ValueError, match="duplicate.*head.weight"):
        load_checkpoint_weights(
            model,
            [("head.weight", torch.ones(5, 3))] * 2,
        )
