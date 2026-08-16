# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_neuron.model.deepseek_v4.config import normalize_config
from vllm_neuron.model.deepseek_v4.factory import (
    ComponentRegistry,
    LayerComponents,
    resolve_layer_components,
)


class SlidingAttention: ...


class C4Attention: ...


class C128Attention: ...


class HashMoE: ...


class RoutedMoE: ...


REGISTRY = ComponentRegistry(
    SlidingAttention,
    C4Attention,
    C128Attention,
    HashMoE,
    RoutedMoE,
)


def config():
    return normalize_config(
        SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "heavily_compressed_attention",
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
            ],
            compress_rates={
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            },
            mlp_layer_types=["hash_moe", "moe", "moe", "moe"],
            sliding_window=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=512,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            index_topk=512,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            num_nextn_predict_layers=0,
            scoring_func="sqrtsoftplus",
            topk_method="noaux_tc",
        )
    )


def test_all_attention_and_mlp_variants_resolve_per_layer():
    assert resolve_layer_components(config(), REGISTRY) == (
        LayerComponents(C128Attention, HashMoE),
        LayerComponents(SlidingAttention, RoutedMoE),
        LayerComponents(C4Attention, RoutedMoE),
        LayerComponents(C128Attention, RoutedMoE),
    )


def test_attention_kind_never_infers_mlp_kind():
    resolved = resolve_layer_components(config(), REGISTRY)
    assert resolved[0].attention is C128Attention
    assert resolved[0].mlp is HashMoE
    assert resolved[3].attention is C128Attention
    assert resolved[3].mlp is RoutedMoE
