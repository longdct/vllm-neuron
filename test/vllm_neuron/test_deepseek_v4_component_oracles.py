# SPDX-License-Identifier: Apache-2.0
"""Portable component comparisons against Transformers 5.15."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HashRouter,
    DeepseekV4TopKRouter,
)

from vllm_neuron.model.deepseek_v4.mhc import sinkhorn_positive
from vllm_neuron.model.deepseek_v4.moe import hash_experts, routed_topk


def tiny_config():
    return DeepseekV4Config(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        num_hidden_layers=1,
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["moe"],
    )


def test_sinkhorn_projection_matches_extracted_transformers_math():
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 4)
    eps = 1e-6
    positive = torch.softmax(logits, dim=-1) + eps
    expected = positive / (positive.sum(dim=-2, keepdim=True) + eps)
    for _ in range(19):
        expected = expected / (expected.sum(dim=-1, keepdim=True) + eps)
        expected = expected / (expected.sum(dim=-2, keepdim=True) + eps)
    torch.testing.assert_close(sinkhorn_positive(positive, 20, eps), expected)


def test_routed_moe_selection_and_weights_match_transformers():
    config = tiny_config()
    oracle = DeepseekV4TopKRouter(config)
    torch.manual_seed(4)
    with torch.no_grad():
        oracle.weight.copy_(torch.randn_like(oracle.weight))
        oracle.e_score_correction_bias.copy_(torch.tensor([0.5, -0.2, 0.1, 0.0]))
    hidden = torch.randn(3, config.hidden_size)
    logits, expected_weights, expected_ids = oracle(hidden)
    ids, weights = routed_topk(
        logits,
        oracle.e_score_correction_bias,
        config.num_experts_per_tok,
        config.routed_scaling_factor,
    )
    # Upstream requests unsorted top-k. Compare selected sets and map target
    # weights by expert ID rather than relying on incidental output order.
    for row in range(hidden.shape[0]):
        assert set(ids[row].tolist()) == set(expected_ids[row].tolist())
        actual = {int(i): float(w) for i, w in zip(ids[row], weights[row])}
        expected = {
            int(i): float(w)
            for i, w in zip(expected_ids[row], expected_weights[row])
        }
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_hash_expert_ids_match_transformers_table_lookup():
    config = tiny_config()
    oracle = DeepseekV4HashRouter(config)
    table = torch.arange(config.vocab_size * config.num_experts_per_tok).reshape(
        config.vocab_size, config.num_experts_per_tok
    ) % config.num_local_experts
    oracle.tid2eid.copy_(table)
    tokens = torch.tensor([0, 7, 31])
    assert torch.equal(hash_experts(tokens, oracle.tid2eid), oracle.tid2eid[tokens])
