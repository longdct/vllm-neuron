# SPDX-License-Identifier: Apache-2.0
"""Portable component comparisons against Transformers 5.15."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACompressor,
    DeepseekV4HCACompressor,
    DeepseekV4HashRouter,
    DeepseekV4TopKRouter,
)

from vllm_neuron.model.deepseek_v4.compressor import (
    compress_csa_chunk,
    compress_hca_chunk,
    finalize_compressed_entries,
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


def test_hca_compressor_matches_extracted_transformers_math_across_chunks():
    generator = torch.Generator().manual_seed(21)
    ratio, head_dim = 4, 7
    kv = torch.randn(2, 11, head_dim, generator=generator)
    gate = torch.randn(2, 11, head_dim, generator=generator)
    bias = torch.randn(ratio, head_dim, generator=generator)
    usable = 8
    windows = kv[:, :usable].view(2, 2, ratio, head_dim)
    logits = gate[:, :usable].view_as(windows) + bias
    expected = (windows * logits.softmax(dim=2, dtype=torch.float32)).sum(dim=2)

    state = None
    outputs = []
    offset = 0
    for size in (3, 2, 6):
        output, state = compress_hca_chunk(
            kv[:, offset : offset + size],
            gate[:, offset : offset + size],
            bias,
            state,
        )
        outputs.append(output)
        offset += size
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
    assert state.total_tokens == 11
    assert state.kv_carry.shape == (2, 3, head_dim)


def test_csa_compressor_matches_extracted_transformers_overlap_math():
    generator = torch.Generator().manual_seed(22)
    ratio, head_dim = 4, 5
    kv = torch.randn(1, 13, 2 * head_dim, generator=generator)
    gate = torch.randn(1, 13, 2 * head_dim, generator=generator)
    bias = torch.randn(ratio, 2 * head_dim, generator=generator)
    windows = kv[:, :12].view(1, 3, ratio, 2 * head_dim)
    logits = gate[:, :12].view_as(windows) + bias
    combined_kv = kv.new_zeros((1, 3, 2 * ratio, head_dim))
    combined_gate = gate.new_full((1, 3, 2 * ratio, head_dim), float("-inf"))
    combined_kv[:, :, ratio:] = windows[..., head_dim:]
    combined_gate[:, :, ratio:] = logits[..., head_dim:]
    combined_kv[:, 1:, :ratio] = windows[:, :-1, :, :head_dim]
    combined_gate[:, 1:, :ratio] = logits[:, :-1, :, :head_dim]
    expected = (
        combined_kv
        * combined_gate.softmax(dim=2, dtype=torch.float32).to(combined_kv.dtype)
    ).sum(dim=2)

    state = None
    outputs = []
    offset = 0
    for size in (5, 3, 5):
        output, state = compress_csa_chunk(
            kv[:, offset : offset + size],
            gate[:, offset : offset + size],
            bias,
            state,
        )
        outputs.append(output)
        offset += size
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
    assert state.total_tokens == 13
    assert state.kv_carry.shape == (1, 1, 2 * head_dim)
    torch.testing.assert_close(state.overlap_kv, windows[:, -1, :, :head_dim])


def test_hca_rmsnorm_and_rope_match_actual_transformers_module():
    config = DeepseekV4Config(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        num_hidden_layers=1,
        layer_types=["heavily_compressed_attention"],
        mlp_layer_types=["moe"],
        num_attention_heads=1,
        head_dim=16,
        q_lora_rank=8,
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 4,
        },
    )
    torch.manual_seed(23)
    oracle = DeepseekV4HCACompressor(config).eval()
    with torch.no_grad():
        for parameter in oracle.parameters():
            parameter.uniform_(-0.25, 0.25)
    hidden = torch.randn(2, 11, config.hidden_size)
    token_positions = torch.arange(11).expand(2, -1)
    expected, _ = oracle(hidden, torch.empty(0), token_positions, None, 0)

    kv = oracle.kv_proj(hidden)
    gate = oracle.gate_proj(hidden)
    reduced, state = compress_hca_chunk(kv, gate, oracle.position_bias)
    entry_positions = torch.arange(reduced.shape[1]) * oracle.compress_rate
    entry_positions = entry_positions.expand(hidden.shape[0], -1)
    cos, sin = oracle.rotary_emb(
        reduced,
        position_ids=entry_positions,
        layer_type=oracle.rope_layer_type,
    )
    actual = finalize_compressed_entries(
        reduced,
        oracle.kv_norm.weight,
        oracle.kv_norm.variance_epsilon,
        cos,
        sin,
    )
    torch.testing.assert_close(actual.unsqueeze(1), expected)
    assert state.kv_carry.shape[1] == 3


def test_csa_full_compressed_cache_matches_actual_transformers_module():
    config = DeepseekV4Config(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        num_hidden_layers=1,
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["moe"],
        num_attention_heads=1,
        head_dim=16,
        q_lora_rank=8,
        index_topk=3,
    )

    class FixedIndexer(torch.nn.Module):
        def forward(self, hidden, q_residual, positions, cache, layer_idx):
            batch, sequence = hidden.shape[:2]
            return torch.zeros(
                batch, sequence, config.index_topk, dtype=torch.long
            )

    torch.manual_seed(24)
    oracle = DeepseekV4CSACompressor(config).eval()
    with torch.no_grad():
        for parameter in oracle.parameters():
            parameter.uniform_(-0.25, 0.25)
    oracle.indexer = FixedIndexer()
    hidden = torch.randn(2, 13, config.hidden_size)
    token_positions = torch.arange(13).expand(2, -1)
    expected, _ = oracle(hidden, torch.empty(0), token_positions, None, 0)

    kv = oracle.kv_proj(hidden)
    gate = oracle.gate_proj(hidden)
    reduced, state = compress_csa_chunk(kv, gate, oracle.position_bias)
    entry_positions = torch.arange(reduced.shape[1]) * oracle.compress_rate
    entry_positions = entry_positions.expand(hidden.shape[0], -1)
    cos, sin = oracle.rotary_emb(
        reduced,
        position_ids=entry_positions,
        layer_type=oracle.rope_layer_type,
    )
    actual = finalize_compressed_entries(
        reduced,
        oracle.kv_norm.weight,
        oracle.kv_norm.variance_epsilon,
        cos,
        sin,
    )
    torch.testing.assert_close(actual.unsqueeze(1), expected)
    assert state.kv_carry.shape[1] == 1
