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
    DeepseekV4HyperConnection,
    DeepseekV4IndexerScorer,
    DeepseekV4TopKRouter,
)

from vllm_neuron.model.deepseek_v4.compressor import (
    compress_csa_chunk,
    compress_hca_chunk,
    finalize_compressed_entries,
)
from vllm_neuron.model.deepseek_v4.indexer import (
    lightning_index_scores,
    select_compressed_entries,
    selection_mask_from_indices,
)
from vllm_neuron.model.deepseek_v4.mhc import (
    apply_hyperconnection,
    hyperconnection_reference,
    sinkhorn_positive,
)
from vllm_neuron.model.deepseek_v4.moe import hash_experts, hash_topk, routed_topk
from vllm_neuron.model.deepseek_v4.model import DeepseekV4GroupedLinear


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 4, 8), (3, 4, 8), (2, 3, 4, 8)])
def test_grouped_linear_matches_packed_bmm_oracle(dtype, shape):
    """The decode-safe formulation preserves the checkpoint operation."""
    generator = torch.Generator().manual_seed(31)
    module = DeepseekV4GroupedLinear(8, 24, 4).to(dtype=dtype).eval()
    with torch.no_grad():
        module.weight.copy_(
            torch.randn(module.weight.shape, generator=generator).to(dtype)
        )
    x = torch.randn(shape, generator=generator).to(dtype)
    weight = module.weight.view(4, -1, 8).transpose(1, 2)
    flat = x.reshape(-1, 4, 8).transpose(0, 1)
    expected = torch.bmm(flat, weight).transpose(0, 1).reshape(*shape[:-2], 4, -1)

    actual = module(x)

    tolerance = 1e-6 if dtype is torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_grouped_linear_export_has_no_batched_matmul_or_dynamic_scalar_ops():
    module = DeepseekV4GroupedLinear(8, 24, 4).eval()
    exported = torch.export.export(module, (torch.randn(1, 4, 8),))
    graph = str(exported.graph_module.graph)
    forbidden = (
        "bmm",
        "_local_scalar_dense",
        "_assert_scalar",
        "sym_size",
        "nonzero",
    )
    assert not any(operation in graph for operation in forbidden), graph


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


def test_full_hyperconnection_matches_transformers_module():
    config = tiny_config()
    generator = torch.Generator().manual_seed(18)
    oracle = DeepseekV4HyperConnection(config).eval()
    with torch.no_grad():
        for parameter in oracle.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    streams = torch.randn(
        2,
        3,
        config.hc_mult,
        config.hidden_size,
        generator=generator,
    )
    expected_post, expected_comb, expected_collapsed = oracle(streams)
    post, comb, collapsed = hyperconnection_reference(
        streams,
        oracle.fn,
        oracle.base,
        oracle.scale,
        norm_eps=config.rms_norm_eps,
        hc_eps=config.hc_eps,
        iterations=config.hc_sinkhorn_iters,
    )
    torch.testing.assert_close(post, expected_post)
    torch.testing.assert_close(comb, expected_comb)
    torch.testing.assert_close(collapsed, expected_collapsed)

    update = torch.randn(2, 3, config.hidden_size, generator=generator)
    expected_streams = expected_post.unsqueeze(-1) * update.unsqueeze(-2) + torch.matmul(
        expected_comb.transpose(-1, -2), streams
    )
    torch.testing.assert_close(
        apply_hyperconnection(streams, update, post, comb), expected_streams
    )


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


def test_hash_selection_and_learned_weights_match_transformers():
    config = tiny_config()
    oracle = DeepseekV4HashRouter(config)
    table = torch.tensor(
        [[row % 4, (row + 2) % 4] for row in range(config.vocab_size)]
    )
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        oracle.tid2eid.copy_(table)
        oracle.weight.copy_(torch.randn(oracle.weight.shape, generator=generator))
    hidden = torch.randn(2, 3, config.hidden_size, generator=generator)
    # Non-monotonic IDs pin flattening/alignment across batch and sequence.
    input_ids = torch.tensor([[7, 0, 31], [4, 12, 1]])
    logits, expected_weights, expected_ids = oracle(hidden, input_ids)
    actual_ids, actual_weights = hash_topk(
        logits,
        input_ids,
        oracle.tid2eid,
        config.routed_scaling_factor,
    )
    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_weights, expected_weights)


def test_hash_routing_rejects_token_logit_misalignment():
    with pytest.raises(ValueError, match="not aligned"):
        hash_topk(torch.zeros(3, 4), torch.tensor([1, 2]), torch.zeros(8, 2))


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


def indexer_config(index_topk=3, index_n_heads=4, index_head_dim=8):
    """A CSA-only config small enough that top-k actually excludes entries."""
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
        num_attention_heads=1,
        head_dim=16,
        q_lora_rank=8,
        index_topk=index_topk,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
    )


def _seeded_scorer(config, seed=7):
    scorer = DeepseekV4IndexerScorer(config).eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in scorer.parameters():
            parameter.uniform_(-0.5, 0.5)
    return scorer


def _reference_selection(scores, position_ids, compress_rate, index_topk):
    """The reference's own selection, extracted from DeepseekV4Indexer.forward.

    Transformers 5.15 ``modeling_deepseek_v4.py``: mask entries at or past the
    causal threshold, take the top-k, then replace any pick that still points
    past the threshold with the ``-1`` sentinel.
    """
    entries = scores.shape[-1]
    threshold = (position_ids + 1) // compress_rate
    future = torch.arange(entries).view(1, 1, -1) >= threshold.unsqueeze(-1)
    masked = scores.masked_fill(future, float("-inf"))
    chosen = masked.topk(min(index_topk, entries), dim=-1).indices
    invalid = chosen >= threshold.unsqueeze(-1)
    return torch.where(invalid, torch.full_like(chosen, -1), chosen), threshold


def test_index_scores_match_actual_transformers_scorer():
    """``∑_h w_h · ReLU(q_h · k_s)``, bit-exact against the real module."""
    config = indexer_config()
    scorer = _seeded_scorer(config)
    torch.manual_seed(21)
    query = torch.randn(2, 5, config.index_n_heads, config.index_head_dim)
    keys = torch.randn(2, 9, config.index_head_dim)
    hidden = torch.randn(2, 5, config.hidden_size)
    with torch.no_grad():
        expected = scorer(query, keys, hidden)
        actual = lightning_index_scores(query, keys, scorer.weights_proj(hidden))
    torch.testing.assert_close(actual, expected)


def test_selection_matches_extracted_transformers_math_in_the_sparse_regime():
    """40 tokens at ratio 4 gives 10 candidates for a budget of 3.

    Below the dense bound the comparison is vacuous -- selecting the top-k is
    selecting everything -- so this deliberately runs past it and asserts that
    entries really were dropped.
    """
    config = indexer_config()
    compress_rate = config.compress_rates["compressed_sparse_attention"]
    scorer = _seeded_scorer(config, seed=11)
    torch.manual_seed(11)
    entries = 10
    query = torch.randn(2, 40, config.index_n_heads, config.index_head_dim)
    keys = torch.randn(2, entries, config.index_head_dim)
    hidden = torch.randn(2, 40, config.hidden_size)
    position_ids = torch.arange(40).expand(2, -1)
    with torch.no_grad():
        scores = scorer(query, keys, hidden)

    expected, threshold = _reference_selection(
        scores, position_ids, compress_rate, config.index_topk
    )
    actual = select_compressed_entries(scores, threshold, config.index_topk)
    torch.testing.assert_close(actual, expected)

    kept = selection_mask_from_indices(actual, entries).sum(-1)
    assert torch.equal(kept, torch.minimum(threshold, torch.tensor(config.index_topk)))
    assert int((threshold - kept).clamp(min=0).max()) > 0, "nothing was pruned"


def test_selection_mask_matches_the_reference_block_bias():
    """The reference's ``-inf``/0 bias and this plugin's bool mask agree.

    Different spelling, same statement: the reference scatters into a buffer one
    column wider than the entry axis so ``-1`` sentinels have somewhere to land.
    """
    config = indexer_config()
    compress_rate = config.compress_rates["compressed_sparse_attention"]
    scorer = _seeded_scorer(config, seed=5)
    torch.manual_seed(5)
    batch, tokens, entries = 2, 24, 6
    query = torch.randn(batch, tokens, config.index_n_heads, config.index_head_dim)
    keys = torch.randn(batch, entries, config.index_head_dim)
    hidden = torch.randn(batch, tokens, config.hidden_size)
    position_ids = torch.arange(tokens).expand(batch, -1)
    with torch.no_grad():
        scores = scorer(query, keys, hidden)
    chosen, _ = _reference_selection(
        scores, position_ids, compress_rate, config.index_topk
    )

    safe = torch.where(chosen >= 0, chosen, torch.full_like(chosen, entries))
    bias = torch.full((batch, 1, tokens, entries + 1), float("-inf"))
    bias.scatter_(-1, safe.unsqueeze(1), 0.0)
    expected = bias[..., :entries].squeeze(1) == 0.0

    assert torch.equal(selection_mask_from_indices(chosen, entries), expected)


def test_tied_scores_keep_the_right_number_of_entries():
    """Identical keys make every score identical.

    Which entries win is then a tie-break, and tie-breaks are not guaranteed to
    agree between torch CPU and the Torch-XLA bridge. What must hold either way
    is that the scores agree and the budget is spent exactly.
    """
    config = indexer_config()
    compress_rate = config.compress_rates["compressed_sparse_attention"]
    scorer = _seeded_scorer(config, seed=3)
    torch.manual_seed(3)
    batch, tokens, entries = 2, 32, 8
    query = torch.randn(batch, tokens, config.index_n_heads, config.index_head_dim)
    keys = torch.randn(batch, 1, config.index_head_dim).expand(
        batch, entries, config.index_head_dim
    ).contiguous()
    hidden = torch.randn(batch, tokens, config.hidden_size)
    position_ids = torch.arange(tokens).expand(batch, -1)
    with torch.no_grad():
        expected_scores = scorer(query, keys, hidden)
        actual_scores = lightning_index_scores(
            query, keys, scorer.weights_proj(hidden)
        )
    torch.testing.assert_close(actual_scores, expected_scores)

    threshold = (position_ids + 1) // compress_rate
    kept = selection_mask_from_indices(
        select_compressed_entries(actual_scores, threshold, config.index_topk), entries
    ).sum(-1)
    assert torch.equal(kept, torch.minimum(threshold, torch.tensor(config.index_topk)))


def test_indexer_is_exact_dense_attention_below_the_bound():
    """The equivalence ``dense_csa``'s admission bound is derived from.

    Where the eligible set fits inside ``index_topk``, the indexer selects every
    visible entry and nothing else -- so omitting it is not an approximation.
    """
    config = indexer_config(index_topk=16)
    compress_rate = config.compress_rates["compressed_sparse_attention"]
    scorer = _seeded_scorer(config, seed=9)
    torch.manual_seed(9)
    batch, tokens, entries = 2, 28, 7
    query = torch.randn(batch, tokens, config.index_n_heads, config.index_head_dim)
    keys = torch.randn(batch, entries, config.index_head_dim)
    hidden = torch.randn(batch, tokens, config.hidden_size)
    position_ids = torch.arange(tokens).expand(batch, -1)
    with torch.no_grad():
        scores = scorer(query, keys, hidden)

    threshold = (position_ids + 1) // compress_rate
    assert int(threshold.max()) <= config.index_topk, "not below the bound"
    mask = selection_mask_from_indices(
        select_compressed_entries(scores, threshold, config.index_topk), entries
    )
    dense = torch.arange(entries).view(1, 1, -1) < threshold.unsqueeze(-1)
    assert torch.equal(mask, dense)
