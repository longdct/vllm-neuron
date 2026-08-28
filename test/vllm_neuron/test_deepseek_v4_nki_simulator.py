# SPDX-License-Identifier: Apache-2.0

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nki")

import nki

from vllm_neuron.model.deepseek_v4.attention import (
    gather_bounded_paged_latent,
    shared_latent_attention,
)
from vllm_neuron.model.deepseek_v4.indexer import lightning_index_scores
from vllm_neuron.model.deepseek_v4.nki_indexer import (
    _projected_bf16_indexer_kernel,
)
from vllm_neuron.model.deepseek_v4.nki_mla import (
    _manual_shared_latent_mla_kernel,
    _paged_shared_latent_mla_kernel,
    simulate_512_mla,
    torch_reference,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="P2.b requires explicit NKI_SIMULATOR=1",
)


@pytest.mark.parametrize(
    ("query_length", "context_length", "causal"),
    [(1, 8, False), (8, 8, True)],
)
def test_512d_prefill_and_decode_match_fp32_reference(
    query_length, context_length, causal
):
    generator = torch.Generator().manual_seed(11 + query_length)
    query = torch.randn(1, query_length, 512, generator=generator, dtype=torch.bfloat16)
    key = torch.randn(1, context_length, 512, generator=generator, dtype=torch.bfloat16)
    value = torch.randn(
        1, context_length, 512, generator=generator, dtype=torch.bfloat16
    )
    expected = torch_reference(query, key, value, causal=causal)
    actual = simulate_512_mla(query, key, value, causal=causal)
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)


@pytest.mark.parametrize("compressed_count", [32, 256, 512, 1024])
def test_paged_shared_latent_mla_gathers_separate_streams_inside_kernel(
    compressed_count,
):
    torch.manual_seed(887)
    query = torch.randn(1, 1, 64, 512, dtype=torch.bfloat16)
    sliding_cache = torch.randn(4, 1, 64, 512, dtype=torch.bfloat16)
    compressed_cache = torch.randn(
        compressed_count // 32 + 1, 1, 32, 512, dtype=torch.bfloat16
    )
    sliding_slots = torch.randperm(256, dtype=torch.int32)[:128].reshape(1, 128)
    compressed_capacity = compressed_cache.shape[0] * compressed_cache.shape[2]
    compressed_slots = torch.randperm(compressed_capacity, dtype=torch.int32)[
        :compressed_count
    ].reshape(1, compressed_count)
    sliding_valid = (torch.arange(128) % 5 != 0)[None]
    compressed_valid = (torch.arange(compressed_count) % 3 != 0)[None]
    sinks = torch.randn(64, dtype=torch.bfloat16)
    zero = torch.tensor(0, dtype=torch.bfloat16)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.bfloat16)
    actual = nki.simulate(_paged_shared_latent_mla_kernel[1])(
        query,
        sliding_cache,
        sliding_slots,
        torch.where(sliding_valid, zero, neg_inf),
        compressed_cache,
        compressed_slots,
        torch.where(compressed_valid, zero, neg_inf),
        sinks,
    )
    compressed, compressed_valid = gather_bounded_paged_latent(
        compressed_cache, compressed_slots, compressed_valid
    )
    sliding, sliding_valid = gather_bounded_paged_latent(
        sliding_cache, sliding_slots, sliding_valid
    )
    expected = shared_latent_attention(
        query,
        torch.cat((compressed, sliding), dim=1),
        visibility=torch.cat((compressed_valid, sliding_valid), dim=1),
        attention_sinks=sinks,
    )
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)


def test_projected_indexer_page_merge_matches_dense_top512():
    torch.manual_seed(103)
    query = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
    keys = torch.randn(1024, 128, dtype=torch.bfloat16)
    gate = torch.randn(1, 1, 64, dtype=torch.bfloat16)
    visible = torch.tensor([901], dtype=torch.int32)
    visible_pages = torch.tensor([2], dtype=torch.int32)
    indices, used = nki.simulate(_projected_bf16_indexer_kernel[1])(
        query, keys, gate, visible, visible_pages, 512
    )
    scores = lightning_index_scores(query, keys.unsqueeze(0), gate)[0, 0]
    dense = torch.topk(scores[:901], 512).indices
    assert int(used[0]) == 512
    assert set(indices[0].tolist()) == set(dense.tolist())


def test_projected_indexer_paged_indirection_and_null_block():
    torch.manual_seed(107)
    blocks, stride, logical = 16, 128, 32
    query = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
    gate = torch.randn(1, 1, 64, dtype=torch.bfloat16)
    cache = torch.randn(blocks, stride, 128, dtype=torch.bfloat16)
    table = torch.arange(blocks - 1, -1, -1, dtype=torch.int32)
    block_valid = torch.ones(blocks, dtype=torch.int32)
    block_valid[3] = 0
    safe_table = table.clone()
    safe_table[3] = 0
    visible = torch.tensor([377], dtype=torch.int32)
    visible_pages = torch.tensor([1], dtype=torch.int32)
    indices, used = nki.simulate(_projected_bf16_indexer_kernel[1])(
        query,
        cache.reshape(-1, 128),
        gate,
        visible,
        visible_pages,
        512,
        safe_table,
        block_valid,
        stride,
        logical,
    )
    eligible = torch.arange(377)
    eligible = eligible[(eligible // logical) != 3]
    assert int(used[0]) == eligible.numel()
    assert set(indices[0, : int(used[0])].tolist()) == set(eligible.tolist())


def test_projected_indexer_q16_tile_handles_runtime_visible_page_counts():
    torch.manual_seed(109)
    query = torch.randn(16, 1, 64, 128, dtype=torch.bfloat16)
    keys = torch.randn(512, 128, dtype=torch.bfloat16)
    gate = torch.randn(16, 1, 64, dtype=torch.bfloat16)
    visible = torch.tensor(
        [0, 1, 17, 31, 32, 127, 255, 511, 512, 3, 33, 129, 257, 377, 500, 64],
        dtype=torch.int32,
    )
    visible_pages = ((visible + 511) // 512).to(torch.int32)
    indices, used = nki.simulate(_projected_bf16_indexer_kernel[2])(
        query, keys, gate, visible, visible_pages, 512
    )

    assert torch.equal(used, visible)
    for q_idx, count in enumerate(visible.tolist()):
        if count:
            assert set(indices[q_idx, :count].tolist()) == set(range(count))


@pytest.mark.parametrize(
    ("history", "used"), [(128, 73), (160, 117), (640, 503), (1152, 901)]
)
def test_manual_shared_latent_mla_matches_sink_aware_oracle(history, used):
    torch.manual_seed(211 + history)
    query = torch.randn(1, 1, 64, 512, dtype=torch.bfloat16)
    latent = torch.randn(1, history, 512, dtype=torch.bfloat16)
    validity = (torch.arange(history) < used)[None]
    sinks = torch.randn(64, dtype=torch.float32)
    attention_mask = torch.where(validity, 0.0, float("-inf")).to(torch.bfloat16)
    actual = nki.simulate(_manual_shared_latent_mla_kernel[1])(
        query, latent, attention_mask, sinks.to(torch.bfloat16)
    )
    expected = shared_latent_attention(
        query, latent, visibility=validity, attention_sinks=sinks
    )
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)


def test_manual_shared_latent_mla_shards_queries_across_lnc2_programs():
    torch.manual_seed(701)
    query = torch.randn(2, 1, 64, 512, dtype=torch.bfloat16)
    latent = torch.randn(2, 128, 512, dtype=torch.bfloat16)
    valid = torch.ones(2, 128, dtype=torch.bool)
    sinks = torch.randn(64, dtype=torch.bfloat16)
    actual = nki.simulate(_manual_shared_latent_mla_kernel[2])(
        query,
        latent,
        torch.zeros(2, 128, dtype=torch.bfloat16),
        sinks,
    )
    expected = shared_latent_attention(
        query, latent, visibility=valid, attention_sinks=sinks
    )
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)
