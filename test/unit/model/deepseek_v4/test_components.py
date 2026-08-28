# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.attention import (
    P2_REPRESENTATIVE_BUCKETS,
    SharedLatentAttentionContract,
    apply_partial_rotary,
    compose_swa_and_compressed_history,
    compressed_entry_slot_mapping,
    gather_bounded_paged_latent,
    gather_paged_latent,
    logical_to_physical_slots_batched,
    mla_attention_reference,
    recent_compressed_logical_indices,
    recent_sliding_logical_indices,
    shared_latent_attention,
    shared_latent_attention_contract_reference,
    visible_compressed_entries,
)


def test_recent_sliding_indices_have_exact_width_and_preserve_leading_holes():
    logical, valid = recent_sliding_logical_indices(torch.tensor([1, 5]), count=4)
    assert logical.tolist() == [[-1, -1, 0, 1], [2, 3, 4, 5]]
    assert valid.tolist() == [
        [False, False, True, True],
        [True, True, True, True],
    ]


def test_recent_compressed_indices_are_bounded_prefix_packed_suffixes():
    logical, valid = recent_compressed_logical_indices(
        torch.tensor([0, 15, 39]), compress_ratio=4, count=4
    )
    assert logical.tolist() == [
        [-1, -1, -1, -1],
        [0, 1, 2, 3],
        [6, 7, 8, 9],
    ]
    assert valid.tolist() == [
        [False, False, False, False],
        [True, True, True, True],
        [True, True, True, True],
    ]


@pytest.mark.parametrize(
    ("raw_capacity", "compress_ratio", "count"),
    [(4096, 128, 32), (32768, 128, 256), (65536, 128, 512), (2048, 4, 512)],
)
def test_capacity_sized_compressed_rows_are_identical_for_every_query(
    raw_capacity, compress_ratio, count
):
    """Sizing ``count`` from capacity makes every query request the same rows.

    This is the premise ``nki_mla._build_uniform_span`` rests on, so pin it
    where the arithmetic lives rather than implicitly inside the kernel. When
    ``count >= raw_capacity // compress_ratio``, no reachable position can push
    ``visible`` past ``count``, so ``used == visible``, ``start`` is identically
    zero, and only ``valid`` varies across queries.
    """
    assert count >= raw_capacity // compress_ratio
    # Every position the context can hold, including its two last ones.
    positions = torch.tensor(
        sorted({*range(min(raw_capacity, 4096)), raw_capacity - 2, raw_capacity - 1})
    )
    logical, valid = recent_compressed_logical_indices(
        positions, compress_ratio=compress_ratio, count=count
    )

    # Every query asks for logical entries 0..count-1; -1 appears only as
    # padding, exactly where ``valid`` is False.
    expected = torch.arange(count, dtype=torch.int32).expand(logical.shape[0], -1)
    assert torch.equal(logical[valid], expected[valid])
    assert (logical[~valid] == -1).all()
    # Validity is non-decreasing in position: the last query's prefix is a
    # superset of every earlier one's, which is why the span is built from it.
    used = valid.sum(dim=1)
    assert torch.equal(used, used.cummax(dim=0).values)


def test_bounded_logical_mapping_validates_requests_columns_and_blocks():
    logical = torch.tensor([[0, 3, 4, -1], [1, 8, 2, 7]], dtype=torch.int32)
    requested = torch.ones_like(logical, dtype=torch.bool)
    tables = torch.tensor([[2, -1], [1, 99]])
    slots, valid = logical_to_physical_slots_batched(
        logical,
        requested,
        tables,
        torch.tensor([0, 1]),
        logical_slots_per_block=4,
        physical_page_stride=8,
        cache_blocks=3,
    )
    assert slots.tolist() == [[16, 19, -1, -1], [9, -1, 10, -1]]
    assert valid.tolist() == [
        [True, True, False, False],
        [True, False, True, False],
    ]


def test_bounded_logical_mapping_rejects_invalid_request_ownership():
    logical = torch.zeros((2, 1), dtype=torch.int32)
    slots, valid = logical_to_physical_slots_batched(
        logical,
        torch.ones_like(logical, dtype=torch.bool),
        torch.tensor([[0]]),
        torch.tensor([-1, 1]),
        logical_slots_per_block=1,
        physical_page_stride=1,
        cache_blocks=1,
    )
    assert slots.tolist() == [[-1], [-1]]
    assert not valid.any()


def test_bounded_paged_latent_contract_handles_all_sentinels_and_slot_zero():
    cache = torch.arange(4 * 1 * 2 * 3, dtype=torch.float32).view(4, 1, 2, 3)
    indices = torch.tensor([[0, -1, 7, 8], [2, 1, 99, 4]])
    visibility = torch.tensor([[True, True, True, True], [False, True, True, True]])
    values, valid = gather_bounded_paged_latent(cache, indices, visibility)
    assert valid.tolist() == [[True, False, True, False], [False, True, False, True]]
    torch.testing.assert_close(values[0, 0], cache[0, 0, 0])
    torch.testing.assert_close(values[0, 2], cache[3, 0, 1])
    assert torch.count_nonzero(values[0, 1]) == 0
    assert torch.count_nonzero(values[1, 0]) == 0


def test_bounded_attention_contract_matches_direct_selected_history():
    torch.manual_seed(19)
    cache = torch.randn(3, 1, 4, 8, dtype=torch.bfloat16)
    query = torch.randn(2, 1, 4, 8, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 3, 8, -1], [4, 5, 11, 12]])
    visible = torch.ones_like(indices, dtype=torch.bool)
    sinks = torch.randn(4, dtype=torch.bfloat16)
    contract = SharedLatentAttentionContract(query, cache, indices, visible, sinks)
    gathered, valid = gather_bounded_paged_latent(cache, indices, visible)
    expected = shared_latent_attention(
        query, gathered, visibility=valid, attention_sinks=sinks
    )
    torch.testing.assert_close(
        shared_latent_attention_contract_reference(contract), expected, rtol=0, atol=0
    )


from vllm_neuron.model.deepseek_v4.compressor import compress_chunk
from vllm_neuron.model.deepseek_v4.mhc import sinkhorn
from vllm_neuron.model.deepseek_v4.moe import (
    dense_expert_affinities,
    hash_experts,
    routed_topk,
)
from vllm_neuron.model.deepseek_v4.nki_mla import shared_latent_mla


def test_nki_mla_cpu_oracle_supports_per_query_validity_and_sinks():
    torch.manual_seed(23)
    query = torch.randn(3, 1, 4, 512, dtype=torch.bfloat16)
    latent = torch.randn(3, 7, 512, dtype=torch.bfloat16)
    validity = torch.arange(7)[None] < torch.tensor([[1], [4], [7]])
    sinks = torch.randn(4, dtype=torch.bfloat16)
    expected = shared_latent_attention(
        query, latent, visibility=validity, attention_sinks=sinks
    )
    torch.testing.assert_close(
        shared_latent_mla(query, latent, validity, sinks), expected, rtol=0, atol=0
    )


def test_nki_mla_rejects_an_unsupported_query_bucket(monkeypatch):
    import vllm_neuron.model.deepseek_v4.nki_mla as mla

    monkeypatch.setattr(mla, "can_run_kernel", lambda _: True)
    query = torch.zeros(2, 1, 16, 512, dtype=torch.bfloat16)
    latent = torch.zeros(2, 128, 512, dtype=torch.bfloat16)
    valid = torch.ones(2, 128, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="query bucket"):
        mla.shared_latent_mla(
            query, latent, valid, torch.zeros(16, dtype=torch.bfloat16)
        )


def test_nki_mla_cpu_oracle_accepts_holes_between_bounded_streams():
    torch.manual_seed(29)
    query = torch.randn(1, 1, 2, 512, dtype=torch.bfloat16)
    latent = torch.randn(1, 9, 512, dtype=torch.bfloat16)
    valid = torch.tensor([[True, False, True, False, True, True, False, True, False]])
    sinks = torch.randn(2, dtype=torch.bfloat16)
    expected = shared_latent_attention(
        query, latent, visibility=valid, attention_sinks=sinks
    )
    torch.testing.assert_close(
        shared_latent_mla(query, latent, valid, sinks), expected, rtol=0, atol=0
    )


def test_dense_affinities_preserve_duplicate_routing_slots():
    ids = torch.tensor([[1, 1, 3, -1], [0, 2, 2, 9]])
    weights = torch.tensor([[0.1, 0.2, 0.4, 7.0], [0.5, 0.1, 0.3, 8.0]])
    actual = dense_expert_affinities(ids, weights, 4)
    expected = torch.tensor([[0.0, 0.3, 0.0, 0.4], [0.5, 0.0, 0.4, 0.0]])
    torch.testing.assert_close(actual, expected)


def test_shared_latent_attention_matches_identity_projection_oracle():
    torch.manual_seed(7)
    q = torch.randn(3, 1, 4, 8, dtype=torch.bfloat16)
    latent = torch.randn(3, 6, 8, dtype=torch.bfloat16)
    visible = torch.rand(3, 6) > 0.3
    visible[:, 0] = True
    sinks = torch.randn(4, dtype=torch.bfloat16)
    eye = torch.eye(8, dtype=torch.bfloat16).expand(4, -1, -1)
    expected = mla_attention_reference(
        q, latent, eye, eye, attention_sinks=sinks, key_valid=visible
    )
    actual = shared_latent_attention(
        q, latent, visibility=visible, attention_sinks=sinks
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_512d_mla_matches_explicit_fp32_oracle():
    torch.manual_seed(1)
    q = torch.randn(1, 3, 2, 512, dtype=torch.bfloat16)
    latent = torch.randn(1, 3, 512, dtype=torch.bfloat16)
    kw = torch.randn(2, 512, 512, dtype=torch.bfloat16) / 32
    vw = torch.randn(2, 512, 16, dtype=torch.bfloat16) / 32
    actual = mla_attention_reference(q, latent, kw, vw)
    k = torch.einsum("bsl,hld->bshd", latent.float(), kw.float())
    v = torch.einsum("bsl,hlv->bshv", latent.float(), vw.float())
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k) / (512**0.5)
    scores = scores.masked_fill(
        ~torch.ones(3, 3, dtype=torch.bool).tril()[None, None], float("-inf")
    )
    expected = torch.einsum("bhts,bshv->bthv", scores.softmax(-1), v).to(q.dtype)
    torch.testing.assert_close(actual, expected)


def test_decode_sees_complete_paged_history():
    q = torch.ones(1, 1, 1, 4)
    latent = torch.arange(12.0).view(1, 3, 4)
    eye = torch.eye(4).view(1, 4, 4)
    out = mla_attention_reference(q, latent, eye, eye)
    assert out.shape == (1, 1, 1, 4)
    assert torch.isfinite(out).all()


def test_partial_rotary_inverse_round_trip():
    x = torch.randn(2, 3, 16)
    angle = torch.randn(2, 3, 4)
    cos, sin = angle.cos(), angle.sin()
    rotated = apply_partial_rotary(x, cos, sin, rope_dim=8)
    restored = apply_partial_rotary(rotated, cos, sin, rope_dim=8, inverse=True)
    torch.testing.assert_close(restored, x, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(rotated[..., :-8], x[..., :-8])


def test_attention_sinks_consume_probability_without_producing_values():
    q = torch.ones(1, 1, 1, 4)
    latent = torch.ones(1, 1, 4)
    eye = torch.eye(4).view(1, 4, 4)
    no_sink = mla_attention_reference(q, latent, eye, eye)
    with_sink = mla_attention_reference(
        q, latent, eye, eye, attention_sinks=torch.tensor([10.0])
    )
    assert with_sink.abs().max() < no_sink.abs().max()


def test_paged_decode_gathers_physical_blocks_in_logical_order():
    cache = torch.arange(4 * 1 * 2 * 3.0).view(4, 1, 2, 3)
    gathered = gather_paged_latent(cache, torch.tensor([2, 0]), sequence_length=3)
    expected = torch.cat((cache[2].transpose(0, 1), cache[0].transpose(0, 1)), dim=0)[
        :3
    ]
    torch.testing.assert_close(gathered, expected)


def test_swa_and_compressed_history_composition():
    local = torch.arange(6.0).view(1, 6, 1)
    compressed = torch.tensor([[[-2.0], [-1.0]]])
    combined = compose_swa_and_compressed_history(local, compressed, sliding_window=3)
    assert combined.flatten().tolist() == [-2.0, -1.0, 3.0, 4.0, 5.0]


def test_representative_buckets_pin_prefill_and_decode_shapes():
    assert {bucket.query_length == 1 for bucket in P2_REPRESENTATIVE_BUCKETS} == {
        False,
        True,
    }
    assert all(bucket.head_dim == 512 for bucket in P2_REPRESENTATIVE_BUCKETS)


def test_sinkhorn_is_doubly_stochastic_after_twenty_iterations():
    matrix = sinkhorn(torch.randn(4, 4), iterations=20).float()
    torch.testing.assert_close(matrix.sum(-1), torch.ones(4), atol=2e-5, rtol=0)
    torch.testing.assert_close(matrix.sum(-2), torch.ones(4), atol=2e-5, rtol=0)


@pytest.mark.parametrize("chunks", [(13,), (1, 3, 2, 7), (4, 4, 5)])
def test_compressor_is_chunk_boundary_invariant(chunks):
    hidden = torch.arange(13 * 3.0).view(13, 3)
    ape = torch.tensor([0.1, 0.2, 0.3, 0.4])
    expected, expected_state = compress_chunk(hidden, ape, 4)
    outputs, state, offset = [], None, 0
    for size in chunks:
        out, state = compress_chunk(hidden[offset : offset + size], ape, 4, state)
        outputs.append(out)
        offset += size
    torch.testing.assert_close(torch.cat(outputs), expected)
    torch.testing.assert_close(state.carry, expected_state.carry)
    assert state.total_tokens == 13


def test_noaux_bias_changes_selection_but_not_selected_gate_values():
    logits = torch.tensor([[4.0, 3.0, 1.0]])
    bias = torch.tensor([0.0, -10.0, 10.0])
    ids, weights = routed_topk(logits, bias, topk=2)
    assert ids.tolist() == [[2, 0]]
    raw = torch.nn.functional.softplus(logits).sqrt()
    selected = raw.gather(-1, ids)
    expected = selected / selected.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(weights, expected)


def test_routed_topk_accepts_xla_list_result(monkeypatch):
    original_topk = torch.topk

    def list_topk(*args, **kwargs):
        result = original_topk(*args, **kwargs)
        return [result.values, result.indices]

    monkeypatch.setattr(torch, "topk", list_topk)
    ids, weights = routed_topk(
        torch.tensor([[4.0, 3.0, 1.0]]), torch.tensor([0.0, -10.0, 10.0]), 2
    )
    assert ids.tolist() == [[2, 0]]
    assert torch.isfinite(weights).all()


def test_hash_routing_is_exact_lookup():
    table = torch.tensor([[2, 1], [0, 3], [3, 2]])
    assert hash_experts(torch.tensor([2, 0]), table).tolist() == [[3, 2], [2, 1]]


def test_compressed_entry_is_visible_to_the_query_that_completes_it():
    """The read side must not lag the write side by one token.

    ``compressed_entry_slot_mapping`` emits an entry when
    ``(pos + 1) % ratio == 0`` and the compressor writes it *before* attention
    reads the history, so that entry is already present for the completing
    query. Counting ``pos // ratio`` instead hides it, which diverges from the
    reference at exactly ``pos % ratio == ratio - 1`` and agrees everywhere
    else -- a pattern sparse enough to survive a casual eyeball.
    """
    ratio = 4
    positions = torch.arange(12)
    visible = visible_compressed_entries(positions, ratio)
    torch.testing.assert_close(visible, (positions + 1) // ratio)

    # Every entry the write side has emitted up to and including `pos` is
    # visible at `pos`, and nothing beyond it is.
    raw_slots = torch.arange(12)
    written = compressed_entry_slot_mapping(
        raw_slots, ratio, raw_block_size=12, physical_page_stride=3
    )
    for pos in range(12):
        emitted = int((written[: pos + 1] >= 0).sum())
        assert int(visible[pos]) == emitted, f"position {pos}"


def test_visible_compressed_entries_rejects_a_non_positive_ratio():
    with pytest.raises(ValueError, match="compress_ratio must be positive"):
        visible_compressed_entries(torch.arange(4), 0)
    (gather_bounded_paged_latent,)
    (shared_latent_attention_contract_reference,)
