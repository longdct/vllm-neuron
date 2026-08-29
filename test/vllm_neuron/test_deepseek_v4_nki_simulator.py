# SPDX-License-Identifier: Apache-2.0

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nki")

import nki

from vllm_neuron.model.deepseek_v4 import nki_mla
from vllm_neuron.model.deepseek_v4.attention import (
    gather_bounded_paged_latent,
    shared_latent_attention,
)
from vllm_neuron.model.deepseek_v4.compressor import (
    compress_csa_chunk,
    compress_hca_chunk,
)
from vllm_neuron.model.deepseek_v4.indexer import lightning_index_scores
from vllm_neuron.model.deepseek_v4.nki_compressor import (
    _paged_candidate_windows,
    _paged_gated_compressor_kernel,
)
from vllm_neuron.model.deepseek_v4.nki_indexer import (
    _decode_indexer_kernel,
    _projected_bf16_indexer_kernel,
    _visible_page_counts,
)
from vllm_neuron.model.deepseek_v4.nki_mla import (
    _manual_shared_latent_mla_kernel,
    _paged_shared_latent_mla_kernel,
    _paged_sliding_latent_mla_kernel,
    simulate_512_mla,
    torch_reference,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="P2.b requires explicit NKI_SIMULATOR=1",
)


def _compressor_simulator_case(
    *,
    ratio: int,
    head_dim: int,
    query: int,
    start: int,
    real_queries: int | None = None,
    null_column: int | None = None,
):
    """Build paged candidate metadata and its portable compressor oracle."""
    overlap = ratio == 4
    coff = 2 if overlap else 1
    width = coff * head_dim
    page = 16
    real_queries = query if real_queries is None else real_queries
    real_positions = torch.arange(start, start + real_queries, dtype=torch.long)
    if real_queries < query:
        positions = torch.cat(
            (real_positions, real_positions[-1:].repeat(query - real_queries))
        )
    else:
        positions = real_positions
    owners = torch.zeros(query, dtype=torch.long)
    if real_queries < query:
        owners[real_queries:] = 1  # padded tail cannot inherit request ownership

    logical_rows = max(int(positions.max()) + 1, coff * ratio)
    columns = (logical_rows + page - 1) // page
    physical_blocks = columns + 3
    generator = torch.Generator().manual_seed(3100 + ratio + head_dim + start)
    permutation = torch.randperm(physical_blocks, generator=generator)
    block_table = permutation[:columns].reshape(1, -1).long()
    if null_column is not None:
        block_table[0, null_column] = -1
    state_cache = (
        torch.randn(
            physical_blocks,
            1,
            page,
            2 * width,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.1
    )
    position_bias = (
        torch.randn(ratio, width, generator=generator, dtype=torch.bfloat16) * 0.1
    )
    boundary = ((positions + 1) % ratio == 0) & (torch.arange(query) < real_queries)
    output_slots = torch.where(
        boundary,
        torch.arange(query, dtype=torch.long) + 17,
        torch.full((query,), -1, dtype=torch.long),
    )
    slots, mask, candidate_positions, selected_slots, valid = _paged_candidate_windows(
        state_cache,
        positions,
        owners,
        block_table,
        output_slots,
        ratio=ratio,
        overlap=overlap,
    )
    lnc = 1 if slots.shape[0] == 1 else 2
    actual = nki.simulate(_paged_gated_compressor_kernel[lnc])(
        state_cache,
        slots,
        mask,
        valid.to(torch.float32),
        position_bias,
        overlap,
    )

    flat = state_cache[:, 0].reshape(-1, 2 * width)
    reference = []
    compressor = compress_csa_chunk if overlap else compress_hca_chunk
    row_valid = mask == 0
    for candidate in range(slots.shape[0]):
        if not bool(valid[candidate]):
            reference.append(torch.zeros(1, head_dim))
            continue
        replay = flat[slots[candidate].long()]
        reduced, _ = compressor(
            replay[None, :, :width],
            replay[None, :, width:],
            position_bias,
            carry_valid=row_valid[candidate],
        )
        reference.append(reduced[:, -1].float())
    expected = torch.cat(reference, dim=0)
    return actual, expected, candidate_positions, selected_slots, valid, row_valid


@pytest.mark.parametrize(
    ("ratio", "head_dim", "position", "is_boundary"),
    [
        (128, 512, 127, True),
        (128, 512, 126, False),
        (4, 512, 3, True),
        (4, 512, 2, False),
    ],
)
def test_paged_gated_compressor_q1_boundary_and_early_history(
    ratio, head_dim, position, is_boundary
):
    actual, expected, candidate_positions, selected_slots, valid, row_valid = (
        _compressor_simulator_case(
            ratio=ratio, head_dim=head_dim, query=1, start=position
        )
    )
    assert valid.tolist() == [is_boundary]
    assert (selected_slots >= 0).tolist() == [is_boundary]
    if ratio == 4 and is_boundary:
        # CSA's first entry has no prior Ca half, but its current Cb half is real.
        assert row_valid[0].tolist() == [False] * 4 + [True] * 4
        assert candidate_positions.tolist() == [0]
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)


@pytest.mark.parametrize(
    ("ratio", "head_dim", "start", "real_queries", "null_column"),
    [
        (128, 512, 37, 460, 3),
        (4, 512, 5, 495, 4),
        (4, 128, 19, 501, 7),
    ],
)
def test_paged_gated_compressor_q512_lnc2_matches_chunk_oracle(
    ratio, head_dim, start, real_queries, null_column
):
    actual, expected, candidate_positions, selected_slots, valid, _ = (
        _compressor_simulator_case(
            ratio=ratio,
            head_dim=head_dim,
            query=512,
            start=start,
            real_queries=real_queries,
            null_column=null_column,
        )
    )
    assert actual.shape == (512 // ratio, head_dim)
    assert candidate_positions[0] % ratio == 0
    assert torch.all(candidate_positions[1:] - candidate_positions[:-1] == ratio)
    assert torch.equal(valid, selected_slots >= 0)
    assert not bool(valid[-1])  # repeated padded tail is never emitted
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)


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


@pytest.mark.parametrize("compressed_count", [0, 512, 1024])
def test_paged_streaming_mla_tile_lnc2_matches_oracle(compressed_count):
    tile = nki_mla._PREFILL_QUERY_TILE
    torch.manual_seed(1901 + compressed_count)
    query = torch.randn(tile, 1, 64, 512, dtype=torch.bfloat16)
    sliding_cache = torch.randn(8, 1, 64, 512, dtype=torch.bfloat16)
    # Sliding rows must be a *shifted* window -- row q is row 0 advanced by q --
    # because that is what recent_sliding_logical_indices produces for the
    # consecutive positions of a prefill tile, and _build_sliding_span relies on
    # it to gather the run once. See test_paged_span_requires_a_shifted_window.
    sliding_slots = (
        torch.arange(128, dtype=torch.int32)[None]
        + torch.arange(tile, dtype=torch.int32)[:, None]
    )
    sliding_valid = torch.arange(128)[None] <= torch.arange(tile)[:, None] * 3
    sinks = torch.randn(64, dtype=torch.bfloat16)
    zero = torch.tensor(0, dtype=torch.bfloat16)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.bfloat16)

    if not compressed_count:
        actual = nki.simulate(_paged_sliding_latent_mla_kernel[2])(
            query,
            sliding_cache,
            sliding_slots,
            torch.where(sliding_valid, zero, neg_inf),
            sinks,
        )
        latent, valid = gather_bounded_paged_latent(
            sliding_cache, sliding_slots, sliding_valid
        )
    else:
        compressed_cache = torch.randn(
            compressed_count // 64 + 1, 1, 64, 512, dtype=torch.bfloat16
        )
        compressed_slots = torch.arange(compressed_count, dtype=torch.int32).repeat(
            tile, 1
        )
        compressed_valid = (torch.arange(compressed_count)[None] % 7 != 0).repeat(
            tile, 1
        )
        # Exercise sanitized invalid physical slots: address zero is harmless
        # because the corresponding validity mask excludes the gathered row.
        compressed_slots[:, ::7] = 0
        actual = nki.simulate(_paged_shared_latent_mla_kernel[2])(
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
        latent = torch.cat((compressed, sliding), dim=1)
        valid = torch.cat((compressed_valid, sliding_valid), dim=1)

    expected = shared_latent_attention(
        query, latent, visibility=valid, attention_sinks=sinks
    )
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)


def _shifted_paged_tile(compressed_count, seed):
    """Build a prefill-shaped tile: sliding rows shifted one position per query."""
    tile = nki_mla._PREFILL_QUERY_TILE
    torch.manual_seed(seed)
    zero = torch.tensor(0, dtype=torch.bfloat16)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.bfloat16)
    sliding_valid = torch.arange(128)[None] <= torch.arange(tile)[:, None] * 3
    inputs = [
        torch.randn(tile, 1, 64, 512, dtype=torch.bfloat16),
        torch.randn(8, 1, 64, 512, dtype=torch.bfloat16),
        torch.arange(128, dtype=torch.int32)[None]
        + torch.arange(tile, dtype=torch.int32)[:, None],
        torch.where(sliding_valid, zero, neg_inf),
    ]
    if compressed_count:
        compressed_slots = torch.arange(compressed_count, dtype=torch.int32).repeat(
            tile, 1
        )
        compressed_valid = (torch.arange(compressed_count)[None] % 7 != 0).repeat(
            tile, 1
        )
        compressed_slots[:, ::7] = 0
        inputs += [
            torch.randn(compressed_count // 64 + 1, 1, 64, 512, dtype=torch.bfloat16),
            compressed_slots,
            torch.where(compressed_valid, zero, neg_inf),
        ]
    inputs.append(torch.randn(64, dtype=torch.bfloat16))
    kernel = (
        _paged_sliding_latent_mla_kernel
        if not compressed_count
        else _paged_shared_latent_mla_kernel
    )
    return kernel, inputs


@pytest.mark.parametrize("compressed_count", [0, 512])
def test_paged_span_gather_is_bit_exact_against_the_per_row_gather(
    monkeypatch, compressed_count
):
    """The span gather must fetch the same rows, in the same order.

    The FP32 oracle cannot establish this: the kernel is BF16 online softmax, so
    it only ever agrees with the oracle to ~2.5%, which would hide a genuine
    reordering. Compare the two gather paths against each other instead.
    """
    kernel, inputs = _shifted_paged_tile(compressed_count, seed=4177)

    monkeypatch.setattr(nki_mla, "_SPAN_GATHER", False)
    per_row = nki.simulate(kernel[2])(*inputs)
    monkeypatch.setattr(nki_mla, "_SPAN_GATHER", True)
    spanned = nki.simulate(kernel[2])(*inputs)

    torch.testing.assert_close(spanned, per_row, rtol=0, atol=0)


def _uniform_paged_tile(compressed_count, *, partial, seed):
    """Build an HCA-shaped tile: one compressed row repeated, prefix packed.

    This mirrors what ``paged_shared_latent_mla``'s ``safe_slots`` hands the
    kernel. Sizing the suffix from the addressable entry capacity makes every
    query request logical entries ``0..count-1`` (``recent_compressed_logical_
    indices`` returns ``start == 0``), so the rows differ only in how many of
    them are visible -- and an entry past a query's prefix is both masked
    ``-inf`` *and* has its physical slot zeroed.

    That zeroing is why the span must be built from the run's **last** query:
    it holds the longest valid prefix, so its row is a superset of every
    earlier query's real slots. Building from the first would feed a later
    query cache row 0 for entries it can actually see.
    """
    tile = nki_mla._PREFILL_QUERY_TILE
    torch.manual_seed(seed)
    zero = torch.tensor(0, dtype=torch.bfloat16)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.bfloat16)

    blocks, page = compressed_count // 64 + 1, 64
    compressed_cache = torch.randn(blocks, 1, page, 512, dtype=torch.bfloat16)
    # A scattered block table, as a real allocator produces: the logical prefix
    # is contiguous but its physical slots are not.
    row = torch.randperm(blocks * page, dtype=torch.int32)[:compressed_count]
    if partial:
        step = compressed_count // tile + 1
        used = (torch.arange(tile) * step + 1).clamp(max=compressed_count)
    else:
        used = torch.full((tile,), compressed_count)
    compressed_valid = torch.arange(compressed_count)[None] < used[:, None]
    compressed_slots = torch.where(
        compressed_valid, row[None].expand(tile, -1), torch.zeros_like(row)
    ).to(torch.int32)

    sliding_valid = torch.arange(128)[None] <= torch.arange(tile)[:, None] * 3
    inputs = [
        torch.randn(tile, 1, 64, 512, dtype=torch.bfloat16),
        torch.randn(8, 1, 64, 512, dtype=torch.bfloat16),
        torch.arange(128, dtype=torch.int32)[None]
        + torch.arange(tile, dtype=torch.int32)[:, None],
        torch.where(sliding_valid, zero, neg_inf),
        compressed_cache,
        compressed_slots,
        torch.where(compressed_valid, zero, neg_inf),
        torch.randn(64, dtype=torch.bfloat16),
        True,  # compressed_uniform
    ]
    return inputs, compressed_valid, sliding_valid


@pytest.mark.parametrize("compressed_count", [64, 256])
@pytest.mark.parametrize("partial", [False, True])
def test_uniform_compressed_span_is_bit_exact_against_the_per_row_gather(
    monkeypatch, compressed_count, partial
):
    """HCA's once-per-launch compressed gather must fetch the same rows.

    ``partial=True`` is the case that matters: early queries see fewer entries
    than the run's last one, so the span source is not interchangeable. The FP32
    oracle agrees only to ~2.5% (BF16 online softmax), which would hide a
    genuine reordering -- compare the two gather paths against each other.
    """
    inputs, _, _ = _uniform_paged_tile(compressed_count, partial=partial, seed=8329)

    monkeypatch.setattr(nki_mla, "_SPAN_GATHER", False)
    per_row = nki.simulate(_paged_shared_latent_mla_kernel[2])(*inputs)
    monkeypatch.setattr(nki_mla, "_SPAN_GATHER", True)
    spanned = nki.simulate(_paged_shared_latent_mla_kernel[2])(*inputs)

    torch.testing.assert_close(spanned, per_row, rtol=0, atol=0)


@pytest.mark.parametrize("partial", [False, True])
def test_uniform_compressed_span_matches_the_per_query_oracle(partial):
    """Pin the span source against ground truth, not just against the other path.

    Both gather paths agreeing cannot rule out both being wrong. Gathering each
    query's own slots independently can: a span built from the run's first query
    instead of its last would feed later queries cache row 0 where they can see
    a real entry, and this comparison would fail.
    """
    inputs, compressed_valid, sliding_valid = _uniform_paged_tile(
        256, partial=partial, seed=1471
    )
    actual = nki.simulate(_paged_shared_latent_mla_kernel[2])(*inputs)

    query, sliding_cache, sliding_slots = inputs[0], inputs[1], inputs[2]
    compressed_cache, compressed_slots, sinks = inputs[4], inputs[5], inputs[7]
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


@pytest.mark.parametrize("compressed_count", [32, 512])
def test_uniform_compressed_span_handles_the_q1_decode_launch(compressed_count):
    """Decode launches one query per program, so the span is its own row.

    ``queries_per_program == 1`` makes ``last_q == first_q == 0``. The span must
    degenerate cleanly rather than address past the single slot row.
    """
    torch.manual_seed(2203 + compressed_count)
    zero = torch.tensor(0, dtype=torch.bfloat16)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.bfloat16)
    query = torch.randn(1, 1, 64, 512, dtype=torch.bfloat16)
    sliding_cache = torch.randn(4, 1, 64, 512, dtype=torch.bfloat16)
    sliding_slots = torch.randperm(256, dtype=torch.int32)[:128].reshape(1, 128)
    sliding_valid = (torch.arange(128) % 5 != 0)[None]
    blocks = compressed_count // 32 + 1
    compressed_cache = torch.randn(blocks, 1, 32, 512, dtype=torch.bfloat16)
    row = torch.randperm(blocks * 32, dtype=torch.int32)[:compressed_count]
    # A partially filled suffix, prefix packed and slot-zeroed past the prefix,
    # exactly as `safe_slots` hands it over.
    compressed_valid = (torch.arange(compressed_count) < compressed_count - 5)[None]
    compressed_slots = torch.where(
        compressed_valid, row[None], torch.zeros_like(row)
    ).to(torch.int32)

    actual = nki.simulate(_paged_shared_latent_mla_kernel[1])(
        query,
        sliding_cache,
        sliding_slots,
        torch.where(sliding_valid, zero, neg_inf),
        compressed_cache,
        compressed_slots,
        torch.where(compressed_valid, zero, neg_inf),
        torch.zeros(64, dtype=torch.bfloat16),
        True,  # compressed_uniform
    )

    compressed, compressed_gathered = gather_bounded_paged_latent(
        compressed_cache, compressed_slots, compressed_valid
    )
    sliding, sliding_gathered = gather_bounded_paged_latent(
        sliding_cache, sliding_slots, sliding_valid
    )
    expected = shared_latent_attention(
        query,
        torch.cat((compressed, sliding), dim=1),
        visibility=torch.cat((compressed_gathered, sliding_gathered), dim=1),
        attention_sinks=torch.zeros(64, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)


def test_capacity_sized_compressed_suffix_matches_the_padded_one():
    """Trailing all-invalid compressed entries are exactly neutral.

    Prefill sizes the HCA suffix from context capacity instead of always
    requesting 1024 entries (model.py::_forward_packed). That is only safe if
    dropping trailing entries which were masked -inf leaves the result
    bit-identical: an all-invalid tile has tile max -inf, so neg_merged_max
    leaves the running max untouched, prior_scale is exp(0) == 1, and the tile
    contributes nothing to the sum. Assert it rather than trusting the argument.
    """
    tile = nki_mla._PREFILL_QUERY_TILE
    torch.manual_seed(6203)
    zero = torch.tensor(0, dtype=torch.bfloat16)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.bfloat16)
    query = torch.randn(tile, 1, 64, 512, dtype=torch.bfloat16)
    sliding_cache = torch.randn(8, 1, 64, 512, dtype=torch.bfloat16)
    sliding_slots = (
        torch.arange(128, dtype=torch.int32)[None]
        + torch.arange(tile, dtype=torch.int32)[:, None]
    )
    sliding_mask = torch.where(
        torch.arange(128)[None] <= torch.arange(tile)[:, None] * 3, zero, neg_inf
    )
    sinks = torch.randn(64, dtype=torch.bfloat16)
    compressed_cache = torch.randn(9, 1, 64, 512, dtype=torch.bfloat16)

    live = 32
    slots_small = torch.arange(live, dtype=torch.int32).repeat(tile, 1)
    mask_small = torch.zeros(tile, live, dtype=torch.bfloat16)
    # The same live entries, padded out to the next compiled bucket.
    slots_big = torch.zeros(tile, 256, dtype=torch.int32)
    slots_big[:, :live] = slots_small
    mask_big = torch.full((tile, 256), float("-inf"), dtype=torch.bfloat16)
    mask_big[:, :live] = 0

    def run(slots, mask):
        return nki.simulate(_paged_shared_latent_mla_kernel[2])(
            query,
            sliding_cache,
            sliding_slots,
            sliding_mask,
            compressed_cache,
            slots,
            mask,
            sinks,
        )

    torch.testing.assert_close(
        run(slots_small, mask_small), run(slots_big, mask_big), rtol=0, atol=0
    )


def test_paged_span_requires_a_shifted_window():
    """Identical sliding rows are the prefill padding tail, and stay finite.

    _build_sliding_span gathers a query run once on the assumption that row q is
    row 0 advanced by q -- true for the consecutive positions of a prefill tile.
    Padding rows break it: they repeat the last real position, so their windows
    are identical rather than shifted. Those outputs are discarded downstream, so
    what matters is that they stay finite and do not contaminate the real rows.
    """
    tile = nki_mla._PREFILL_QUERY_TILE
    torch.manual_seed(5501)
    zero = torch.tensor(0, dtype=torch.bfloat16)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.bfloat16)
    query = torch.randn(tile, 1, 64, 512, dtype=torch.bfloat16)
    sliding_cache = torch.randn(8, 1, 64, 512, dtype=torch.bfloat16)
    # Every row identical: the degenerate all-padding tile.
    sliding_slots = torch.arange(128, dtype=torch.int32).repeat(tile, 1)
    sliding_valid = torch.ones(tile, 128, dtype=torch.bool)
    actual = nki.simulate(_paged_sliding_latent_mla_kernel[2])(
        query,
        sliding_cache,
        sliding_slots,
        torch.where(sliding_valid, zero, neg_inf),
        torch.randn(64, dtype=torch.bfloat16),
    )

    assert torch.isfinite(actual.float()).all()
    # Row 0 of each LNC program's run reads span[0:128], which is exactly its own
    # slot row, so it is correct regardless of the shift assumption.
    expected_first, valid_first = gather_bounded_paged_latent(
        sliding_cache, sliding_slots[:1], sliding_valid[:1]
    )
    torch.testing.assert_close(
        actual[:1],
        shared_latent_attention(
            query[:1],
            expected_first,
            visibility=valid_first,
            attention_sinks=torch.zeros(64, dtype=torch.bfloat16),
        ),
        rtol=0.025,
        atol=0.025,
    )


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
    visible_pages = _visible_page_counts(visible, 512)
    indices, used = nki.simulate(_projected_bf16_indexer_kernel[2])(
        query, keys, gate, visible, visible_pages, 512
    )

    assert torch.equal(used, visible)
    for q_idx, count in enumerate(visible.tolist()):
        if count:
            assert set(indices[q_idx, :count].tolist()) == set(range(count))


def _decode_batch_of_one(visible_row, q_count=8):
    """Shape a batch-1 decode launch the way ``_pad_single_query_bucket`` does."""
    visible = torch.zeros(q_count, dtype=torch.int32)
    visible[0] = visible_row
    return visible


def test_decode_indexer_static_page_unroll_matches_dense_top512():
    """The unrolled decode kernel selects exactly the dense top-512.

    Capacity is four score pages but only 901 entries are visible, so the last
    two pages are scanned and must contribute nothing.  This is the property
    that makes a trace-time page bound safe: work changes, selection does not.
    """
    torch.manual_seed(113)
    query = torch.randn(8, 1, 64, 128, dtype=torch.bfloat16)
    keys = torch.randn(2048, 128, dtype=torch.bfloat16)
    gate = torch.randn(8, 1, 64, dtype=torch.bfloat16)
    visible = _decode_batch_of_one(901)

    indices, used = nki.simulate(_decode_indexer_kernel[2])(
        query, keys, gate, visible, 512, 2048 // 512
    )

    scores = lightning_index_scores(query[:1], keys.unsqueeze(0), gate[:1])[0, 0]
    dense = torch.topk(scores[:901], 512).indices
    assert int(used[0]) == 512

    # The property the static page bound has to preserve: the two pages past
    # ``visible`` were scanned, so nothing they contain may be selected.
    assert int(indices[0].max()) < 901

    # Selection agrees with the dense reference up to BF16 noise at the cut.
    # Scoring runs in BF16, so entries straddling the 512th-ranked score can
    # swap with the 513th; the kept score profile is the invariant claim.
    kept = scores[indices[0].long()].float().sort(descending=True).values
    want = scores[dense.long()].float().sort(descending=True).values
    torch.testing.assert_close(kept, want, rtol=1e-3, atol=1e-3)

    # The seven padded rows are inert.
    assert used[1:].tolist() == [0] * 7


def test_decode_indexer_agrees_with_the_runtime_loop_kernel():
    """Static decode and the runtime-loop prefill kernel select identically.

    The two kernels scan a different number of pages -- decode unrolls the full
    four-page capacity while the prefill path stops at the batch-max visible
    page count -- so agreeing here is the equivalence the fix rests on.
    """
    torch.manual_seed(127)
    query = torch.randn(8, 1, 64, 128, dtype=torch.bfloat16)
    keys = torch.randn(2048, 128, dtype=torch.bfloat16)
    gate = torch.randn(8, 1, 64, dtype=torch.bfloat16)
    visible = _decode_batch_of_one(901)

    static_indices, static_used = nki.simulate(_decode_indexer_kernel[2])(
        query, keys, gate, visible, 512, 2048 // 512
    )
    visible_pages = _visible_page_counts(visible, 2048)
    loop_indices, loop_used = nki.simulate(_projected_bf16_indexer_kernel[2])(
        query, keys, gate, visible, visible_pages, 512
    )

    assert torch.equal(static_used, loop_used)
    assert set(static_indices[0].tolist()) == set(loop_indices[0].tolist())


def test_decode_indexer_paged_indirection_and_null_block():
    """Static page unroll survives block indirection and an invalid block."""
    torch.manual_seed(131)
    blocks, stride, logical = 16, 128, 32
    query = torch.randn(8, 1, 64, 128, dtype=torch.bfloat16)
    gate = torch.randn(8, 1, 64, dtype=torch.bfloat16)
    cache = torch.randn(blocks, stride, 128, dtype=torch.bfloat16)
    table = torch.arange(blocks - 1, -1, -1, dtype=torch.int32)
    block_valid = torch.ones(blocks, dtype=torch.int32)
    block_valid[3] = 0
    safe_table = table.clone()
    safe_table[3] = 0
    visible = _decode_batch_of_one(377)

    indices, used = nki.simulate(_decode_indexer_kernel[2])(
        query,
        cache.reshape(-1, 128),
        gate,
        visible,
        512,
        (blocks * logical) // 512,
        safe_table,
        block_valid,
        stride,
        logical,
    )

    eligible = torch.arange(377)
    eligible = eligible[(eligible // logical) != 3]
    assert int(used[0]) == eligible.numel()
    assert set(indices[0, : int(used[0])].tolist()) == set(eligible.tolist())
    assert used[1:].tolist() == [0] * 7


def test_decode_indexer_zero_visible_row_selects_nothing():
    """A fully masked decode row needs no minimum-one-page workaround."""
    torch.manual_seed(137)
    query = torch.randn(8, 1, 64, 128, dtype=torch.bfloat16)
    keys = torch.randn(2048, 128, dtype=torch.bfloat16)
    gate = torch.randn(8, 1, 64, dtype=torch.bfloat16)
    visible = torch.zeros(8, dtype=torch.int32)

    _, used = nki.simulate(_decode_indexer_kernel[2])(
        query, keys, gate, visible, 512, 2048 // 512
    )
    assert used.tolist() == [0] * 8


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
