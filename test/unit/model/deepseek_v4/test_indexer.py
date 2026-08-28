# SPDX-License-Identifier: Apache-2.0
"""Properties of lightning-indexer scoring and selection.

These pin behaviour that holds for *any* weights, so they need no reference
model: that the selection never looks into the future, never keeps more than
its budget, never keeps padding, and -- the property the whole staged bring-up
rests on -- degenerates to dense attention once the budget covers every visible
entry. The bit-exact comparison against Transformers lives in
``test/vllm_neuron/test_deepseek_v4_component_oracles.py``; this file is what
still runs when transformers is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.indexer import (
    lightning_index_scores,
    select_compressed_entries,
    selection_mask_from_indices,
    streaming_topk_compressed_entries,
)


def _scores(batch=2, tokens=6, entries=8, heads=4, head_dim=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, head_dim, generator=generator)
    keys = torch.randn(batch, entries, head_dim, generator=generator)
    gate = torch.randn(batch, tokens, heads, generator=generator)
    return lightning_index_scores(query, keys, gate)


class TestScoring:
    def test_scores_are_fp32_regardless_of_input_dtype(self):
        """Ties decide which entries attention sees, so scoring stays fp32."""
        query = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        keys = torch.randn(1, 5, 8, dtype=torch.bfloat16)
        gate = torch.randn(1, 2, 4, dtype=torch.bfloat16)
        assert lightning_index_scores(query, keys, gate).dtype is torch.float32

    def test_relu_makes_a_zero_key_score_exactly_zero(self):
        """The reader zeroes padding rows; those must contribute nothing.

        This is why masking before the top-k is required rather than tidy: a
        real entry with a negative gate scores below a zeroed padding row.
        """
        query = torch.randn(1, 3, 4, 8)
        keys = torch.zeros(1, 5, 8)
        gate = torch.randn(1, 3, 4)
        assert torch.equal(
            lightning_index_scores(query, keys, gate), torch.zeros(1, 3, 5)
        )

    def test_negative_gate_produces_scores_below_a_zeroed_row(self):
        query = torch.ones(1, 1, 2, 4)
        keys = torch.cat([torch.ones(1, 1, 4), torch.zeros(1, 1, 4)], dim=1)
        gate = torch.full((1, 1, 2), -1.0)
        scores = lightning_index_scores(query, keys, gate)
        assert scores[0, 0, 0] < scores[0, 0, 1] == 0.0

    @pytest.mark.parametrize(
        "query,keys,gate",
        [
            (torch.randn(2, 4, 8), torch.randn(2, 5, 8), torch.randn(2, 3, 4)),
            (torch.randn(1, 3, 4, 8), torch.randn(1, 5, 6), torch.randn(1, 3, 4)),
            (torch.randn(1, 3, 4, 8), torch.randn(1, 5, 8), torch.randn(1, 3, 9)),
        ],
    )
    def test_shape_disagreements_raise(self, query, keys, gate):
        with pytest.raises(ValueError):
            lightning_index_scores(query, keys, gate)


class TestSelection:
    def test_never_selects_an_entry_the_token_cannot_see(self):
        scores = _scores(tokens=6, entries=8)
        visible = torch.arange(6).expand(2, -1)
        chosen = select_compressed_entries(scores, visible, 3)
        real = chosen >= 0
        assert bool(
            (chosen[real] < visible.unsqueeze(-1).expand_as(chosen)[real]).all()
        )

    def test_keeps_exactly_min_of_visible_and_budget(self):
        entries, topk = 8, 3
        scores = _scores(tokens=10, entries=entries)
        visible = torch.arange(10).clamp(max=entries).expand(2, -1)
        mask = selection_mask_from_indices(
            select_compressed_entries(scores, visible, topk), entries
        )
        expected = torch.minimum(visible, torch.tensor(topk))
        assert torch.equal(mask.sum(-1), expected)

    def test_a_token_with_nothing_visible_selects_nothing(self):
        scores = _scores(tokens=4, entries=6)
        visible = torch.zeros(2, 4, dtype=torch.long)
        chosen = select_compressed_entries(scores, visible, 3)
        assert torch.equal(chosen, torch.full_like(chosen, -1))
        assert not bool(selection_mask_from_indices(chosen, 6).any())

    def test_output_width_is_min_of_budget_and_entries(self):
        """Width is a compile-time constant, so it cannot track the live count."""
        scores = _scores(tokens=3, entries=5)
        visible = torch.full((2, 3), 5)
        assert select_compressed_entries(scores, visible, 2).shape[-1] == 2
        assert select_compressed_entries(scores, visible, 9).shape[-1] == 5

    def test_selection_picks_the_highest_scoring_visible_entries(self):
        scores = torch.tensor([[[0.1, 0.9, 0.5, 0.7]]])
        visible = torch.tensor([[4]])
        chosen = select_compressed_entries(scores, visible, 2)
        assert sorted(chosen[0, 0].tolist()) == [1, 3]

    def test_rejects_a_non_positive_budget(self):
        with pytest.raises(ValueError, match="index_topk"):
            select_compressed_entries(_scores(), torch.zeros(2, 6, dtype=torch.long), 0)


class TestIndexDtypeAndRange:
    """Indices must be signed and in range before anything indexes with them.

    Neuron's top-k returns **uint32** where CPU returns int64. On an unsigned
    type the ``-1`` sentinel wraps to 4294967295 and ``>= 0`` is vacuously
    true, so the sentinel is never recognised and that value is handed to the
    scatter as a real index -- which the device reports as "indirect memory
    copy via vector DGE out-of-bound access" against the whole NEFF, naming no
    op. Measured on trn2: with nothing visible, ``safe`` came back as
    4294967295 for a 33-wide buffer.

    CPU cannot reproduce the dtype, so these pin the contract instead.
    """

    def test_selection_is_signed_so_the_sentinel_survives(self):
        scores = _scores(tokens=4, entries=6)
        visible = torch.zeros(2, 4, dtype=torch.long)
        chosen = select_compressed_entries(scores, visible, 3)
        assert chosen.dtype is torch.int64
        assert int(chosen.min()) == -1

    def test_selection_indices_stay_inside_the_entry_axis(self):
        """Nothing visible means every row is -inf, and only CPU promises
        0..k-1 there. The clamp makes the result well-defined regardless."""
        entries = 6
        scores = _scores(tokens=5, entries=entries)
        for visible_value in (0, 1, entries):
            visible = torch.full((2, 5), visible_value, dtype=torch.long)
            chosen = select_compressed_entries(scores, visible, 3)
            real = chosen[chosen >= 0]
            assert real.numel() == 0 or int(real.max()) < entries

    def test_mask_tolerates_an_unsigned_index_tensor(self):
        """The exact shape of the device bug, forced on CPU."""
        entries = 6
        wrapped = torch.tensor([[[4294967295, 2]]], dtype=torch.uint32)
        mask = selection_mask_from_indices(wrapped, entries)
        assert mask.shape == (1, 1, entries)
        # The wrapped sentinel must not select anything, and must not index
        # outside the buffer; entry 2 is a genuine pick and must survive.
        assert mask[0, 0].tolist() == [False, False, True, False, False, False]

    def test_mask_ignores_an_out_of_range_index(self):
        indices = torch.tensor([[[99, 1, -1]]], dtype=torch.long)
        mask = selection_mask_from_indices(indices, 4)
        assert mask.shape == (1, 1, 4)
        assert bool(mask[0, 0, 1])


class TestDenseEquivalence:
    """The property ``dense_csa``'s admission bound is derived from.

    Where the budget covers every visible entry, selecting the top-k *is*
    selecting everything, so skipping the indexer is exact rather than
    approximate. If this ever fails, the bound is not a bound.
    """

    @pytest.mark.parametrize("entries", [1, 5, 16])
    def test_budget_at_or_above_the_visible_count_selects_all_of_it(self, entries):
        tokens = 7
        scores = _scores(tokens=tokens, entries=entries, seed=entries)
        visible = torch.arange(tokens).clamp(max=entries).expand(2, -1)
        mask = selection_mask_from_indices(
            select_compressed_entries(scores, visible, entries), entries
        )
        positions = torch.arange(entries).view(1, 1, -1)
        assert torch.equal(mask, positions < visible.unsqueeze(-1))

    def test_one_entry_over_the_budget_is_the_first_pruning(self):
        entries, topk = 6, 3
        scores = _scores(tokens=8, entries=entries, seed=5)
        visible = torch.arange(8).clamp(max=entries).expand(2, -1)
        kept = selection_mask_from_indices(
            select_compressed_entries(scores, visible, topk), entries
        ).sum(-1)
        assert bool((kept[visible <= topk] == visible[visible <= topk]).all())
        assert bool((kept[visible > topk] == topk).all())


class TestGraphSafety:
    def test_selection_exports_without_dynamic_shape_ops(self):
        """Nothing here may make a shape depend on a tensor value."""

        class Selection(torch.nn.Module):
            def forward(self, query, keys, gate, visible):
                scores = lightning_index_scores(query, keys, gate)
                chosen = select_compressed_entries(scores, visible, 3)
                return selection_mask_from_indices(chosen, keys.shape[1])

        exported = torch.export.export(
            Selection(),
            (
                torch.randn(1, 1, 4, 8),
                torch.randn(1, 6, 8),
                torch.randn(1, 1, 4),
                torch.tensor([[4]]),
            ),
        )
        graph = str(exported.graph_module.graph)
        forbidden = ("_local_scalar_dense", "_assert_scalar", "sym_size", "nonzero")
        assert not any(operation in graph for operation in forbidden), graph


@pytest.mark.parametrize("entries", [0, 17, 512, 513, 1301])
def test_streamed_page_merge_matches_dense_topk(entries):
    torch.manual_seed(31 + entries)
    query = torch.randn(1, 3, 4, 8)
    keys = torch.randn(1, entries, 8)
    gate = torch.randn(1, 3, 4)
    visible = torch.tensor([[0, min(entries, 300), entries]])
    selected = streaming_topk_compressed_entries(
        query, keys, gate, visible, topk=512, page_size=512
    )
    assert selected.logical_indices.dtype == torch.int32
    assert torch.equal(selected.valid, selected.logical_indices >= 0)
    if entries:
        scores = lightning_index_scores(query, keys, gate)
        dense = select_compressed_entries(scores, visible, 512)[0]
        for row in range(3):
            streamed_ids = selected.logical_indices[row][selected.valid[row]].long()
            dense_ids = dense[row][dense[row] >= 0]
            assert streamed_ids.numel() == dense_ids.numel()
            torch.testing.assert_close(
                scores[0, row, streamed_ids].sort(descending=True).values,
                scores[0, row, dense_ids].sort(descending=True).values,
                rtol=0,
                atol=0,
            )


def test_streamed_selection_supports_one_independent_history_per_packed_query():
    torch.manual_seed(41)
    query = torch.randn(3, 1, 4, 8)
    keys = torch.randn(3, 700, 8)
    gate = torch.randn(3, 1, 4)
    visible = torch.tensor([[0], [17], [700]])
    selected = streaming_topk_compressed_entries(query, keys, gate, visible)
    assert selected.logical_indices.shape == (3, 512)
    assert selected.valid.sum(1).tolist() == [0, 17, 512]
