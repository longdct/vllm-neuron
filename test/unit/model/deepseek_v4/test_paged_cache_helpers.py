# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Step 2 paged-cache addressing helpers.

Covers ``attention.py``'s ``scatter_paged_latent``/``compressed_entry_slot_mapping``
and ``compressor.py``'s ``carry_gather_length``/``carry_replay_already_emitted``
in isolation, before they're exercised end-to-end in
``test/vllm_neuron/test_deepseek_v4_model_assembly.py``.
"""

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.attention import (
    compressed_entry_slot_mapping,
    gather_paged_latent,
    scatter_paged_latent,
)
from vllm_neuron.model.deepseek_v4.compressor import (
    carry_gather_length,
    carry_gather_length_tensor,
    carry_replay_already_emitted,
    compress_csa_chunk,
    compress_hca_chunk,
)


def test_scatter_paged_latent_writes_and_ignores_padding():
    cache = torch.zeros(2, 1, 4, 3)  # [blocks, heads, slots, latent]
    values = torch.tensor([[1.0, 1, 1], [2.0, 2, 2], [9.0, 9, 9]])
    slots = torch.tensor([0, 5, -1])  # block0/slot0, block1/slot1, padding
    scatter_paged_latent(cache, slots, values)
    assert torch.equal(cache[0, 0, 0], values[0])
    assert torch.equal(cache[1, 0, 1], values[1])
    assert torch.equal(cache[0, 0, 1], torch.zeros(3))  # untouched
    assert cache.sum().item() == pytest.approx(float(values[:2].sum()))


def test_scatter_paged_latent_round_trips_with_gather():
    cache = torch.zeros(3, 1, 2, 4)
    values = torch.randn(5, 4)
    slots = torch.arange(5)  # blocks 0,0,1,1,2 ; offsets 0,1,0,1,0
    scatter_paged_latent(cache, slots, values)
    block_table = torch.arange(3, dtype=torch.long)
    gathered = gather_paged_latent(cache, block_table, 5).squeeze(1)
    torch.testing.assert_close(gathered, values)


def test_gather_paged_latent_reads_stale_columns_once_swa_window_has_advanced():
    """Reproduces the (now-fixed) bug directly: a block table shaped exactly
    like real vLLM produces once a sliding-window sequence has run past one
    window -- old (low-index) columns remapped to the null block (id 0,
    matching this plugin's own ``test_sliding_window_remapping_uses_null_
    blocks_but_latents_remain_stable``), live data at high-index columns.
    ``gather_paged_latent`` must be given ``start_token`` (the live window's
    true start) to read the right columns -- see
    docs/model-dev/deepseek-v4-swa-null-block-bug.md and
    ``tools/deepseek_v4/check_swa_null_block_bug.py`` for a standalone,
    narrated version of this same repro.
    """
    block_size = 2
    sliding_window = 4
    cached_seq_len = 10  # already-computed tokens, well past one window
    live_start_token = cached_seq_len - sliding_window  # = 6

    # Logical column i covers tokens [i*block_size, (i+1)*block_size). Null
    # out any column entirely below live_start_token; give every other
    # column a distinct physical block holding its own token positions.
    cache = torch.zeros(9, 1, block_size, 1)
    cache[0] = -999.0  # the null block: a recognizably-wrong sentinel
    block_table_row = []
    for logical_idx in range(8):
        block_start = logical_idx * block_size
        if block_start + block_size <= live_start_token:
            block_table_row.append(0)
            continue
        physical_id = logical_idx + 1
        block_table_row.append(physical_id)
        for slot in range(block_size):
            token = block_start + slot
            if token < cached_seq_len:
                cache[physical_id, 0, slot, 0] = float(token)
    block_table = torch.tensor(block_table_row, dtype=torch.long)

    gather_len = min(cached_seq_len, sliding_window)
    gathered = gather_paged_latent(
        cache, block_table, gather_len, start_token=live_start_token
    ).squeeze(1).squeeze(-1)
    expected = torch.arange(cached_seq_len - gather_len, cached_seq_len, dtype=torch.float32)
    torch.testing.assert_close(gathered, expected)


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressed_entry_slot_mapping_matches_upstream_formula(ratio):
    """Ported from vLLM 0.24's own DeepSeek-V4 GPU backend
    (sparse_swa.py::_compressed_slot_mapping_kernel): valid iff
    (pos+1) % ratio == 0, entry = pos // ratio."""
    positions = torch.arange(0, 3 * ratio)
    block_table_stub = torch.zeros(1, dtype=torch.long)  # single block, block 0
    raw_slot = positions  # block_size == ratio*k for some k; use block 0 only
    entry_slot = compressed_entry_slot_mapping(raw_slot, ratio, 3 * ratio, 3)
    expected_valid = (positions + 1) % ratio == 0
    assert torch.equal(entry_slot >= 0, expected_valid)
    assert torch.equal(entry_slot[expected_valid], positions[expected_valid] // ratio)
    del block_table_stub


def test_compressed_entry_slot_mapping_treats_padding_as_invalid():
    raw_slot = torch.tensor([-1, 3, 6, -1])
    entry_slot = compressed_entry_slot_mapping(raw_slot, 4, 8, 2)
    # slot 3: (3+1)%4==0 -> completes entry 0. slot 6: (6+1)%4==3 -> not a
    # boundary, invalid regardless of padding.
    assert entry_slot.tolist() == [-1, 0, -1, -1]


@pytest.mark.parametrize("ratio,needs_overlap", [(128, False), (4, True)])
def test_carry_gather_length_never_exceeds_the_declared_state_window(
    ratio, needs_overlap
):
    coff = 2 if needs_overlap else 1
    window = coff * ratio
    for cached_seq_len in range(0, 5 * ratio):
        n = carry_gather_length(cached_seq_len, ratio, needs_overlap=needs_overlap)
        assert 0 <= n <= min(cached_seq_len, window)


def test_carry_replay_reproduces_incremental_hca_chunking():
    """Ground-truth cross-check: replaying carry_gather_length rows through
    the stateless function, dropping carry_replay_already_emitted leading
    rows, exactly reproduces incremental state-based chunking -- for
    arbitrary chunk boundaries, including exact multiples of ratio."""
    torch.manual_seed(0)
    ratio, head_dim = 4, 6
    total_len = 20
    kv_all = torch.randn(1, total_len, head_dim)
    gate_all = torch.randn(1, total_len, head_dim)
    bias = torch.randn(ratio, head_dim)

    for chunks in ([3, 5, 2, 10], [4, 4, 4, 4, 4], [1] * 20, [20]):
        state = None
        ref_outputs = []
        pos = 0
        for size in chunks:
            out, state = compress_hca_chunk(
                kv_all[:, pos : pos + size], gate_all[:, pos : pos + size], bias, state
            )
            ref_outputs.append(out)
            pos += size
        ref = torch.cat(ref_outputs, dim=1)

        history_kv = history_gate = torch.zeros(1, 0, head_dim)
        dev_outputs = []
        cached_seq_len = 0
        pos = 0
        for size in chunks:
            kv_c = kv_all[:, pos : pos + size]
            gate_c = gate_all[:, pos : pos + size]
            n = carry_gather_length(cached_seq_len, ratio, needs_overlap=False)
            drop = carry_replay_already_emitted(
                cached_seq_len, ratio, needs_overlap=False
            )
            replay_kv = torch.cat((history_kv[:, -n:] if n else history_kv[:, :0], kv_c), dim=1)
            replay_gate = torch.cat(
                (history_gate[:, -n:] if n else history_gate[:, :0], gate_c), dim=1
            )
            out, _ = compress_hca_chunk(replay_kv, replay_gate, bias, None)
            dev_outputs.append(out[:, drop:])
            history_kv = torch.cat((history_kv, kv_c), dim=1)
            history_gate = torch.cat((history_gate, gate_c), dim=1)
            cached_seq_len += size
            pos += size
        dev = torch.cat(dev_outputs, dim=1)
        torch.testing.assert_close(dev, ref, rtol=0, atol=1e-5)


@pytest.mark.parametrize("ratio,needs_overlap", [(128, False), (4, True)])
def test_carry_gather_length_tensor_matches_python_int_version(ratio, needs_overlap):
    """Stage A equivalence check: carry_gather_length_tensor's torch.where/
    torch.minimum formula must reproduce carry_gather_length's Python-int
    result exactly, for every cached_seq_len -- it's the tensor-valued
    building block DeepseekV4Compressor._carry_rows uses to build a validity
    mask instead of a data-dependent-length slice."""
    for cached_seq_len in range(0, 5 * ratio):
        expected = carry_gather_length(cached_seq_len, ratio, needs_overlap=needs_overlap)
        actual = carry_gather_length_tensor(
            torch.tensor(cached_seq_len), ratio, needs_overlap=needs_overlap
        )
        assert int(actual) == expected


def _fixed_window_carry(history_kv, history_gate, cached_seq_len, ratio, coff, needs_overlap):
    """Test-only stand-in for DeepseekV4Compressor._carry_rows's masking
    logic, operating on a plain in-memory history buffer instead of a paged
    cache + gather_recent_window (that combination is covered separately by
    test_carry_rows_reads_live_window_correctly_past_carry_cache_eviction).
    Isolates the carry_valid masking/replay math from the paged-cache
    plumbing."""
    carry_window = coff * ratio - 1
    available = history_kv.shape[1]
    take = min(carry_window, available)
    pad = carry_window - take
    carry_kv = torch.cat((history_kv.new_zeros((1, pad, history_kv.shape[-1])), history_kv[:, -take:] if take else history_kv[:, :0]), dim=1)
    carry_gate = torch.cat((history_gate.new_zeros((1, pad, history_gate.shape[-1])), history_gate[:, -take:] if take else history_gate[:, :0]), dim=1)
    exists = torch.cat((torch.zeros(pad, dtype=torch.bool), torch.ones(take, dtype=torch.bool)))
    gather_n = carry_gather_length_tensor(
        torch.tensor(cached_seq_len), ratio, needs_overlap=needs_overlap
    )
    idx = torch.arange(carry_window)
    carry_valid = exists & (idx >= (carry_window - gather_n))
    full_valid = torch.cat((carry_valid, carry_valid.new_ones(1)))
    return carry_kv, carry_gate, full_valid


def test_compress_hca_chunk_carry_valid_masking_matches_incremental_slicing():
    """The key regression test for the _carry_rows Dynamo fix: walking a
    token-by-token sequence (matching the real per-token call pattern --
    DeepseekV4Compressor.forward has exactly one call site, always fed one
    new raw token), the new fixed-window + carry_valid masking path must
    reproduce the old, already-oracle-validated carry_gather_length + slice
    path's output bit-exactly at every window-completing step."""
    torch.manual_seed(4)
    ratio, head_dim, coff = 4, 6, 1
    total_len = 5 * ratio
    kv_all = torch.randn(1, total_len, head_dim)
    gate_all = torch.randn(1, total_len, head_dim)
    bias = torch.randn(ratio, head_dim)

    history_kv = history_gate = torch.zeros(1, 0, head_dim)
    cached_seq_len = 0
    for pos in range(total_len):
        kv_c, gate_c = kv_all[:, pos : pos + 1], gate_all[:, pos : pos + 1]

        n = carry_gather_length(cached_seq_len, ratio, needs_overlap=False)
        drop = carry_replay_already_emitted(cached_seq_len, ratio, needs_overlap=False)
        old_kv = torch.cat((history_kv[:, -n:] if n else history_kv[:, :0], kv_c), dim=1)
        old_gate = torch.cat((history_gate[:, -n:] if n else history_gate[:, :0], gate_c), dim=1)
        out_old, _ = compress_hca_chunk(old_kv, old_gate, bias, None)
        out_old = out_old[:, drop:]

        carry_kv, carry_gate, valid = _fixed_window_carry(
            history_kv, history_gate, cached_seq_len, ratio, coff, needs_overlap=False
        )
        new_kv = torch.cat((carry_kv, kv_c), dim=1)
        new_gate = torch.cat((carry_gate, gate_c), dim=1)
        out_new, _ = compress_hca_chunk(new_kv, new_gate, bias, None, carry_valid=valid)
        entry_new = out_new[:, -1:]

        assert torch.isfinite(entry_new).all()
        if out_old.shape[1] == 1:
            torch.testing.assert_close(entry_new, out_old, rtol=0, atol=1e-5)

        history_kv = torch.cat((history_kv, kv_c), dim=1)
        history_gate = torch.cat((history_gate, gate_c), dim=1)
        cached_seq_len += 1


def test_compress_csa_chunk_carry_valid_masking_matches_incremental_slicing():
    """Same as the HCA version, for CSA's overlap reconstruction --
    specifically covers the first-ever completed window (an all-masked
    "previous window" block 0, verifying softmax-of-all--inf there never
    contaminates the block actually read) and the second (the first real
    overlap block)."""
    torch.manual_seed(5)
    ratio, head_dim, coff = 4, 5, 2
    double_width = 2 * head_dim
    total_len = 6 * ratio
    kv_all = torch.randn(1, total_len, double_width)
    gate_all = torch.randn(1, total_len, double_width)
    bias = torch.randn(ratio, double_width)

    history_kv = history_gate = torch.zeros(1, 0, double_width)
    cached_seq_len = 0
    completions_checked = 0
    for pos in range(total_len):
        kv_c, gate_c = kv_all[:, pos : pos + 1], gate_all[:, pos : pos + 1]

        n = carry_gather_length(cached_seq_len, ratio, needs_overlap=True)
        drop = carry_replay_already_emitted(cached_seq_len, ratio, needs_overlap=True)
        old_kv = torch.cat((history_kv[:, -n:] if n else history_kv[:, :0], kv_c), dim=1)
        old_gate = torch.cat((history_gate[:, -n:] if n else history_gate[:, :0], gate_c), dim=1)
        out_old, _ = compress_csa_chunk(old_kv, old_gate, bias, None)
        out_old = out_old[:, drop:]

        carry_kv, carry_gate, valid = _fixed_window_carry(
            history_kv, history_gate, cached_seq_len, ratio, coff, needs_overlap=True
        )
        new_kv = torch.cat((carry_kv, kv_c), dim=1)
        new_gate = torch.cat((carry_gate, gate_c), dim=1)
        out_new, _ = compress_csa_chunk(new_kv, new_gate, bias, None, carry_valid=valid)
        entry_new = out_new[:, -1:]

        assert torch.isfinite(entry_new).all()
        if out_old.shape[1] == 1:
            torch.testing.assert_close(entry_new, out_old, rtol=0, atol=1e-5)
            completions_checked += 1

        history_kv = torch.cat((history_kv, kv_c), dim=1)
        history_gate = torch.cat((history_gate, gate_c), dim=1)
        cached_seq_len += 1

    # Sanity: this walk actually exercised >1 completion (both the
    # all-masked-block-0 first completion and later, real-overlap ones), not
    # just the trivially-easy first case.
    assert completions_checked >= 2


def test_compress_hca_chunk_and_csa_chunk_carry_valid_default_none_unchanged():
    """carry_valid=None (the default) must reproduce pre-change output
    exactly -- the new parameter is opt-in, existing callers (the oracle
    tests in test_deepseek_v4_component_oracles.py, this file's own
    incremental-chunking cross-checks) are untouched."""
    torch.manual_seed(6)
    ratio, head_dim = 4, 6
    kv = torch.randn(1, 11, head_dim)
    gate = torch.randn(1, 11, head_dim)
    bias = torch.randn(ratio, head_dim)
    out_implicit, state_implicit = compress_hca_chunk(kv, gate, bias, None)
    out_explicit, state_explicit = compress_hca_chunk(kv, gate, bias, None, carry_valid=None)
    torch.testing.assert_close(out_implicit, out_explicit, rtol=0, atol=0)
    torch.testing.assert_close(state_implicit.kv_carry, state_explicit.kv_carry, rtol=0, atol=0)

    double_width = 2 * head_dim
    kv2 = torch.randn(1, 13, double_width)
    gate2 = torch.randn(1, 13, double_width)
    bias2 = torch.randn(ratio, double_width)
    out_implicit2, _ = compress_csa_chunk(kv2, gate2, bias2, None)
    out_explicit2, _ = compress_csa_chunk(kv2, gate2, bias2, None, carry_valid=None)
    torch.testing.assert_close(out_implicit2, out_explicit2, rtol=0, atol=0)


def test_carry_rows_reads_live_window_correctly_past_carry_cache_eviction():
    """Regression test closing the coverage gap
    docs/model-dev/deepseek-v4-swa-null-block-bug.md flags for _carry_rows:
    unlike _swa_history (covered by
    test_attention_matches_real_module_after_swa_eviction_past_one_window in
    test_deepseek_v4_matches_real_architecture.py), no existing test drove
    _carry_rows itself through a real paged state_cache past one carry
    window's worth of eviction. Mirrors
    test_gather_paged_latent_reads_stale_columns_once_swa_window_has_advanced's
    null-block simulation, applied to DeepseekV4Compressor._carry_rows.
    """
    from vllm_neuron.model.deepseek_v4.model import DeepseekV4Compressor

    torch.manual_seed(7)
    ratio, head_dim, needs_overlap = 4, 3, True
    coff = 2
    width = coff * head_dim
    cache_window = coff * ratio  # state_cache's own physical sliding window
    block_size = 2
    cached_seq_len = 21  # well past cache_window (8) and past one evicted block

    comp = DeepseekV4Compressor(
        hidden_size=8,
        head_dim=head_dim,
        ratio=ratio,
        rms_norm_eps=1e-6,
        rotary_emb=None,
        qk_rope_head_dim=head_dim,
    )
    assert comp.coff == coff and comp.width == width and comp.overlap == needs_overlap

    history_kv = torch.randn(1, cached_seq_len, width)
    history_gate = torch.randn(1, cached_seq_len, width)
    history = torch.cat((history_kv, history_gate), dim=-1).squeeze(0)

    live_start = max(0, cached_seq_len - cache_window)  # = 13
    n_cols = -(-cached_seq_len // block_size)
    cache = torch.zeros(n_cols + 1, 1, block_size, 2 * width)
    cache[0] = -999.0  # null-block sentinel -- a valid read must never see this
    block_table_row = []
    for col in range(n_cols):
        col_start = col * block_size
        if col_start + block_size <= live_start:
            block_table_row.append(0)
            continue
        physical = col + 1
        block_table_row.append(physical)
        for slot in range(block_size):
            token = col_start + slot
            if token < cached_seq_len:
                cache[physical, 0, slot] = history[token]
    block_table = torch.tensor(block_table_row, dtype=torch.long)

    comp.state_cache = cache
    position_ids = torch.tensor([[cached_seq_len]], dtype=torch.long)
    carry_kv, carry_gate, carry_valid = comp._carry_rows(block_table, position_ids)

    gather_n = carry_gather_length(cached_seq_len, ratio, needs_overlap=needs_overlap)
    assert gather_n == 5
    expected_kv = history_kv[:, -gather_n:]
    expected_gate = history_gate[:, -gather_n:]
    valid_carry = carry_valid[:-1]  # drop the trailing (always-True) new-token slot
    assert int(valid_carry.sum()) == gather_n
    torch.testing.assert_close(carry_kv[:, valid_carry], expected_kv, rtol=0, atol=0)
    torch.testing.assert_close(carry_gate[:, valid_carry], expected_gate, rtol=0, atol=0)
    assert not bool((carry_kv[:, valid_carry] == -999.0).any())
    assert not bool((carry_gate[:, valid_carry] == -999.0).any())


def test_carry_replay_reproduces_incremental_csa_chunking_with_overlap():
    torch.manual_seed(1)
    ratio, head_dim = 4, 5
    double_width = 2 * head_dim
    total_len = 25
    kv_all = torch.randn(1, total_len, double_width)
    gate_all = torch.randn(1, total_len, double_width)
    bias = torch.randn(ratio, double_width)

    for chunks in ([9, 1, 4, 15, 4], [4, 4, 4, 4, 4, 4], [1] * 25, [4], [4, 4]):
        state = None
        ref_outputs = []
        pos = 0
        for size in chunks:
            if pos + size > total_len:
                continue
            out, state = compress_csa_chunk(
                kv_all[:, pos : pos + size], gate_all[:, pos : pos + size], bias, state
            )
            ref_outputs.append(out)
            pos += size
        ref = torch.cat(ref_outputs, dim=1)

        history_kv = history_gate = torch.zeros(1, 0, double_width)
        dev_outputs = []
        cached_seq_len = 0
        pos = 0
        for size in chunks:
            if pos + size > total_len:
                continue
            kv_c = kv_all[:, pos : pos + size]
            gate_c = gate_all[:, pos : pos + size]
            n = carry_gather_length(cached_seq_len, ratio, needs_overlap=True)
            drop = carry_replay_already_emitted(
                cached_seq_len, ratio, needs_overlap=True
            )
            replay_kv = torch.cat((history_kv[:, -n:] if n else history_kv[:, :0], kv_c), dim=1)
            replay_gate = torch.cat(
                (history_gate[:, -n:] if n else history_gate[:, :0], gate_c), dim=1
            )
            out, _ = compress_csa_chunk(replay_kv, replay_gate, bias, None)
            dev_outputs.append(out[:, drop:])
            history_kv = torch.cat((history_kv, kv_c), dim=1)
            history_gate = torch.cat((history_gate, gate_c), dim=1)
            cached_seq_len += size
            pos += size
        dev = torch.cat(dev_outputs, dim=1)
        torch.testing.assert_close(dev, ref, rtol=0, atol=1e-5)
