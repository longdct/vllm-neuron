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
    entry_slot = compressed_entry_slot_mapping(raw_slot, ratio)
    expected_valid = (positions + 1) % ratio == 0
    assert torch.equal(entry_slot >= 0, expected_valid)
    assert torch.equal(entry_slot[expected_valid], positions[expected_valid] // ratio)
    del block_table_stub


def test_compressed_entry_slot_mapping_treats_padding_as_invalid():
    raw_slot = torch.tensor([-1, 3, 6, -1])
    entry_slot = compressed_entry_slot_mapping(raw_slot, 4)
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
