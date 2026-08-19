#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduces the SWA/carry null-block windowing bug (see
docs/model-dev/deepseek-v4-swa-null-block-bug.md).

No vLLM/device dependency -- exercises ``gather_paged_latent`` directly
against a hand-built cache + block table that simulates exactly what real
vLLM's ``SlidingWindowManager`` produces once a sequence has run past one
sliding window: the block-table row never compacts, it only remaps columns
that have fallen entirely out of the window to a shared null block (id 0,
matching this plugin's own
``test_sliding_window_remapping_uses_null_blocks_but_latents_remain_stable``).
Old, low-index columns go null while the live window's real data sits at
ever-increasing column indices -- so a fixed "read the first `required`
columns" gather (what ``_swa_history``/``_carry_rows`` currently do) returns
null-block content instead of history, once the window has advanced.

Exit code is 0 iff the read matches the true window (i.e., iff the bug has
been fixed). Run with no arguments.
"""

from __future__ import annotations

import sys

import torch

from vllm_neuron.model.deepseek_v4.attention import gather_paged_latent

BLOCK_SIZE = 2  # slots per physical block
SLIDING_WINDOW = 4
CACHED_SEQ_LEN = 10  # already-computed tokens; well past one window
LATENT_DIM = 1
NUM_LOGICAL_BLOCKS = 8  # enough columns to cover CACHED_SEQ_LEN
NULL_SENTINEL = -999.0  # recognizably-wrong value stored in the null block


def build_simulated_cache_and_block_table():
    """Mirror real vLLM: block-table columns are fixed at
    ``token_pos // block_size`` and never shift. A column is remapped to the
    null block once every token it covers has fallen outside the window
    (``SingleTypeKVCacheManager.get_num_skipped_tokens``); otherwise it holds
    whatever distinct physical block was allocated for it (never reused here,
    for clarity -- physical id reuse doesn't change the bug).
    """
    live_start_token = max(0, CACHED_SEQ_LEN - SLIDING_WINDOW)
    physical_pool: dict[int, int | None] = {}
    for logical_idx in range(NUM_LOGICAL_BLOCKS):
        block_start = logical_idx * BLOCK_SIZE
        block_end = block_start + BLOCK_SIZE
        out_of_window = block_end <= live_start_token or block_start >= CACHED_SEQ_LEN
        physical_pool[logical_idx] = None if out_of_window else logical_idx + 1

    num_physical = max((v for v in physical_pool.values() if v is not None), default=0) + 1
    cache = torch.zeros((max(num_physical, 2), 1, BLOCK_SIZE, LATENT_DIM))
    cache[0] = NULL_SENTINEL  # physical block 0 == the shared null block

    for logical_idx, physical_id in physical_pool.items():
        if physical_id is None:
            continue
        for slot in range(BLOCK_SIZE):
            token = logical_idx * BLOCK_SIZE + slot
            if 0 <= token < CACHED_SEQ_LEN:
                cache[physical_id, 0, slot, 0] = float(token)

    block_table_row = torch.tensor(
        [physical_pool[i] or 0 for i in range(NUM_LOGICAL_BLOCKS)], dtype=torch.int64
    )
    return cache, block_table_row


def main() -> int:
    cache, block_table_row = build_simulated_cache_and_block_table()
    gather_len = min(CACHED_SEQ_LEN, SLIDING_WINDOW)
    live_start_token = max(0, CACHED_SEQ_LEN - SLIDING_WINDOW)
    result = gather_paged_latent(
        cache, block_table_row, gather_len, start_token=live_start_token
    ).squeeze(1).squeeze(-1)
    expected = [float(t) for t in range(CACHED_SEQ_LEN - gather_len, CACHED_SEQ_LEN)]

    print(f"cached_seq_len={CACHED_SEQ_LEN} sliding_window={SLIDING_WINDOW}")
    print("block_table_row:", block_table_row.tolist())
    print("gathered values:", result.tolist())
    print("expected values:", expected)

    if result.tolist() == expected:
        print("PASS -- gather_paged_latent reads the current window correctly.")
        return 0
    print(
        "FAIL -- gather_paged_latent returned null-block content instead of the "
        "true recent window. See docs/model-dev/deepseek-v4-swa-null-block-bug.md."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
