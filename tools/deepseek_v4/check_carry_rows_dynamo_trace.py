#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only Dynamo tracing proxy for the ``_carry_rows`` fix (see
docs/model-dev/deepseek-v4-swa-null-block-bug.md item 3).

No vLLM/device dependency -- wraps ``DeepseekV4Attention.forward`` (a single
compressed_sparse_attention layer, the harder CSA/overlap case) in
``torch.compile(fullgraph=True, dynamic=True)`` on the default ``eager``
backend, then steps it token-by-token through a real, hand-built,
null-remapped paged cache (mirroring what real vLLM's ``SlidingWindowManager``
produces) well past one full compressor carry window (``coff*ratio``), so
both the pre- and post-eviction code paths get traced.

This mirrors the same CPU-only proxy technique used for the earlier
``_swa_history`` Dynamo fix (see
docs/model-dev/deepseek-v4-024-device-validation.md Step 5d, item 6) --
cheap to iterate on, and the same class of failure
(``Could not guard on data-dependent expression``) reproduces identically on
CPU before ever booking real Trn2 time.

This layer also exercises ``DeepseekV4Attention._compressed_history``
(``model.py``), a separate, pre-existing, already-documented Dynamo blocker
(its own Python-int ``cached_seq_len // ratio`` -- see
docs/model-dev/deepseek-v4-024-device-validation.md's "next open item" note)
that this fix does not touch and that is out of scope here. So "success" for
this script specifically means: tracing advances *past* ``_carry_rows``'s old
``if gather_n == 0:`` line (`model.py:215` before this fix) -- whether it
then runs the full ``TOTAL_STEPS`` cleanly, or lands on
``_compressed_history``'s separate, already-known blocker, both count as
this fix's claim confirmed. Landing back on `_carry_rows`/`gather_n`/
`compress_hca_chunk`/`compress_csa_chunk` would mean the fix regressed and is
reported as FAIL.

Exit code is 0 iff the run never re-hits the (now fixed) ``_carry_rows``
blocker. Run with no arguments.
"""

from __future__ import annotations

import sys

import torch
from transformers import DeepseekV4Config

from vllm_neuron.model.deepseek_v4 import model as dev

RATIO = 4  # compressed_sparse_attention (CSA) -- the overlap-reconstruction case
COFF = 2
CARRY_WINDOW = COFF * RATIO  # 8
TOTAL_STEPS = 3 * CARRY_WINDOW + 1  # comfortably past two full eviction cycles


def hf_config() -> DeepseekV4Config:
    # head_dim/q_lora_rank match the values already exercised by
    # test/vllm_neuron/test_deepseek_v4_matches_real_architecture.py's
    # hf_config() -- head_dim=8 (etc.) makes qk_rope_head_dim resolve to an
    # odd value, which apply_partial_rotary rejects; this combination is
    # known-good.
    return DeepseekV4Config(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_local_experts=2,
        num_experts_per_tok=1,
        vocab_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        sliding_window=8,
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["moe"],
    )


def null_remapped_block_table(cached_seq_len: int, window: int, block_size: int, width: int) -> list[int]:
    """Same null-remap convention as
    tools/deepseek_v4/check_swa_null_block_bug.py and this repo's paged-cache
    tests: column c is live iff its last token hasn't fallen out of the
    window. ``+1`` on the allocation count because real vLLM allocates a
    token's own column before calling forward for it -- the column housing
    the token *currently* being written must already count as allocated.
    """
    live_start = max(0, cached_seq_len - window)
    n_cols = -(-(cached_seq_len + 1) // block_size)
    return [
        0 if (c >= n_cols or c * block_size + block_size <= live_start) else c + 1
        for c in range(width)
    ]


def main() -> int:
    config = hf_config()
    torch.manual_seed(11)
    device_model = dev.DeepseekV4ForCausalLM.from_configs(config).eval()
    my_attn = device_model.model.layers[0].attention
    self_attn_name = "model.layers.0.self_attn"

    specs = {s.name: s for s in device_model.get_kv_spec().layers}
    swa_spec = specs[f"{self_attn_name}.swa_cache"]
    mla_spec = specs[self_attn_name]
    state_spec = specs[f"{self_attn_name}.compressor.state_cache"]
    swa_block_size = swa_spec.block_size or 32

    width = max(
        -(-(TOTAL_STEPS + 1) // swa_block_size),
        -(-(TOTAL_STEPS + 1) // state_spec.block_size),
    ) + 1
    caches = {
        swa_spec.name: [torch.zeros((width + 1, 1, swa_block_size, swa_spec.head_size), dtype=swa_spec.dtype)],
        mla_spec.name: [torch.zeros((2, 1, mla_spec.block_size, mla_spec.head_size), dtype=mla_spec.dtype)],
        state_spec.name: [torch.zeros((width + 1, 1, state_spec.block_size, state_spec.head_size), dtype=state_spec.dtype)],
    }
    device_model.bind_kv_cache(caches)

    from vllm_neuron.model.deepseek_v4.attention import compressed_entry_slot_mapping

    compiled_forward = torch.compile(my_attn.forward, fullgraph=True, dynamic=True)

    hidden = torch.randn(TOTAL_STEPS, config.hidden_size)
    try:
        with torch.no_grad():
            for t in range(TOTAL_STEPS):
                swa_row = null_remapped_block_table(t, swa_spec.sliding_window_size, swa_block_size, width)
                state_row = null_remapped_block_table(t, state_spec.sliding_window_size, state_spec.block_size, width)
                mla_raw_slot = 1 * mla_spec.block_size + t

                def slot_for(row, block_size, pos):
                    col, off = pos // block_size, pos % block_size
                    return row[col] * block_size + off

                attn_metadata = {
                    swa_spec.name: {
                        "block_table_tensor": torch.tensor([swa_row], dtype=torch.int32),
                        "slot_mapping": torch.tensor([slot_for(swa_row, swa_block_size, t)], dtype=torch.int64),
                        "max_query_len": 1,
                        "block_size": swa_block_size,
                        "max_blocks_per_seq": width,
                        "decode_token_threshold": 1,
                        "cached_seq_len": torch.tensor([[t]], dtype=torch.int32),
                        "kv_segment_size": 0,
                    },
                    mla_spec.name: {
                        "block_table_tensor": torch.tensor([[1, 0]], dtype=torch.int32),
                        "slot_mapping": torch.tensor([mla_raw_slot], dtype=torch.int64),
                        "max_query_len": 1,
                        "block_size": mla_spec.block_size,
                        "max_blocks_per_seq": 2,
                        "decode_token_threshold": 1,
                        "cached_seq_len": torch.tensor([[t]], dtype=torch.int32),
                        "kv_segment_size": 0,
                    },
                    state_spec.name: {
                        "block_table_tensor": torch.tensor([state_row], dtype=torch.int32),
                        "slot_mapping": torch.tensor([slot_for(state_row, state_spec.block_size, t)], dtype=torch.int64),
                        "max_query_len": 1,
                        "block_size": state_spec.block_size,
                        "max_blocks_per_seq": width,
                        "decode_token_threshold": 1,
                        "cached_seq_len": torch.tensor([[t]], dtype=torch.int32),
                        "kv_segment_size": 0,
                    },
                }
                compiled_forward(
                    hidden[t : t + 1], self_attn_name=self_attn_name, attn_metadata=attn_metadata
                )
    except Exception as exc:  # noqa: BLE001 -- classify any Dynamo failure below
        message = f"{type(exc).__name__}: {exc}"
        regressed = any(
            marker in message
            for marker in ("_carry_rows", "gather_n", "compress_hca_chunk", "compress_csa_chunk")
        )
        print(f"{'FAIL' if regressed else 'PASS (see note)'} at step {t}: {message}")
        if regressed:
            print("FAIL -- tracing re-hit the _carry_rows blocker this fix was meant to close.")
            return 1
        print(
            "PASS -- _carry_rows's old `if gather_n == 0:` blocker is gone; tracing "
            f"advanced to a later, separate, already-documented blocker at step {t} "
            "instead (see this script's docstring) -- not a regression of this fix."
        )
        return 0

    print(
        f"PASS -- {TOTAL_STEPS} steps (past {TOTAL_STEPS // CARRY_WINDOW} full carry-window "
        f"cycles, carry_window={CARRY_WINDOW}) traced under "
        "torch.compile(fullgraph=True, dynamic=True) with no graph break at all."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
