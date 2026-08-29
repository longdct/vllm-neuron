#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-compile and run a CSA indexer-to-paged-MLA dependency chain.

This deliberately omits model projections, cache updates, collectives, and the
vLLM engine.  It is a device reproduction for the two opaque NKI calls whose
standalone probes both complete but whose composed model path times out.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from vllm_neuron.model.deepseek_v4.attention import (
    SharedLatentMLAInputs,
    logical_to_physical_slots_batched,
)
from vllm_neuron.model.deepseek_v4.nki_indexer import (
    paged_projected_bf16_indexer,
)
from vllm_neuron.model.deepseek_v4.nki_mla import paged_shared_latent_mla


class RuntimeIndexerMLA(torch.nn.Module):
    def __init__(self, logical_slots_per_block: int):
        super().__init__()
        self.logical_slots_per_block = logical_slots_per_block

    def forward(
        self,
        index_query,
        index_gate,
        index_cache,
        index_block_table,
        visible,
        mla_query,
        sliding_cache,
        sliding_slots,
        sliding_valid,
        compressed_cache,
        compressed_block_table,
        owners,
        sinks,
    ):
        selected = paged_projected_bf16_indexer(
            index_query,
            index_gate,
            index_cache,
            index_block_table,
            visible,
            logical_slots_per_block=self.logical_slots_per_block,
        )
        compressed_slots, compressed_valid = logical_to_physical_slots_batched(
            selected.logical_indices,
            selected.valid,
            compressed_block_table,
            owners,
            logical_slots_per_block=self.logical_slots_per_block,
            physical_page_stride=compressed_cache.shape[2],
            cache_blocks=compressed_cache.shape[0],
        )
        return paged_shared_latent_mla(
            SharedLatentMLAInputs(
                query=mla_query,
                sliding_cache=sliding_cache,
                sliding_slots=sliding_slots,
                sliding_valid=sliding_valid,
                compressed_cache=compressed_cache,
                compressed_slots=compressed_slots,
                compressed_valid=compressed_valid,
                sinks=sinks,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=int,
        choices=(8, 64, 512, 1024, 2048, 4096, 8192),
        default=8,
    )
    parser.add_argument("--logical-slots", type=int, default=32)
    parser.add_argument("--block-columns", type=int, default=2)
    parser.add_argument("--visible", type=int, default=16)
    parser.add_argument(
        "--prefill-visibility",
        action="store_true",
        help="use the CSA position-0..Q-1 visibility ramp instead of --visible",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("neuron:0")
    q = args.query
    logical_capacity = args.block_columns * args.logical_slots
    physical_stride = max(128, args.logical_slots)

    index_query = torch.zeros(q, 1, 64, 128, dtype=torch.bfloat16, device=device)
    index_gate = torch.zeros(q, 1, 64, dtype=torch.bfloat16, device=device)
    index_cache = torch.zeros(
        1, 1, physical_stride, 128, dtype=torch.bfloat16, device=device
    )
    index_block_table = torch.zeros(
        args.block_columns, dtype=torch.int32, device=device
    )
    if args.prefill_visibility:
        visible = torch.div(
            torch.arange(1, q + 1, dtype=torch.int32, device=device),
            4,
            rounding_mode="floor",
        )
        visible_record: int | list[int] = visible.to("cpu").tolist()
    else:
        visible = torch.full((q,), args.visible, dtype=torch.int32, device=device)
        visible_record = args.visible

    mla_query = torch.zeros(q, 1, 64, 512, dtype=torch.bfloat16, device=device)
    sliding_cache = torch.zeros(1, 1, 128, 512, dtype=torch.bfloat16, device=device)
    sliding_slots = torch.zeros(q, 128, dtype=torch.int32, device=device)
    sliding_valid = torch.zeros(q, 128, dtype=torch.bool, device=device)
    compressed_cache = torch.zeros(
        1, 1, max(512, logical_capacity), 512, dtype=torch.bfloat16, device=device
    )
    compressed_block_table = torch.zeros(
        1, args.block_columns, dtype=torch.int32, device=device
    )
    owners = torch.zeros(q, dtype=torch.int32, device=device)
    sinks = torch.zeros(64, dtype=torch.float32, device=device)

    compiled = torch.compile(
        RuntimeIndexerMLA(args.logical_slots), backend="neuron", dynamic=False
    )
    started = time.monotonic()
    output = compiled(
        index_query,
        index_gate,
        index_cache,
        index_block_table,
        visible,
        mla_query,
        sliding_cache,
        sliding_slots,
        sliding_valid,
        compressed_cache,
        compressed_block_table,
        owners,
        sinks,
    )
    torch.neuron.synchronize()
    elapsed = time.monotonic() - started
    record = {
        "query": q,
        "logical_slots_per_block": args.logical_slots,
        "logical_capacity": logical_capacity,
        "visible_entries": visible_record,
        "wall_seconds": elapsed,
        "peak_rss_kbytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output_shape": list(output.shape),
    }
    text = json.dumps(record, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
