#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-compile the capacity-independent DeepSeek-V4 CSA indexer graph."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from vllm_neuron.model.deepseek_v4.nki_indexer import paged_projected_bf16_indexer


class RuntimeIndexer(torch.nn.Module):
    def __init__(self, logical_slots_per_block: int):
        super().__init__()
        self.logical_slots_per_block = logical_slots_per_block

    def forward(self, query, gate, key_cache, block_table, visible):
        selected = paged_projected_bf16_indexer(
            query,
            gate,
            key_cache,
            block_table,
            visible,
            logical_slots_per_block=self.logical_slots_per_block,
        )
        return selected.logical_indices, selected.valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=int,
        choices=(1, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
        default=512,
    )
    parser.add_argument("--block-columns", type=int, default=1024)
    parser.add_argument("--logical-slots", type=int, default=128)
    parser.add_argument("--visible", type=int, default=0)
    parser.add_argument(
        "--prefill-visibility",
        action="store_true",
        help="use the CSA position-0..Q-1 visibility ramp instead of --visible",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("neuron:0")
    physical_stride = max(128, args.logical_slots)
    query = torch.zeros(args.query, 1, 64, 128, dtype=torch.bfloat16, device=device)
    gate = torch.zeros(args.query, 1, 64, dtype=torch.bfloat16, device=device)
    key_cache = torch.zeros(
        1, 1, physical_stride, 128, dtype=torch.bfloat16, device=device
    )
    block_table = torch.zeros(args.block_columns, dtype=torch.int32, device=device)
    if args.prefill_visibility:
        visible = torch.div(
            torch.arange(1, args.query + 1, dtype=torch.int32, device=device),
            4,
            rounding_mode="floor",
        )
        visible_cpu = visible.to("cpu").long()
        visible_record: int | list[int] | dict[str, int]
        if args.query <= 1024:
            visible_record = visible_cpu.tolist()
        else:
            visible_record = {
                "count": args.query,
                "first": int(visible_cpu[0]),
                "last": int(visible_cpu[-1]),
            }
    else:
        visible = torch.full(
            (args.query,), args.visible, dtype=torch.int32, device=device
        )
        visible_cpu = visible.to("cpu").long()
        visible_record = args.visible

    compiled = torch.compile(
        RuntimeIndexer(args.logical_slots), backend="neuron", dynamic=False
    )
    started = time.monotonic()
    indices, valid = compiled(query, gate, key_cache, block_table, visible)
    torch.neuron.synchronize()
    elapsed = time.monotonic() - started
    indices_cpu = indices.to("cpu")
    valid_cpu = valid.to("cpu")
    expected_visible = visible_cpu
    expected_used = expected_visible.clamp(
        min=0,
        max=min(args.block_columns * args.logical_slots, 512),
    )
    valid_counts = valid_cpu.sum(dim=1)
    valid_count_ok = bool((valid_counts == expected_used).all())
    mismatched_rows = torch.nonzero(valid_counts != expected_used).flatten()
    valid_range_ok = bool(
        (
            ~valid_cpu
            | (
                (indices_cpu >= 0)
                & (indices_cpu < expected_visible[:, None])
            )
        ).all()
    )
    record = {
        "query": args.query,
        "block_columns": args.block_columns,
        "logical_slots_per_block": args.logical_slots,
        "capacity_entries": args.block_columns * args.logical_slots,
        "visible_entries": visible_record,
        "wall_seconds": elapsed,
        "peak_rss_kbytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output_shapes": [list(indices.shape), list(valid.shape)],
        "valid_count_ok": valid_count_ok,
        "valid_range_ok": valid_range_ok,
        "valid_counts_summary": {
            "min": int(valid_counts.min()),
            "max": int(valid_counts.max()),
            "mismatch_count": int(mismatched_rows.numel()),
            "first_mismatch": (
                int(mismatched_rows[0]) if mismatched_rows.numel() else None
            ),
        },
    }
    text = json.dumps(record, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")
    if not valid_count_ok or not valid_range_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
