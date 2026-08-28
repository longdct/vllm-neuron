#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-compile the boundary-only DeepSeek-V4 paged compressor."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import torch

from vllm_neuron.model.deepseek_v4.nki_compressor import paged_gated_compressor


def _nki_artifacts() -> dict[Path, tuple[int, int]]:
    return {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for directory in Path("/tmp").glob("nki_*")
        for path in directory.rglob("*.colz")
    }


def _allocation_records(path: Path) -> int | None:
    try:
        decoded = subprocess.run(
            ["zstd", "-dc"],
            input=path.read_bytes()[16:],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout
        document = json.loads(decoded)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None

    def count(value) -> int:
        if isinstance(value, dict):
            return int("tensor_id" in value) + sum(
                count(item) for item in value.values()
            )
        if isinstance(value, list):
            return sum(count(item) for item in value)
        return 0

    return count(document)


class RuntimeCompressor(torch.nn.Module):
    def __init__(self, ratio: int, overlap: bool):
        super().__init__()
        self.ratio = ratio
        self.overlap = overlap

    def forward(
        self,
        state_cache,
        positions,
        owners,
        block_tables,
        output_slots,
        position_bias,
    ):
        return paged_gated_compressor(
            state_cache,
            positions,
            owners,
            block_tables,
            output_slots,
            position_bias,
            ratio=self.ratio,
            overlap=self.overlap,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=int, choices=(4, 128), required=True)
    parser.add_argument("--head-dim", type=int, choices=(128, 512), required=True)
    parser.add_argument("--query", type=int, choices=(1, 512, 1024), default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ratio == 128 and args.head_dim != 512:
        parser.error("ratio 128 supports only --head-dim 512")

    overlap = args.ratio == 4
    coff = 2 if overlap else 1
    width = coff * args.head_dim
    page = 128
    start = args.ratio - 1
    positions_cpu = torch.arange(start, start + args.query, dtype=torch.long)
    columns = (int(positions_cpu[-1]) + page) // page
    device = torch.device("neuron:0")
    state_cache = torch.zeros(
        columns,
        1,
        page,
        2 * width,
        # Production CacheKind.COMPRESSOR_STATE pages are intentionally FP32.
        dtype=torch.float32,
        device=device,
    )
    positions = positions_cpu.to(device)
    owners = torch.zeros(args.query, dtype=torch.long, device=device)
    block_tables = torch.arange(columns, dtype=torch.long, device=device).reshape(1, -1)
    boundary = (positions + 1) % args.ratio == 0
    output_slots = torch.where(
        boundary,
        torch.arange(args.query, dtype=torch.long, device=device),
        torch.full((args.query,), -1, dtype=torch.long, device=device),
    )
    position_bias = torch.zeros(args.ratio, width, dtype=torch.bfloat16, device=device)

    artifacts_before = _nki_artifacts()
    compiled = torch.compile(
        RuntimeCompressor(args.ratio, overlap), backend="neuron", dynamic=False
    )
    started = time.monotonic()
    reduced, candidate_positions, selected_slots, valid = compiled(
        state_cache,
        positions,
        owners,
        block_tables,
        output_slots,
        position_bias,
    )
    torch.neuron.synchronize()
    elapsed = time.monotonic() - started
    artifacts_after = _nki_artifacts()
    generated = [
        path
        for path, metadata in artifacts_after.items()
        if artifacts_before.get(path) != metadata
    ]
    record = {
        "ratio": args.ratio,
        "head_dim": args.head_dim,
        "query": args.query,
        "overlap": overlap,
        "candidate_count": int(reduced.shape[0]),
        "raw_window_rows": coff * args.ratio,
        "wall_seconds": elapsed,
        "peak_rss_kbytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output_shape": list(reduced.shape),
        "candidate_positions_shape": list(candidate_positions.shape),
        "selected_slots_shape": list(selected_slots.shape),
        "valid_candidates": int(valid.sum().item()),
        "nki_artifacts": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "tensor_allocation_records": _allocation_records(path),
            }
            for path in sorted(generated)
        ],
    }
    text = json.dumps(record, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
