#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-compile a bounded DeepSeek-V4 paged MLA component."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import torch

from vllm_neuron.model.deepseek_v4.attention import SharedLatentMLAInputs
from vllm_neuron.model.deepseek_v4.nki_mla import paged_shared_latent_mla


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
            return int("tensor_id" in value) + sum(count(item) for item in value.values())
        if isinstance(value, list):
            return sum(count(item) for item in value)
        return 0

    return count(document)


class RuntimeMLA(torch.nn.Module):
    def __init__(self, compressed: int):
        super().__init__()
        self.compressed = compressed

    def forward(
        self,
        query,
        sliding_cache,
        sliding_slots,
        sliding_valid,
        compressed_cache,
        compressed_slots,
        compressed_valid,
        sinks,
    ):
        return paged_shared_latent_mla(
            SharedLatentMLAInputs(
                query=query,
                sliding_cache=sliding_cache,
                sliding_slots=sliding_slots,
                sliding_valid=sliding_valid,
                compressed_cache=compressed_cache if self.compressed else None,
                compressed_slots=compressed_slots if self.compressed else None,
                compressed_valid=compressed_valid if self.compressed else None,
                sinks=sinks,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=int,
        choices=(1, 2, 4, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
        default=512,
    )
    parser.add_argument("--compressed", type=int, choices=(0, 512, 1024), default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("neuron:0")
    query = torch.zeros(args.query, 1, 64, 512, dtype=torch.bfloat16, device=device)
    sliding_cache = torch.zeros(1, 1, 128, 512, dtype=torch.bfloat16, device=device)
    sliding_slots = torch.zeros(args.query, 128, dtype=torch.int32, device=device)
    sliding_valid = torch.zeros(args.query, 128, dtype=torch.bool, device=device)
    compressed_cache = torch.zeros(
        1, 1, max(args.compressed, 1), 512, dtype=torch.bfloat16, device=device
    )
    compressed_slots = torch.zeros(
        args.query, max(args.compressed, 1), dtype=torch.int32, device=device
    )
    compressed_valid = torch.zeros(
        args.query, max(args.compressed, 1), dtype=torch.bool, device=device
    )
    sinks = torch.zeros(64, dtype=torch.float32, device=device)

    artifacts_before = _nki_artifacts()
    compiled = torch.compile(
        RuntimeMLA(args.compressed), backend="neuron", dynamic=False
    )
    started = time.monotonic()
    output = compiled(
        query,
        sliding_cache,
        sliding_slots,
        sliding_valid,
        compressed_cache,
        compressed_slots,
        compressed_valid,
        sinks,
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
        "query": args.query,
        "compressed_history": args.compressed,
        "sliding_history": 128,
        "wall_seconds": elapsed,
        "peak_rss_kbytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output_shape": list(output.shape),
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
