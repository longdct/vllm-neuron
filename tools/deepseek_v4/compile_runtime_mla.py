#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-compile a bounded DeepSeek-V4 paged MLA component."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from vllm_neuron.model.deepseek_v4.attention import SharedLatentMLAInputs
from vllm_neuron.model.deepseek_v4.nki_mla import paged_shared_latent_mla


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
    parser.add_argument("--query", type=int, choices=(512, 1024), default=512)
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
    record = {
        "query": args.query,
        "compressed_history": args.compressed,
        "sliding_history": 128,
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
