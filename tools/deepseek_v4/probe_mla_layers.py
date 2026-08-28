#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure paged-MLA cold-compile cost as a function of stacked layer count.

Each "layer" performs one ``paged_shared_latent_mla`` call, which the current
dispatcher expands into ``Q // _PREFILL_QUERY_TILE`` opaque NKI custom calls.
Chaining layers therefore scales custom-call sites while holding the compiled
kernel body fixed, isolating outer-graph cost from kernel cost.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from vllm_neuron.model.deepseek_v4.attention import SharedLatentMLAInputs
from vllm_neuron.model.deepseek_v4.nki_mla import paged_shared_latent_mla


class StackedMLA(torch.nn.Module):
    def __init__(self, layers: int, compressed: int, uniform: bool = False):
        super().__init__()
        self.layers = layers
        self.compressed = compressed
        # HCA's compressed rows repeat across a launch, so the kernel gathers
        # them once instead of once per query.  CSA's do not.  Both reach this
        # probe at the same shapes, so the flag is what separates them.
        self.uniform = uniform

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
        hidden = query
        for _ in range(self.layers):
            hidden = paged_shared_latent_mla(
                SharedLatentMLAInputs(
                    query=hidden,
                    sliding_cache=sliding_cache,
                    sliding_slots=sliding_slots,
                    sliding_valid=sliding_valid,
                    compressed_cache=compressed_cache if self.compressed else None,
                    compressed_slots=compressed_slots if self.compressed else None,
                    compressed_valid=compressed_valid if self.compressed else None,
                    sinks=sinks,
                    compressed_uniform=bool(self.compressed) and self.uniform,
                )
            )
        return hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=int, default=512)
    parser.add_argument("--compressed", type=int, default=512)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="model the HCA stream, whose compressed rows repeat per launch",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("neuron:0")
    q = args.query
    query = torch.zeros(q, 1, 64, 512, dtype=torch.bfloat16, device=device)
    sliding_cache = torch.zeros(1, 1, 128, 512, dtype=torch.bfloat16, device=device)
    sliding_slots = torch.zeros(q, 128, dtype=torch.int32, device=device)
    sliding_valid = torch.zeros(q, 128, dtype=torch.bool, device=device)
    width = max(args.compressed, 1)
    blocks = max(width // 128, 1)
    compressed_cache = torch.zeros(
        blocks, 1, 128, 512, dtype=torch.bfloat16, device=device
    )
    compressed_slots = torch.zeros(q, width, dtype=torch.int32, device=device)
    compressed_valid = torch.zeros(q, width, dtype=torch.bool, device=device)
    sinks = torch.zeros(64, dtype=torch.bfloat16, device=device)

    model = StackedMLA(args.layers, args.compressed, args.uniform)
    start = time.perf_counter()
    out = model(
        query,
        sliding_cache,
        sliding_slots,
        sliding_valid,
        compressed_cache,
        compressed_slots,
        compressed_valid,
        sinks,
    )
    out_cpu = out.to("cpu")
    wall = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    self_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "query": q,
        "compressed_history": args.compressed,
        "compressed_uniform": args.uniform,
        "layers": args.layers,
        "wall_seconds": wall,
        "peak_rss_children_kbytes": peak,
        "peak_rss_self_kbytes": self_peak,
        "output_shape": list(out_cpu.shape),
    }
    print(json.dumps(record, indent=2))
    if args.output:
        args.output.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
