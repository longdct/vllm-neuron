#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the runtime-query paged MLA kernel with its CPU oracle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vllm_neuron.model.deepseek_v4.attention import (
    SharedLatentMLAInputs,
    gather_bounded_paged_latent,
    shared_latent_attention,
)
from vllm_neuron.model.deepseek_v4.nki_mla import paged_shared_latent_mla


class RuntimeMLAOracle(torch.nn.Module):
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
                compressed_cache=compressed_cache,
                compressed_slots=compressed_slots,
                compressed_valid=compressed_valid,
                sinks=sinks,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=int, choices=(64, 128), default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(20260828)
    q_count = args.query
    query = torch.randn(
        q_count, 1, 1, 512, generator=generator, dtype=torch.bfloat16
    )
    sliding_cache = torch.randn(
        2, 1, 128, 512, generator=generator, dtype=torch.bfloat16
    )
    compressed_cache = torch.randn(
        2, 1, 512, 512, generator=generator, dtype=torch.bfloat16
    )
    rows = torch.arange(q_count, dtype=torch.int32)[:, None]
    sliding_slots = torch.arange(128, dtype=torch.int32)[None, :] + rows
    compressed_slots = torch.arange(512, dtype=torch.int32)[None, :] + rows
    sliding_used = (torch.arange(q_count) * 3 + 5).clamp(max=128)
    compressed_used = (torch.arange(q_count) * 8 + 1).clamp(max=512)
    sliding_valid = torch.arange(128)[None, :] < sliding_used[:, None]
    compressed_valid = torch.arange(512)[None, :] < compressed_used[:, None]
    sinks = torch.randn(1, generator=generator, dtype=torch.bfloat16)

    compressed, compressed_gathered = gather_bounded_paged_latent(
        compressed_cache, compressed_slots, compressed_valid
    )
    sliding, sliding_gathered = gather_bounded_paged_latent(
        sliding_cache, sliding_slots, sliding_valid
    )
    expected = shared_latent_attention(
        query,
        torch.cat((compressed, sliding), dim=1),
        visibility=torch.cat((compressed_gathered, sliding_gathered), dim=1),
        attention_sinks=sinks,
    )

    device = torch.device("neuron:0")
    compiled = torch.compile(RuntimeMLAOracle(), backend="neuron", dynamic=False)
    started = time.monotonic()
    actual = compiled(
        query.to(device),
        sliding_cache.to(device),
        sliding_slots.to(device),
        sliding_valid.to(device),
        compressed_cache.to(device),
        compressed_slots.to(device),
        compressed_valid.to(device),
        sinks.to(device),
    )
    torch.neuron.synchronize()
    wall_seconds = time.monotonic() - started
    actual_cpu = actual.to("cpu")
    difference = (actual_cpu.float() - expected.float()).abs()
    torch.testing.assert_close(actual_cpu, expected, rtol=0.03, atol=0.03)
    record = {
        "query": q_count,
        "heads": 1,
        "compressed_history": 512,
        "sliding_history": 128,
        "wall_seconds": wall_seconds,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "oracle_close": True,
    }
    text = json.dumps(record, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
