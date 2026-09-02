#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the compiled on-device sampler actually return the token it chose?

On-device sampling on DeepSeek-V4 returns token ids far outside the vocabulary
(`967439869`, `-1167771948`). Read as raw int32 those are float32 bit patterns
for `+3.24e-4` and `-8.74e-4` -- inside the `(-1e-3, 1e-3)` interval vLLM's
dummy-weight initializer draws from. That is not a miscomputed index; it is a
buffer nobody wrote, read through an int32 view.

The full model cannot tell apart the three ways that can happen: a wrong
sampler, an output the NEFF never writes, or an async readback of recycled
memory. This probe removes the model. It compiles *only* the sampler over
logits with a known unique maximum, so a mismatch here convicts the sampler
itself and a match sends the investigation to the graph's output plumbing.

The mismatch report bit-casts whatever came back to float32, because that
reinterpretation is what identified the defect in the first place and is worth
having automatically rather than by hand.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch

from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig
from vllm_neuron.nn.sampler import Sampler


def build_logits(batch: int, vocab: int, seed: int) -> torch.Tensor:
    """Logits whose per-row argmax is unique and known.

    A unique maximum matters: tie-breaking between the device reduce and
    ``torch.argmax`` is a legitimate difference, and would muddy the signal
    this probe exists to produce.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = torch.rand((batch, vocab), generator=generator) * 2.0 - 1.0
    winners = torch.randint(0, vocab, (batch,), generator=generator)
    logits[torch.arange(batch), winners] = 10.0
    return logits, winners


def as_float_bits(token: int) -> float | None:
    """Reinterpret an int32 token id as float32, or None if it does not fit."""
    try:
        return struct.unpack("<f", struct.pack("<i", int(token)))[0]
    except (struct.error, OverflowError, ValueError):
        return None


class RuntimeSampler(torch.nn.Module):
    def __init__(self, sampler: Sampler) -> None:
        super().__init__()
        self.sampler = sampler

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.sampler(logits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=129280)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--greedy",
        choices=("all-greedy", "generic"),
        default="all-greedy",
        help=(
            "'all-greedy' takes sampling.py's argmax fast path; 'generic' goes "
            "through top-k/top-p with a zero temperature, which is the path a "
            "real request with temperature=0 takes."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    logits, winners = build_logits(args.batch, args.vocab, args.seed)
    expected = torch.argmax(logits, dim=-1)
    assert torch.equal(expected, winners), "probe construction is wrong"

    sampler = Sampler(
        OnDeviceSamplingConfig(
            all_greedy=args.greedy == "all-greedy",
            max_top_k=256,
            deterministic=True,
        ),
        process_group=None,
        vocab_size=args.vocab,
    )
    cpu_tokens = RuntimeSampler(sampler)(logits).tolist()

    device = torch.device("neuron:0")
    compiled = torch.compile(
        RuntimeSampler(sampler).to(device), backend="neuron", dynamic=False
    )
    device_tokens = compiled(logits.to(device))
    torch.neuron.synchronize()
    device_tokens = device_tokens.to("cpu").tolist()

    matches = device_tokens == expected.tolist()
    report = {
        "batch": args.batch,
        "vocab": args.vocab,
        "greedy_path": args.greedy,
        "expected_tokens": expected.tolist(),
        "cpu_sampler_tokens": cpu_tokens,
        "device_sampler_tokens": device_tokens,
        "device_matches_expected": matches,
        "cpu_matches_expected": cpu_tokens == expected.tolist(),
    }
    if not matches:
        report["device_tokens_as_float32"] = [
            as_float_bits(token) for token in device_tokens
        ]
        report["in_vocab_range"] = [
            0 <= token < args.vocab for token in device_tokens
        ]

    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    raise SystemExit(0 if matches else 1)


if __name__ == "__main__":
    main()
