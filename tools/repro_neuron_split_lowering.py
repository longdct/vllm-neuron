#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone reproducer: `Tensor.split` with a list of sizes mis-lowers on Neuron.

`x.split([4, 4, 16], dim=-1)` returns **wrong data** on Neuron -- silently, with
no error, for every chunk and every row after the first. The same call is
correct on CPU, and the captured FX graph is correct (`split` -> `getitem`), so
the defect is below FX in the torch-xla / neuronx-cc lowering.

Mechanism: the lowering computes the right *starting offset* for each chunk and
then reads a **contiguous run** of `rows * width` elements from there, as if the
source were flat 1-D. It ignores the source's row stride. Row 0 therefore comes
out correct (its offset is right and the first `width` elements from it are the
right ones) and every later row is filled with whatever sits next in memory --
real, valid, adjacent data, which is why nothing downstream ever complains.

This script predicts the wrong output from that model and checks the prediction,
so the diagnosis is verified rather than asserted.

Depends only on torch and `vllm_neuron` (imported solely to register the Dynamo
backends).

Usage::

    # on Neuron hardware -- reproduces the bug
    PATH="$PWD/.venv-neuron/bin:$PATH" \
    NEURON_VISIBLE_DEVICES=0 NEURON_RT_VISIBLE_CORES=0 \
    NEURON_SKIP_EFA_AFFINITY=1 VLLM_CACHE_ROOT=/tmp/split-repro \
    python tools/repro_neuron_split_lowering.py

    # control: the same script on CPU, where everything passes
    VLLM_NEURON_CPU_MODE=1 python tools/repro_neuron_split_lowering.py

Both `NEURON_VISIBLE_DEVICES` and `NEURON_RT_VISIBLE_CORES` must be set; the
runtime rejects one without the other.

Exits 0 if the backend is correct, 1 if the defect is present.

Observed on: trn2.3xlarge, neuronx-cc 2.27.5334.0, torch/torch-xla 2.11,
libtorch-neuronx-lite 2.11.
"""

from __future__ import annotations

import torch

import vllm_neuron  # noqa: F401  -- registers the Dynamo backends
from vllm_neuron import envs
from vllm_neuron.envs import get_compile_backend_name

ROWS, COLS = 2, 24
SIZES = [4, 4, 16]


def device_name() -> str:
    return "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


def run(module: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """Compile `module` for the target device and return its outputs on CPU."""
    compiled = torch.compile(module.to(device_name()), backend=get_compile_backend_name())
    with torch.no_grad():
        return [t.to("cpu") for t in compiled(x.to(device_name()))]


class Split(torch.nn.Module):
    def __init__(self, sizes, dim, op="split"):
        super().__init__()
        self.sizes, self.dim, self.op = sizes, dim, op

    def forward(self, x):
        if self.op == "chunk":
            return list(x.chunk(self.sizes, dim=self.dim))
        return list(x.split(self.sizes, dim=self.dim))


def contiguous_read_prediction(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """What each chunk becomes if the source is read as flat, stride-ignoring memory."""
    flat, predicted, offset = x.reshape(-1), [], 0
    for width in sizes:
        count = x.shape[0] * width
        predicted.append(flat[offset : offset + count].reshape(x.shape[0], width))
        offset += width
    return predicted


def demonstrate() -> bool:
    """Show the wrong values, and confirm they match the stride-ignoring model."""
    x = torch.arange(ROWS * COLS, dtype=torch.float32).reshape(ROWS, COLS)
    expected = list(x.split(SIZES, dim=-1))          # eager CPU: the ground truth
    actual = run(Split(SIZES, -1), x)
    predicted = contiguous_read_prediction(x, SIZES)

    print(f"input: arange({ROWS * COLS}).reshape({ROWS}, {COLS}), "
          f"split({SIZES}, dim=-1) on {device_name()}\n")
    correct, matches_model = True, True
    start = 0
    for i, (want, got, guess) in enumerate(zip(expected, actual, predicted)):
        print(f"chunk {i}  (columns {start}..{start + SIZES[i] - 1})")
        start += SIZES[i]
        for row in range(ROWS):
            w = [int(v) for v in want[row].tolist()]
            g = [int(v) for v in got[row].tolist()]
            flag = "" if w == g else "   <-- WRONG"
            print(f"   row {row}  expected {w}")
            print(f"          got      {g}{flag}")
        correct &= torch.equal(want, got)
        matches_model &= torch.equal(got, guess)
        print()

    if not correct:
        print("The wrong values are not arbitrary. Predicting them by reading the "
              "source\nas flat memory and ignoring its row stride reproduces the "
              "device output\n"
              f"exactly: {matches_model}\n")
    return correct


def survey() -> bool:
    """Which spelling of the same operation is safe?"""
    x = torch.arange(8 * 24, dtype=torch.float32).reshape(8, 24)
    cases = (
        ("split([4,4,16], dim=-1)", [4, 4, 16], -1, "split"),
        ("split([8,8,8],  dim=-1)", [8, 8, 8], -1, "split"),
        ("split([4,20],   dim=1)", [4, 20], 1, "split"),
        ("split(8,        dim=-1)", 8, -1, "split"),
        ("chunk(3,        dim=-1)", 3, -1, "chunk"),
        ("split([2,6],    dim=0)", [2, 6], 0, "split"),
    )
    print(f"which forms are affected, on {device_name()}:\n")
    ok = True
    for name, sizes, dim, op in cases:
        module = Split(sizes, dim, op)
        with torch.no_grad():
            expected = module(x)
        actual = run(Split(sizes, dim, op), x)
        wrong = [i for i, (a, b) in enumerate(zip(expected, actual))
                 if not torch.equal(a, b)]
        ok &= not wrong
        print(f"   {name:<26} {'wrong chunks: ' + str(wrong) if wrong else 'correct'}")
    print()
    return ok


def main() -> None:
    ok = demonstrate()
    ok &= survey()
    if ok:
        print("PASS -- this backend lowers Tensor.split correctly.")
    else:
        print("FAIL -- Tensor.split with a list of sizes is mis-lowered.\n"
              "Workaround: slice explicitly (`x[..., 4:8]`), or use narrow /\n"
              "index_select / chunk / an int split size, all of which are correct.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
