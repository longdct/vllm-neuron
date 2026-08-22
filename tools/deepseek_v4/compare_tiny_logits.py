#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare captured CPU and Neuron pre-sampling logits."""

import argparse
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("expected", type=Path)
parser.add_argument("actual", type=Path)
parser.add_argument("--tolerance", type=float, default=0.025)
args = parser.parse_args()
expected = sorted(args.expected.glob("logits-*.pt"))
actual = sorted(args.actual.glob("logits-*.pt"))
if not expected or len(expected) != len(actual):
    raise SystemExit(f"capture count differs: CPU={len(expected)}, Neuron={len(actual)}")
for step, (expected_path, actual_path) in enumerate(zip(expected, actual)):
    lhs = torch.load(expected_path, weights_only=True)
    rhs = torch.load(actual_path, weights_only=True)
    if not torch.isfinite(rhs).all():
        raise SystemExit(f"step {step}: Neuron logits are non-finite")
    if not torch.equal(lhs.argmax(-1), rhs.argmax(-1)):
        raise SystemExit(f"step {step}: argmax token differs")
    torch.testing.assert_close(lhs, rhs, atol=args.tolerance, rtol=args.tolerance)
