#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare rank-local DeepSeek-V4 captures with a router-tie-robust metric."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vllm_neuron.accuracy.tensor_compare import align_shapes
from vllm_neuron.accuracy.utils import natural_sort_key


def median_row_relative_rms(
    reference: torch.Tensor, actual: torch.Tensor
) -> tuple[float, float]:
    """Return median and p95 row-relative RMS after standard shape alignment."""
    reference, actual, matched = align_shapes(reference.float(), actual.float())
    if not matched:
        raise ValueError(
            f"capture shapes do not align: {tuple(reference.shape)} vs "
            f"{tuple(actual.shape)}"
        )
    if reference.ndim == 0:
        reference = reference.reshape(1, 1)
        actual = actual.reshape(1, 1)
    elif reference.ndim == 1:
        reference = reference.reshape(1, -1)
        actual = actual.reshape(1, -1)
    else:
        reference = reference.reshape(-1, reference.shape[-1])
        actual = actual.reshape(-1, actual.shape[-1])

    error_rms = (reference - actual).square().mean(dim=-1).sqrt()
    reference_rms = reference.square().mean(dim=-1).sqrt()
    relative = error_rms / reference_rms.clamp_min(torch.finfo(torch.float32).eps)
    return (
        torch.quantile(relative, 0.5).item(),
        torch.quantile(relative, 0.95).item(),
    )


def _capture_files(root: Path, rank: int) -> dict[tuple[str, str, str], Path]:
    suffix = f"_rank{rank}.pt"
    captures = {}
    for path in root.glob(f"prompt_*/step_*/*{suffix}"):
        module = path.name[: -len(suffix)]
        key = (path.parent.parent.name, path.parent.name, module)
        captures[key] = path
    return captures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TP capture directories using median row-relative RMS; "
            "this is less sensitive to isolated BF16 router ties than maximum error."
        )
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.rank < 0:
        parser.error("--rank must be non-negative")
    if args.threshold < 0:
        parser.error("--threshold must be non-negative")

    reference_files = _capture_files(args.reference, args.rank)
    actual_files = _capture_files(args.actual, args.rank)
    reference_only = reference_files.keys() - actual_files.keys()
    actual_only = actual_files.keys() - reference_files.keys()
    if reference_only or actual_only:
        missing = []
        if reference_only:
            missing.append(f"{len(reference_only)} missing from actual")
        if actual_only:
            missing.append(f"{len(actual_only)} missing from reference")
        parser.error("capture sets differ: " + ", ".join(missing))
    common = sorted(
        reference_files.keys() & actual_files.keys(),
        key=lambda key: natural_sort_key("/".join(key)),
    )
    if not common:
        parser.error(
            f"no common rank-{args.rank} captures under {args.reference} and "
            f"{args.actual}"
        )

    failed = False
    print(f"{'Prompt/step/module':<72} {'median':>12} {'p95':>12} status")
    for key in common:
        reference = torch.load(reference_files[key], weights_only=True)
        actual = torch.load(actual_files[key], weights_only=True)
        label = "/".join(key)
        try:
            median, p95 = median_row_relative_rms(reference, actual)
            passed = median <= args.threshold
            status = "PASS" if passed else "FAIL"
            print(f"{label:<72} {median:12.4e} {p95:12.4e} {status}")
            failed |= not passed
        except ValueError as error:
            print(f"{label:<72} {'-':>12} {'-':>12} SHAPE: {error}")
            failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
