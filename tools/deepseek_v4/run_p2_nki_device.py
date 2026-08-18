#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Execute P2's 512-wide MLA kernel on Trainium with synthetic inputs.

This is intentionally independent of vLLM and model checkpoints.  It exercises
the exact NKI kernel used by the P2 spike through NKI's standalone NumPy path
and retains enough data to reproduce the numerical comparison off-device.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import ml_dtypes
import numpy as np
from nkilib.core.attention.attention_cte import attention_cte


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return "not-on-PATH"
    return (result.stdout + result.stderr).strip()


def reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool) -> np.ndarray:
    scores = q.astype(np.float32) @ k.astype(np.float32)
    scores /= math.sqrt(q.shape[-1])
    if causal:
        query_length, context_length = q.shape[-2], k.shape[-1]
        query_positions = np.arange(context_length - query_length, context_length)[:, None]
        key_positions = np.arange(context_length)[None, :]
        scores = np.where(key_positions[None] <= query_positions[None], scores, -np.inf)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return probabilities @ v.astype(np.float32)


def run_case(
    *, query_length: int, context_length: int, causal: bool, lnc: int, seed: int,
    warmup: int, iterations: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    q = np.asarray(rng.standard_normal((1, query_length, 512)), dtype=ml_dtypes.bfloat16)
    k = np.asarray(rng.standard_normal((1, 512, context_length)), dtype=ml_dtypes.bfloat16)
    v = np.asarray(rng.standard_normal((1, context_length, 512)), dtype=ml_dtypes.bfloat16)
    expected = reference(q, k, v, causal)

    durations = []
    output = None
    for index in range(warmup + iterations):
        started = time.monotonic()
        output = attention_cte[lnc](
            q=q, k=k, v=v, scale=1 / math.sqrt(512), causal_mask=causal,
            tp_q=True, tp_k=False, tp_out=False,
        )
        elapsed = time.monotonic() - started
        if index >= warmup:
            durations.append(elapsed)
    assert output is not None
    actual = output.astype(np.float32)
    absolute = np.abs(actual - expected)
    denominator = np.maximum(np.abs(expected), 1e-6)
    relative = absolute / denominator
    passed = bool(np.allclose(actual, expected, rtol=0.025, atol=0.025))
    name = f"q{query_length}-kv{context_length}-{'causal' if causal else 'decode'}"
    record = {
        "name": name,
        "query_length": query_length,
        "context_length": context_length,
        "head_dim": 512,
        "causal": causal,
        "seed": seed,
        "lnc": lnc,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "latency_seconds": durations,
        "latency_mean_seconds": float(np.mean(durations)),
        "latency_min_seconds": float(np.min(durations)),
        "max_absolute_error": float(np.max(absolute)),
        "p99_absolute_error": float(np.percentile(absolute, 99)),
        "max_relative_error": float(np.max(relative)),
        "finite": bool(np.isfinite(actual).all()),
        "rtol": 0.025,
        "atol": 0.025,
        "passed": passed,
    }
    arrays = {f"{name}_{key}": value for key, value in {
        "q": q, "k": k, "v": v, "expected_fp32": expected, "actual_bf16": output,
    }.items()}
    return record, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lnc", type=int, choices=(1, 2), default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE") not in ("trn2", "gen3"):
        raise SystemExit("set NEURON_PLATFORM_TARGET_OVERRIDE=trn2")

    records = []
    arrays: dict[str, np.ndarray] = {}
    for case in ((1, 8, False), (8, 8, True)):
        record, case_arrays = run_case(
            query_length=case[0], context_length=case[1], causal=case[2],
            lnc=args.lnc, seed=11 + case[0], warmup=args.warmup,
            iterations=args.iterations,
        )
        records.append(record)
        arrays.update(case_arrays)

    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "tensors.npz", **arrays)
    artifact = {
        "gate": "P2.d-direct-nki-device",
        "scope": "synthetic 512-wide MLA kernel; no model weights",
        "does_not_prove": ["vLLM 0.26 graph capture", "full model execution"],
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "nki": importlib.metadata.version("nki"),
        "neuronx_cc": importlib.metadata.version("neuronx-cc"),
        "torch_neuronx": importlib.metadata.version("torch-neuronx"),
        "neuron_ls": command_output(["neuron-ls"]),
        "records": records,
        "passed": all(record["passed"] and record["finite"] for record in records),
    }
    (args.output / "result.json").write_text(json.dumps(artifact, indent=2) + "\n")
    if not artifact["passed"]:
        raise SystemExit("P2 device numerical comparison failed")


if __name__ == "__main__":
    main()
