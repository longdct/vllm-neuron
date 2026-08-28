#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile P2's 512-d NKI attention buckets and emit reproducible JSON."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from nkilib.core.attention.attention_cte import attention_cte

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vllm_neuron.model.deepseek_v4.attention import P2_REPRESENTATIVE_BUCKETS


def compile_nki(*args, **kwargs):
    raise RuntimeError(
        "Standalone lite NKI compilation was removed. Compile this kernel "
        "through torch_neuronx.nki_hop inside backend='neuron'."
    )


def version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return "not-on-PATH"
    return (result.stdout + result.stderr).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context", type=int, default=512)
    args = parser.parse_args()

    records = []
    for bucket in P2_REPRESENTATIVE_BUCKETS:
        if bucket.context_length > args.max_context:
            continue
        q = torch.zeros(
            bucket.batch_size,
            bucket.query_length,
            bucket.head_dim,
            dtype=torch.bfloat16,
        )
        k = torch.zeros(
            bucket.batch_size,
            bucket.head_dim,
            bucket.context_length,
            dtype=torch.bfloat16,
        )
        v = torch.zeros(
            bucket.batch_size,
            bucket.context_length,
            bucket.head_dim,
            dtype=torch.bfloat16,
        )
        kernel_args = {
            "q": q,
            "k": k,
            "v": v,
            "scale": 1 / math.sqrt(bucket.head_dim),
            "causal_mask": bucket.query_length > 1,
            "tp_q": True,
            "tp_k": False,
            "tp_out": False,
        }
        started = time.monotonic()
        result = compile_nki(attention_cte.func, kernel_args, (1,))
        records.append(
            {
                "bucket": asdict(bucket),
                "wall_seconds": time.monotonic() - started,
                "return_types": [
                    {"dtype": str(dtype), "shape": shape}
                    for dtype, shape in result.return_types
                ],
                "backend_config_base64_chars": len(result.dumped_config),
            }
        )

    artifact = {
        "gate": "P2.c-kernel-compile-only",
        "does_not_prove": ["graph capture", "NEFF generation", "device execution"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "neuronx_cc": importlib.metadata.version("neuronx-cc"),
        "target": "trn2",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")


if __name__ == "__main__":
    main()
