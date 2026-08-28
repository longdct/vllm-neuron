#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize a dumped Neuron graph: NKI call sites, op mix, largest tensors.

Whole-model compile cost on this stack tracks the number of *opaque* NKI custom
calls far more closely than it tracks instruction count, so a graph that looks
small can still take hours.  This decodes a ``neuron_hlo_*.pb`` dump and reports
the call sites per kernel, which is the number to watch.

The dumps produced by the torch-neuronx 2.12 native path are StableHLO MLIR
bytecode, not ``xla.HloModuleProto``; parsing them as protobuf fails with
"Wire format was corrupt".

Usage:
    python tools/deepseek_v4/count_graph_callsites.py /tmp/neuron_hlo_1234_5.pb
"""

from __future__ import annotations

import argparse
import base64
import re
from collections import Counter
from pathlib import Path


def load_module_text(path: Path) -> str:
    from torch_mlir.ir import Context, Module

    context = Context()
    context.allow_unregistered_dialects = True
    module = Module.parse(path.read_bytes(), context)
    return module.operation.get_asm(
        large_elements_limit=8, enable_debug_info=False
    )


def kernel_call_sites(text: str) -> Counter:
    counts: Counter = Counter()
    for payload in re.findall(r'backend_config = "([A-Za-z0-9+/=]+)"', text):
        padded = payload + "=" * (-len(payload) % 4)
        try:
            raw = base64.b64decode(padded)
        except Exception:
            counts["<undecodable>"] += 1
            continue
        match = re.search(rb'"func_name":\s*"([^"]+)"', raw)
        counts[match.group(1).decode() if match else "<unnamed>"] += 1
    return counts


def op_histogram(text: str) -> Counter:
    return Counter(
        op.rsplit("= ", 1)[-1].lstrip('"')
        for op in re.findall(r'= "?(?:stablehlo|chlo|mhlo)\.[a-z_0-9]+', text)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path, help="path to a neuron_hlo_*.pb dump")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    text = load_module_text(args.graph)
    calls = kernel_call_sites(text)
    ops = op_histogram(text)

    print(f"{args.graph}: {len(text.splitlines())} lines of StableHLO")
    print(f"\nNKI custom-call sites (total {sum(calls.values())}):")
    for name, count in calls.most_common():
        print(f"  {count:6d}  {name}")
    print(f"\nTop {args.top} ops:")
    for name, count in ops.most_common(args.top):
        print(f"  {count:6d}  {name}")


if __name__ == "__main__":
    main()
