#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the FP8 checkpoint reconstruct the BF16 oracle exactly?

Milestone 2's gate. The MXFP4 -> FP8 widening is bit-exact by construction, so
this is not a tolerance check: dequantizing the FP8 experts must reproduce the
oracle's bf16 tensors bit for bit, and every non-expert tensor -- dequantized
identically by both builds -- must already be identical.
"""
import sys, torch
from safetensors import safe_open

sys.path.insert(0, "/home/ubuntu/vllm-neuron/.claude/worktrees/deepseek-V4-on-0.24")
from vllm_neuron.model.deepseek_v4.quant_formats import dequantize_fp8_per_channel

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--fp8",
    default="/home/ubuntu/ds-v4-3layer-256experts-fp8/model.safetensors",
)
parser.add_argument(
    "--bf16",
    default="/home/ubuntu/ds-v4-3layer-256experts-real/model.safetensors",
)
parser.add_argument(
    "--experts",
    default="0,1,17,99,255",
    help="expert ids to dequantize and compare bitwise (all of them is slow)",
)
args = parser.parse_args()
FP8, BF16 = args.fp8, args.bf16
SAMPLE = {int(x) for x in args.experts.split(",") if x}

with safe_open(FP8, "pt") as f8, safe_open(BF16, "pt") as f16:
    n8, n16 = set(f8.keys()), set(f16.keys())
    scales = {n for n in n8 if n.endswith(".scale")}
    print(f"fp8 tensors {len(n8)} (of which {len(scales)} scales) | bf16 tensors {len(n16)}")
    missing = n16 - (n8 - scales)
    extra = (n8 - scales) - n16
    print(f"missing vs oracle: {sorted(missing)[:5] or 'none'}")
    print(f"extra   vs oracle: {sorted(extra)[:5] or 'none'}")
    assert not missing and not extra, "inventory mismatch"

    expert_names = sorted(n for n in n8 if ".experts." in n and n.endswith(".weight"))
    print(f"expert weight tensors: {len(expert_names)}")

    # dtypes
    dt8 = {str(f8.get_slice(n).get_dtype()) for n in expert_names}
    non_expert = sorted((n8 - scales) - set(expert_names))
    dtne = {str(f8.get_slice(n).get_dtype()) for n in non_expert}
    print(f"expert dtypes {dt8} | non-expert dtypes {dtne}")

    # Every non-expert tensor must be byte-identical: both builds dequantized
    # the same source the same way.
    bad = []
    for n in non_expert:
        if not torch.equal(f8.get_tensor(n), f16.get_tensor(n)):
            bad.append(n)
    print(f"non-expert tensors identical: {len(non_expert) - len(bad)}/{len(non_expert)}"
          + (f"  MISMATCH: {bad[:4]}" if bad else ""))

    # Experts: dequantize and require bitwise equality with the oracle.
    checked = mism = 0
    worst = 0.0
    for n in expert_names:
        layer = n.split(".")[1]
        eid = int(n.split(".experts.")[1].split(".")[0])
        if eid not in SAMPLE:
            continue
        w = f8.get_tensor(n)
        s = f8.get_tensor(n.rsplit(".", 1)[0] + ".scale")
        restored = dequantize_fp8_per_channel(w, s).to(torch.bfloat16)
        ref = f16.get_tensor(n)
        checked += 1
        if not torch.equal(restored, ref):
            mism += 1
            worst = max(worst, float((restored.float() - ref.float()).abs().max()))
    print(f"expert tensors bitwise-equal to oracle: {checked - mism}/{checked}"
          + (f"  worst abs diff {worst:.3e}" if mism else ""))
    print("RESULT:", "PASS" if not bad and not mism else "FAIL")
