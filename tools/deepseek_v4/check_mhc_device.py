#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile the DeepSeek-V4 mHC gate alone and compare Neuron against CPU.

The full model costs 7-13 minutes per device compile, far too slow to bisect a
numerical bug. This compiles *one module* -- cost scales with that module's
graph, not the model's ~55k nodes -- following
``examples/vllm_neuron/basics/helloworld.py``. In practice it compiles in about
1.5 seconds.

It exists because ``DeepseekV4HyperConnection`` was the first module to diverge
on device while being exact on CPU. Root cause: ``Tensor.split`` with a *list*
of sizes returns wrong data on Neuron for any dim other than 0 (see
``--check-split``). Keep this runnable -- it is the regression gate for that
fix, and the fastest available probe for the next lowering bug.

Usage::

    # harness sanity check, no hardware
    VLLM_NEURON_CPU_MODE=1 python tools/deepseek_v4/check_mhc_device.py

    # the real check
    PATH="$PWD/.venv-neuron/bin:$PATH" \
    NEURON_VISIBLE_DEVICES=0 NEURON_RT_VISIBLE_CORES=0 \
    NEURON_SKIP_EFA_AFFINITY=1 VLLM_CACHE_ROOT=/tmp/mhc-cache \
    python tools/deepseek_v4/check_mhc_device.py

    # re-characterise the underlying split lowering defect
    ... python tools/deepseek_v4/check_mhc_device.py --check-split

``NEURON_RT_VISIBLE_CORES`` must be set alongside ``NEURON_VISIBLE_DEVICES``;
the runtime rejects one without the other, and unlike the vLLM worker this
harness has nothing to set it for you.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

import vllm_neuron  # noqa: F401  -- registers the dynamo backends
from vllm_neuron import envs
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.model.deepseek_v4.model import DeepseekV4HyperConnection

#: The 8-token prompt every DeepSeek-V4 comparison in this repo uses. It exactly
#: fills the length-8 bucket, so no position is padding.
PROMPT = (671, 6102, 294, 8760, 344, 1024, 2048, 4096)


def _device() -> str:
    return "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"


def _compare(name: str, want: torch.Tensor, got: torch.Tensor, tolerance: float) -> bool:
    want, got = want.float(), got.to("cpu").float()
    delta = (want - got).abs()
    per_token = delta.reshape(delta.shape[0], -1).max(dim=-1).values
    worst = delta.max().item()
    print(f"  {name:<10} max|diff|={worst:<14.8g} per-token "
          + " ".join(f"{v:.3g}" for v in per_token.tolist()))
    return worst <= tolerance


def check_gate(slice_dir: Path, tolerance: float) -> bool:
    """Run the real ``DeepseekV4HyperConnection`` on CPU and on device."""
    with safe_open(str(slice_dir / "model.safetensors"), "pt") as handle:
        fn = handle.get_tensor("layers.0.hc_attn_fn").float()
        base = handle.get_tensor("layers.0.hc_attn_base").float()
        scale = handle.get_tensor("layers.0.hc_attn_scale").float()
        embed = handle.get_tensor("embed.weight")

    hidden = embed[torch.tensor(PROMPT)].float()
    # Exactly how DeepseekV4Model.forward builds the residual streams.
    streams = hidden.unsqueeze(-2).expand(-1, 4, -1)

    # fn is [(2 + hc) * hc, hc * width], so the stream count follows from the
    # weight shape rather than being hardcoded.
    width = hidden.shape[-1]
    config = SimpleNamespace(
        hc_mult=fn.shape[1] // width,
        hidden_size=width,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    gate = DeepseekV4HyperConnection(config)
    with torch.no_grad():
        gate.fn.copy_(fn)
        gate.base.copy_(base)
        gate.hc_scale.copy_(scale)
        expected = gate(streams)

    device = _device()
    compiled = torch.compile(gate.to(device), backend=get_compile_backend_name())
    with torch.no_grad():
        actual = compiled(streams.to(device))

    print(f"DeepseekV4HyperConnection on {device}")
    ok = True
    for name, want, got in zip(("post", "comb", "collapsed"), expected, actual):
        ok &= _compare(name, want, got, tolerance)
    return ok


class _Split(torch.nn.Module):
    def __init__(self, sizes, dim, op="split"):
        super().__init__()
        self.sizes, self.dim, self.op = sizes, dim, op

    def forward(self, x):
        if self.op == "chunk":
            return list(x.chunk(self.sizes, dim=self.dim))
        return list(x.split(self.sizes, dim=self.dim))


def check_split() -> bool:
    """Characterise the `Tensor.split` lowering defect this tool was built for.

    Reports which chunk indices come back wrong. A correct backend reports none
    for every case.
    """
    x = torch.arange(8 * 24, dtype=torch.float32).reshape(8, 24)
    cases = (
        ("split([4,4,16], dim=-1)", [4, 4, 16], -1),
        ("split([8,8,8], dim=-1)", [8, 8, 8], -1),
        ("split([4,20], dim=1)", [4, 20], 1),
        ("split(8, dim=-1) int", 8, -1),
        ("chunk(3, dim=-1)", 3, -1, "chunk"),
        ("split([2,6], dim=0)", [2, 6], 0),
    )
    device, ok = _device(), True
    print(f"Tensor.split lowering on {device}")
    for name, sizes, dim, *rest in cases:
        module = _Split(sizes, dim, rest[0] if rest else "split")
        with torch.no_grad():
            want = module(x)
        compiled = torch.compile(module.to(device), backend=get_compile_backend_name())
        with torch.no_grad():
            got = compiled(x.to(device))
        wrong = [
            i for i, (a, b) in enumerate(zip(want, got))
            if not torch.equal(a.float(), b.to("cpu").float())
        ]
        ok &= not wrong
        print(f"  {name:<26} wrong chunks: {wrong if wrong else 'none'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slice-dir", type=Path, default=Path("/home/ssm-user/ds-v4-tiny-real")
    )
    parser.add_argument(
        "--check-split", action="store_true",
        help="Characterise the split lowering defect instead of running the gate.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-3,
        help="Per-element bound. The gate runs in FP32 on both sides, so a "
             "correct lowering agrees far more tightly than this.",
    )
    args = parser.parse_args()
    ok = check_split() if args.check_split else check_gate(args.slice_dir, args.tolerance)
    print("MATCH" if ok else "DIVERGENCE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
