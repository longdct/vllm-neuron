#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Portable mHC / Tensor.split device check -- runs on *both* Neuron stacks.

`tools/deepseek_v4/check_mhc_device.py` imports `vllm_neuron`, so it only runs
on the released stack (torch/torch-xla 2.11). This variant imports nothing but
`mhc.py`, which is pure torch, so the same experiment can also be run against
the from-source torch-neuronx dev stack (torch 2.12.1, torch-mlir/StableHLO)
where vLLM is not installed and cannot be -- vllm 0.24.0 hard-pins torch 2.11.

That makes the interesting comparison possible: the same math, the same
neuronx-cc, two different lowerings.

Usage::

    # released stack (torch-xla 2.11)
    NEURON_VISIBLE_DEVICES=0 NEURON_RT_VISIBLE_CORES=0 \
        .venv-neuron/bin/python tools/deepseek_v4/check_mhc_portable.py

    # dev stack (torch 2.12.1)
    NEURON_VISIBLE_DEVICES=0 NEURON_RT_VISIBLE_CORES=0 \
        ~/.venv-torch-neuronx-dev/bin/python tools/deepseek_v4/check_mhc_portable.py

Pass ``--mhc-path`` to point at an alternate `mhc.py` (e.g. the sliced version
from commit 3932409) to isolate the source change from the stack change.
Exits 0 on match, 1 on divergence.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

DEFAULT_MHC = Path(__file__).resolve().parents[2] / "vllm_neuron/model/deepseek_v4/mhc.py"


def load_mhc(path: Path):
    """Load mhc.py directly. It imports only torch, so no package init runs."""
    name = "_mhc_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: Dynamo resolves a traced function's globals by
    # re-importing its __module__, so an unregistered module aborts the trace.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_stack() -> tuple[str, str, str]:
    """Return (label, device, dynamo backend) for whichever stack is installed."""
    try:
        import torch_neuronx  # noqa: F401  -- registers the PrivateUse1 'neuron' device
        return f"torch-neuronx dev {torch_neuronx.__version__}", "neuron:0", "neuron"
    except ImportError:
        pass
    import vllm_neuron  # noqa: F401  -- registers the dynamo backends
    from vllm_neuron.envs import get_compile_backend_name
    return "vllm-neuron released (torch-xla)", "neuron:0", get_compile_backend_name()


class Gate(torch.nn.Module):
    """`hyperconnection_reference` as a compilable module."""

    def __init__(self, mhc, fn, base, scale):
        super().__init__()
        self.reference = mhc.hyperconnection_reference
        self.register_buffer("fn", fn)
        self.register_buffer("base", base)
        self.register_buffer("scale", scale)

    def forward(self, streams):
        return self.reference(streams, self.fn, self.base, self.scale)


class Split(torch.nn.Module):
    def __init__(self, sizes, dim):
        super().__init__()
        self.sizes, self.dim = sizes, dim

    def forward(self, x):
        return list(x.split(self.sizes, dim=self.dim))


def compare(name: str, want: torch.Tensor, got: torch.Tensor, tolerance: float) -> bool:
    want, got = want.float(), got.to("cpu").float()
    worst = (want - got).abs().max().item()
    verdict = "ok" if worst <= tolerance else "WRONG"
    print(f"  {name:<10} max|diff| = {worst:<16.8g} {verdict}")
    return worst <= tolerance


def check_gate(mhc, device: str, backend: str, tolerance: float) -> bool:
    torch.manual_seed(0)
    hc, width, tokens = 4, 32, 8
    mix = (2 + hc) * hc
    fn = torch.randn(mix, hc * width) * 0.05
    base = torch.randn(mix) * 0.05
    scale = torch.tensor([1.0, 1.0, 1.0])
    streams = torch.randn(tokens, hc, width)

    gate = Gate(mhc, fn, base, scale)
    with torch.no_grad():
        expected = gate(streams)
    compiled = torch.compile(gate.to(device), backend=backend, dynamic=False)
    with torch.no_grad():
        actual = compiled(streams.to(device))

    print(f"hyperconnection_reference  [hc={hc}, width={width}, tokens={tokens}]")
    ok = True
    for name, want, got in zip(("post", "comb", "collapsed"), expected, actual):
        ok &= compare(name, want, got, tolerance)
    return ok


def check_split(device: str, backend: str) -> bool:
    x = torch.arange(8 * 24, dtype=torch.float32).reshape(8, 24)
    cases = (
        ("split([4,4,16], dim=-1)", [4, 4, 16], -1),
        ("split([8,8,8], dim=-1)", [8, 8, 8], -1),
        ("split([4,20], dim=1)", [4, 20], 1),
        ("split(8, dim=-1) int", 8, -1),
        ("split([2,6], dim=0)", [2, 6], 0),
    )
    print("Tensor.split lowering")
    ok = True
    for name, sizes, dim in cases:
        module = Split(sizes, dim)
        with torch.no_grad():
            want = module(x)
        compiled = torch.compile(module.to(device), backend=backend, dynamic=False)
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
    parser.add_argument("--mhc-path", type=Path, default=DEFAULT_MHC)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    label, device, backend = resolve_stack()
    print(f"stack   : {label}")
    print(f"torch   : {torch.__version__}")
    print(f"device  : {device}   backend: {backend}")
    print(f"mhc.py  : {args.mhc_path}\n")

    ok = True
    if not args.skip_split:
        ok &= check_split(device, backend)
        print()
    if not args.skip_gate:
        ok &= check_gate(load_mhc(args.mhc_path), device, backend, args.tolerance)
        print()
    print("MATCH" if ok else "DIVERGENCE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
