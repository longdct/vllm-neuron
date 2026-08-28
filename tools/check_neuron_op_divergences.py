#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the CPU/Neuron op divergences in the inventory, in one run.

Companion to ``docs/model-dev/neuron-cpu-op-divergences.md``. Each check drives
the **unfixed** form of an op -- the shape this project stopped writing -- and
diffs it against CPU. Testing the workarounds instead would pass everywhere and
prove nothing about the backend.

This is a *characterization* harness, not a test suite: a documented divergence
is the expected result, so it fails only when reality stops matching the
document. That makes it a tripwire in both directions. A ``MATCHES`` where
``DIVERGES`` was expected means the stack was fixed and a workaround may be
retirable; an unexpected ``DIVERGES`` means a regression, or an inventory gap.

**Every check runs in its own subprocess**, and that is not defensive
programming. When the backend cannot lower an op it does not raise into Python:
``libtorch_neuronx_lite/compile/cache.py`` logs "Compilation failed --
terminating process for cleanup" and takes the interpreter down. In-process,
one unlowerable op ends the run at that row.

Usage::

    PATH="$VENV/bin:$PATH" PYTHONPATH=$PWD \\
    NEURON_VISIBLE_DEVICES=0 NEURON_RT_VISIBLE_CORES=0 \\
    VLLM_CACHE_ROOT=$(mktemp -d) \\
      $VENV/bin/python tools/check_neuron_op_divergences.py

A private ``VLLM_CACHE_ROOT`` matters: the compile cache is shared across venvs
by default, so without it a NEFF built by another stack can answer for this one.

Exit codes: 0 = reality matches the inventory, 1 = something changed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import torch

DEVICE = "neuron:0"

MATCHES = "MATCHES"     # device result equals CPU, dtype included
DIVERGES = "DIVERGES"   # device ran and produced different values or dtype
REJECTED = "REJECTED"   # device refused to lower/run it
SKIP = "SKIP"


# --------------------------------------------------------------------------
# The op forms. Each is the *pre-workaround* shape.
# --------------------------------------------------------------------------
class SplitList(torch.nn.Module):
    """Entry 1: list-of-sizes split on a non-zero dim."""

    def forward(self, x):
        return list(x.split([8, 8, 8], dim=-1))


class SplitInt(torch.nn.Module):
    """Entry 1 control: an int size must stay correct."""

    def forward(self, x):
        return list(x.split(8, dim=-1))


class CatRotarySuffix(torch.nn.Module):
    """Entry 2: pre-15e548c ``apply_partial_rotary``, rank 4."""

    def forward(self, x, cos, sin):
        rotary = x[..., -2:]
        left, right = rotary[..., :1], rotary[..., 1:]
        rotated = torch.cat((left * cos - right * sin, right * cos + left * sin), -1)
        return torch.cat((x[..., :-2], rotated), dim=-1)


class CastNotFinal(torch.nn.Module):
    """Entry 4: uint32 cast followed by more arithmetic.

    The follow-on op is a ``repeat`` rather than an add: torch CPU has no
    ``add`` kernel for uint32, so an arithmetic follow-on would fail on the
    reference side and measure nothing.
    """

    def forward(self, counts):
        total = counts.to(torch.int32).sum().to(torch.uint32)
        return total.repeat(4)


class NarrowScatter(torch.nn.Module):
    """Entry 5: scatter K columns of indices into a buffer narrower than K."""

    def forward(self, index):
        mask = torch.zeros(index.shape[0], 3, dtype=torch.int32, device=index.device)
        return mask.scatter(-1, index, torch.ones_like(index, dtype=torch.int32))


class TopkIndices(torch.nn.Module):
    """Entry 6: top-k index dtype."""

    def forward(self, scores):
        return torch.topk(scores, 2, dim=-1)[1]


class NonContiguousIndexPut(torch.nn.Module):
    """Entry 7: non-contiguous payload into index_put_."""

    def forward(self, cache, rows, value):
        out = cache.clone()
        out.index_put_((rows,), value.transpose(0, 1))
        return out


class StrideTwoStore(torch.nn.Module):
    """Entry 9: tensor-indexed partial store into a packed ``[.., 2]`` slot."""

    def forward(self, cache, slots, lanes, value):
        out = cache.clone()
        out[slots, :, lanes] = value
        return out


class SliceThenContiguous(torch.nn.Module):
    """Entry 10: ``[:, 1:].contiguous()`` on device."""

    def forward(self, stacked):
        return stacked[:, 1:].contiguous().view(-1)


class DataDependentBranch(torch.nn.Module):
    """Entry 11: control flow that reads tensor values."""

    def forward(self, x):
        if x.min() < 0:
            return x * 2
        return x + 1


def _seeded(*shape):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(0))


def build_checks():
    """(entry, name, expected, builder, note). builder -> (module, args)."""
    wide = torch.arange(8 * 24, dtype=torch.float32).reshape(8, 24)
    return [
        (1, "split(list, dim=-1)", DIVERGES,
         lambda: (SplitList(), (wide,)), ""),
        (1, "split(int, dim=-1)  [control]", MATCHES,
         lambda: (SplitInt(), (wide,)), ""),
        (2, "cat rotary suffix, rank 4", MATCHES,
         lambda: (CatRotarySuffix(),
                  (_seeded(1, 4, 6, 8), _seeded(1, 4, 6, 1), _seeded(1, 4, 6, 1))),
         "does not reproduce standalone; see NOT REPRODUCED below"),
        (3, "wide NKI gather (NKILIB-1592)", SKIP, None,
         "kernel-level, not a torch op -- out of scope here"),
        (4, "uint32 cast then repeat", MATCHES,
         lambda: (CastNotFinal(), (torch.tensor([3, 5, 0, 4], dtype=torch.int32),)),
         "does not reproduce standalone; see NOT REPRODUCED below"),
        (5, "scatter into mask narrower than K", MATCHES,
         lambda: (NarrowScatter(), (torch.tensor([[0, 1, 2, 0, 1, 2]]),)),
         "does not reproduce standalone; see NOT REPRODUCED below"),
        (6, "topk index dtype", DIVERGES,
         lambda: (TopkIndices(), (_seeded(1, 8),)), ""),
        (7, "non-contiguous index_put_", MATCHES,
         lambda: (NonContiguousIndexPut(),
                  (torch.zeros(4, 3), torch.tensor([0, 1, 2, 3]), _seeded(3, 4))),
         "does not reproduce standalone; see NOT REPRODUCED below"),
        (9, "stride-2 partial store", REJECTED,
         lambda: (StrideTwoStore(),
                  (torch.zeros(4, 6, 2), torch.tensor([0, 1]),
                   torch.tensor([1, 0]), _seeded(2, 6))), ""),
        (10, "[:, 1:].contiguous()", MATCHES,
         lambda: (SliceThenContiguous(),
                  (torch.arange(12, dtype=torch.float32).reshape(3, 4),)),
         "does not reproduce standalone; see NOT REPRODUCED below"),
        (11, "data-dependent branching", REJECTED,
         lambda: (DataDependentBranch(), (_seeded(4, 4),)), ""),
    ]


# --------------------------------------------------------------------------
# Child: run exactly one check, print one machine-readable line, exit.
# --------------------------------------------------------------------------
def _same(cpu, dev):
    """Compare CPU against device, dtype included.

    Dtype is part of the contract, not a detail: ``topk`` returns correct index
    *values* in an unsigned type, and the dtype alone is what breaks the ``-1``
    sentinel idiom downstream.
    """
    cpu = cpu if isinstance(cpu, (tuple, list)) else (cpu,)
    dev = dev if isinstance(dev, (tuple, list)) else (dev,)
    if len(cpu) != len(dev):
        return False, f"{len(cpu)} vs {len(dev)} outputs"
    for i, (a, b) in enumerate(zip(cpu, dev)):
        b = b.to("cpu")
        if a.dtype != b.dtype:
            return False, f"output {i} dtype {a.dtype} vs {b.dtype}"
        if a.shape != b.shape:
            return False, f"output {i} shape {tuple(a.shape)} vs {tuple(b.shape)}"
        if not torch.equal(a.float(), b.float()):
            worst = (a.float() - b.float()).abs().max().item()
            wrong = int((a.float() != b.float()).sum())
            return False, f"output {i}: {wrong} elements differ, max|diff|={worst:g}"
    return True, "identical to CPU"


def run_single(index: int) -> int:
    import vllm_neuron  # noqa: F401  -- registers the dynamo backends
    from vllm_neuron.envs import get_compile_backend_name

    _, _, _, builder, _ = build_checks()[index]
    module, args = builder()

    with torch.no_grad():
        try:
            expected = module(*args)
        except Exception as exc:
            print(f"RESULT|{REJECTED}|CPU side raised: {exc}")
            return 0
        try:
            # fullgraph=True matches how the model runner compiles. It matters:
            # with the default fullgraph=False, Dynamo *graph-breaks* around a
            # data-dependent branch and silently runs it in eager, so the check
            # passes and hides the very limitation it is testing for.
            compiled = torch.compile(
                module.to(DEVICE),
                backend=get_compile_backend_name(),
                dynamic=False,
                fullgraph=True,
            )
            actual = compiled(
                *(a.to(DEVICE) if torch.is_tensor(a) else a for a in args)
            )
            # Force materialization: Neuron reports errors asynchronously, so a
            # failure can surface only when the result is pulled back to host.
            actual = (
                [t.to("cpu") for t in actual]
                if isinstance(actual, (tuple, list))
                else actual.to("cpu")
            )
        except Exception as exc:
            line = str(exc).strip().splitlines()
            print(f"RESULT|{REJECTED}|{(line[0] if line else type(exc).__name__)[:100]}")
            return 0

    ok, detail = _same(expected, actual)
    print(f"RESULT|{MATCHES if ok else DIVERGES}|{detail}")
    return 0


# --------------------------------------------------------------------------
# Parent
# --------------------------------------------------------------------------
def run_in_child(index: int) -> tuple[str, str]:
    proc = subprocess.run(
        [sys.executable, __file__, "--single", str(index)],
        capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT|"):
            _, outcome, detail = line.split("|", 2)
            return outcome, detail
    # No RESULT line: the backend terminated the interpreter mid-compile.
    hint = "process terminated by the backend"
    for line in (proc.stdout + proc.stderr).splitlines():
        if "Error while lowering" in line or "Compilation failed" in line:
            hint = line.strip()[:100]
            break
    return REJECTED, hint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--only", type=int, help="run one inventory entry")
    args = parser.parse_args()

    if args.single is not None:
        return run_single(args.single)

    import vllm_neuron  # noqa: F401
    from vllm_neuron.envs import get_compile_backend_name

    import torch_xla

    versions = f"torch {torch.__version__} / torch-xla {torch_xla.__version__}"
    try:
        import importlib.metadata as md

        versions += f" / lite {md.version('libtorch-neuronx-lite')}"
    except Exception:
        pass
    print(f"stack: {versions}")
    print(f"backend: {get_compile_backend_name()!r} on {DEVICE}\n")

    header = f"{'#':>3}  {'op form':<34} {'expected':<9} {'observed':<9} detail"
    print(header)
    print("-" * len(header))

    changed, not_reproduced = [], []
    for index, (entry, name, expected, builder, note) in enumerate(build_checks()):
        if args.only and entry != args.only:
            continue
        if builder is None:
            print(f"{entry:>3}  {name:<34} {'--':<9} {SKIP:<9} {note}")
            continue
        outcome, detail = run_in_child(index)
        flag = "" if outcome == expected else "   <-- CHANGED"
        if flag:
            changed.append((entry, name, expected, outcome))
        if note.startswith("does not reproduce"):
            not_reproduced.append((entry, name))
        print(f"{entry:>3}  {name:<34} {expected:<9} {outcome:<9} {detail}{flag}")

    if not_reproduced:
        print("\nNOT REPRODUCED standalone (workarounds retained):")
        for entry, name in not_reproduced:
            print(f"  entry {entry:>2}  {name}")
        print("  These were observed inside a real model graph. A minimal repro")
        print("  matching CPU here does NOT clear them -- rank, surrounding ops and")
        print("  graph size all mattered for the ones that are understood. Do not")
        print("  remove a workaround on the strength of a green row above.")

    print()
    if not changed:
        print("Reality matches docs/model-dev/neuron-cpu-op-divergences.md.")
        return 0
    print(f"{len(changed)} check(s) no longer match the inventory:")
    for entry, name, expected, outcome in changed:
        if outcome == MATCHES:
            print(f"  entry {entry} ({name}): now MATCHES CPU -- the stack may be "
                  f"fixed. Confirm before retiring the workaround.")
        else:
            print(f"  entry {entry} ({name}): expected {expected}, got {outcome} "
                  f"-- update the inventory.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
