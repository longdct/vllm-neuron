#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device check for the lightning indexer's scoring and selection.

The full tiny model takes ~200 s to compile, which is a poor bisection loop for
"which op does Neuron mis-lower this time". This compiles only ``indexer.py``'s
three functions -- scoring, top-k selection, and the sentinel scatter that turns
picks into a mask -- and compares each against CPU.

The scatter is the one to watch. It writes into a buffer one column wider than
the entry axis so ``-1`` sentinels have somewhere to land, and an index that is
in bounds on CPU but not on device shows up as "Out of bounds access on model
... .neff" from the runtime, several frames away from the op that caused it.

Usage::

    NEURON_VISIBLE_DEVICES=0 NEURON_RT_VISIBLE_CORES=0 \
        python tools/deepseek_v4/check_indexer_device.py

Both variables are required outside vLLM: the runtime refuses
``NEURON_VISIBLE_DEVICES`` on its own, and vLLM refuses
``NEURON_RT_VISIBLE_CORES``, so the two entry points need different env.

``--stack`` selects the backend, because the two available ones lower this
differently -- torch-xla 2.11 returns **uint32** top-k indices where the
from-source torch-neuronx 2.12 returns int64::

    # vllm-neuron's torch-xla 2.11 (what this plugin ships on)
    PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:$PATH \
    PYTHONPATH=$PWD ~/.venv-vllm-neuron/bin/python \
        tools/deepseek_v4/check_indexer_device.py --stack xla

    # from-source torch-neuronx 2.12; its own bin must be on PATH or every
    # compile dies as "NEFF creation failed" (really: neuronx-cc not found)
    PATH=~/.venv-torch-neuronx-dev/bin:$PATH \
    ~/.venv-torch-neuronx-dev/bin/python \
        tools/deepseek_v4/check_indexer_device.py --stack neuronx

``--indexer-path`` points the probe at a different ``indexer.py``, which is how
a *backend* gets tested rather than the fix: the shipped module casts to int64
immediately, so it passes everywhere and says nothing about the lowering. Point
this at a copy with the cast removed to see the defect itself.

Exits 0 when every stage matches CPU, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

DEFAULT_INDEXER = (
    Path(__file__).resolve().parents[2] / "vllm_neuron/model/deepseek_v4/indexer.py"
)


def load_indexer(path: Path):
    """Load indexer.py directly -- it imports only torch, so no package init."""
    name = "_indexer_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: Dynamo resolves a traced function's globals by
    # re-importing its __module__, so an unregistered module aborts the trace.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_stack(preference: str) -> tuple[str, str, str]:
    """Return the TorchNeuron Native device, backend, and description."""
    if preference == "xla":
        raise ValueError("The XLA backend was removed; use native or auto")
    import torch_neuronx

    return (
        "neuron:0",
        "neuron",
        f"torch-neuronx {torch_neuronx.__version__} / torch {torch.__version__}",
    )


class Scores(torch.nn.Module):
    def __init__(self, indexer):
        super().__init__()
        self.fn = indexer.lightning_index_scores

    def forward(self, query, keys, gate):
        return self.fn(query, keys, gate)


class Select(torch.nn.Module):
    def __init__(self, indexer, topk):
        super().__init__()
        self.fn = indexer.select_compressed_entries
        self.topk = topk

    def forward(self, scores, visible):
        return self.fn(scores, visible, self.topk)


class Mask(torch.nn.Module):
    def __init__(self, indexer, entries):
        super().__init__()
        self.fn = indexer.selection_mask_from_indices
        self.entries = entries

    def forward(self, indices):
        return self.fn(indices, self.entries)


class RawTopk(torch.nn.Module):
    """``torch.topk``'s indices with no cast, to read the backend's index dtype.

    ``indexer.py`` casts to int64 immediately, which is the fix; this is the
    unfixed idiom, kept so the probe reports what the backend actually returns
    rather than what the fix makes of it.
    """

    def __init__(self, topk):
        super().__init__()
        self.topk = topk

    def forward(self, scores):
        return torch.topk(scores, self.topk, dim=-1)[1]


class EndToEnd(torch.nn.Module):
    """Everything the attention layer actually calls, in one graph."""

    def __init__(self, indexer, topk, entries):
        super().__init__()
        self.indexer = indexer
        self.topk = topk
        self.entries = entries

    def forward(self, query, keys, gate, visible):
        scores = self.indexer.lightning_index_scores(query, keys, gate)
        chosen = self.indexer.select_compressed_entries(scores, visible, self.topk)
        return self.indexer.selection_mask_from_indices(chosen, self.entries)


def report(name: str, want: torch.Tensor, got: torch.Tensor) -> bool:
    got = got.to("cpu")
    if want.dtype is torch.bool:
        ok = bool(torch.equal(want, got))
        detail = f"{int((want != got).sum())} of {want.numel()} elements differ"
    else:
        worst = (want.float() - got.float()).abs().max().item()
        ok = worst == 0.0 if want.dtype is torch.long else worst <= 1e-4
        detail = f"max|diff| = {worst:.8g}"
    print(f"  {name:<12} {detail:<40} {'ok' if ok else 'WRONG'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indexer-path", type=Path, default=DEFAULT_INDEXER)
    parser.add_argument("--entries", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument(
        "--stack",
        choices=("auto", "xla", "neuronx"),
        default="auto",
        help="which Neuron stack to drive; 'auto' prefers vllm-neuron's "
        "torch-xla and falls back to the from-source torch-neuronx",
    )
    args = parser.parse_args()

    indexer = load_indexer(args.indexer_path)
    device, backend, description = resolve_stack(args.stack)
    print(f"stack: {description}")
    print(f"       dynamo backend {backend!r} on {device}")
    print(
        f"entries={args.entries} topk={args.topk} heads={args.heads} "
        f"head_dim={args.head_dim}"
    )

    torch.manual_seed(0)
    query = torch.randn(1, 1, args.heads, args.head_dim)
    keys = torch.randn(1, args.entries, args.head_dim)
    gate = torch.randn(1, 1, args.heads)
    # Two regimes, and the second is the one that matters. With half the
    # entries visible the top-k has real values to rank. With *nothing* visible
    # every candidate is -inf, top-k may return anything, and Neuron returns
    # **uint32** indices where CPU returns int64 -- on which the -1 sentinel
    # wraps to 4294967295 and indexes far outside the scatter buffer. A probe
    # that only tests the first regime reports MATCH while the model faults.
    #
    # The third regime mixes the two: one visible entry against a budget of
    # ``topk``, so a single real index and a sentinel share one row. That is the
    # combination the model actually meets early in a sequence, and it is the
    # one where a sentinel is easiest to mistake for data -- the row is not
    # uniformly degenerate, so a consumer that checks "did anything get picked"
    # sees yes.
    regimes = [
        (args.entries // 2, "half visible"),
        (0, "nothing visible"),
        (1, "one visible -- real pick and sentinel in one row"),
    ]

    ok = True
    for visible_value, label in regimes:
        print(f"  regime: {label} (visible={visible_value})")
        visible = torch.tensor([[visible_value]])
        ok &= _check(indexer, args, device, backend, query, keys, gate, visible)
    print("MATCH" if ok else "DIVERGENCE")
    return 0 if ok else 1


def _check(indexer, args, device, backend, query, keys, gate, visible):
    ok = True
    with torch.no_grad():
        scores_cpu = Scores(indexer)(query, keys, gate)
        compiled = torch.compile(
            Scores(indexer).to(device), backend=backend, dynamic=False
        )
        ok &= report(
            "scores",
            scores_cpu,
            compiled(query.to(device), keys.to(device), gate.to(device)),
        )

        # Before the cast: what dtype does this backend's top-k hand back?
        # A signed type means the ``-1`` sentinel survives natively; an unsigned
        # one means indexer.py's ``.long()`` is load-bearing rather than tidy.
        masked = scores_cpu.float().masked_fill(
            torch.arange(scores_cpu.shape[-1]).view(1, 1, -1) >= visible.unsqueeze(-1),
            float("-inf"),
        )
        raw_cpu = RawTopk(args.topk)(masked)
        compiled = torch.compile(
            RawTopk(args.topk).to(device), backend=backend, dynamic=False
        )
        raw_dev = compiled(masked.to(device)).to("cpu")
        print(
            f"  {'topk dtype':<12} cpu={raw_cpu.dtype} device={raw_dev.dtype}"
            f"  device values={raw_dev.flatten().tolist()}"
        )

        chosen_cpu = Select(indexer, args.topk)(scores_cpu, visible)
        compiled = torch.compile(
            Select(indexer, args.topk).to(device), backend=backend, dynamic=False
        )
        ok &= report(
            "select",
            chosen_cpu,
            compiled(scores_cpu.to(device), visible.to(device)),
        )

        mask_cpu = Mask(indexer, args.entries)(chosen_cpu)
        compiled = torch.compile(
            Mask(indexer, args.entries).to(device), backend=backend, dynamic=False
        )
        ok &= report("mask", mask_cpu, compiled(chosen_cpu.to(device)))

        end_cpu = EndToEnd(indexer, args.topk, args.entries)(
            query, keys, gate, visible
        )
        compiled = torch.compile(
            EndToEnd(indexer, args.topk, args.entries).to(device),
            backend=backend,
            dynamic=False,
        )
        ok &= report(
            "end-to-end",
            end_cpu,
            compiled(
                query.to(device),
                keys.to(device),
                gate.to(device),
                visible.to(device),
            ),
        )

    return ok


if __name__ == "__main__":
    raise SystemExit(main())
