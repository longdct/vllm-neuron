#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the Gated DeltaNet chunk-scan kernel on Trainium, isolated from the model.

NKI's own standalone path is unusable in this install: ``nki.baremetal`` shells
out to ``python /tmp/nki_XXXXXXXX/None`` and dies with ``[Errno 2] No such file
or directory``, so for a long time the kernel could only ever be judged through
whole-model output. That is how it came to be called "wrong on device" when it
is not.

This goes around that. ``wrap_nki`` already produces a torch-callable HOP, so
compiling a one-op module with the ``neuron`` backend drives the kernel through
*exactly* the lowering production uses -- a better harness than baremetal would
have been, because a bug that only appears under the real lowering is still in
scope while the rest of the model is not.

Operands are built on CPU and the torch chunk rule is evaluated on the same
bits, so any difference is the kernel's alone.

Usage::

    NEURON_VISIBLE_DEVICES=4 VLLM_NEURON_ENABLE_QWEN3_5=1 \\
    VLLM_NEURON_ENABLE_QWEN3_5_SCAN_KERNEL=1 \\
        python tools/qwen3_5/run_scan_device.py --case 1,8,32,64,128

``--case`` is ``batch,heads,chunks,chunk,dim``; the default sweep walks from a
tiny geometry up to the one the 0.8B actually runs (``1,16,32,64,128`` at TP=1,
``1,8,32,64,128`` at TP=2), moving one axis at a time so the first failing case
names the axis.

Note the decay regime, which is easy to get wrong and hides everything: with
``g = -rand(0,1)`` the cumulative decay over a 64-token chunk is ``exp(-32)``,
which annihilates the carried state every chunk and makes the sequential
recurrence very nearly a no-op. ``--g-scale`` sweeps it; small values are the
regime where the carry actually carries.
"""

from __future__ import annotations

import argparse
import sys

import torch

from vllm_neuron.model.qwen3_5 import nki_gdn
from vllm_neuron.model.qwen3_5.gated_deltanet import chunk_gated_delta_rule

DEFAULT_CASES = [
    ((1, 2, 3, 16, 16), "baseline"),
    ((1, 2, 3, 64, 16), "chunk 16->64"),
    ((1, 2, 3, 16, 128), "dim 16->128"),
    ((1, 2, 3, 64, 128), "chunk+dim"),
    ((1, 2, 32, 64, 128), "chunks 3->32"),
    ((1, 8, 32, 64, 128), "0.8B @ TP=2"),
    ((1, 16, 32, 64, 128), "0.8B @ TP=1"),
]


class ScanModule(torch.nn.Module):
    """One op: the wrapped kernel, and nothing else in the graph."""

    def forward(self, q_g_T, k_cumdecay_T, attn_T, k_decay, v_base, g_last, state):
        return nki_gdn._wrapped_gdn_chunk_scan[nki_gdn._SCAN_LNC](
            q_g_T, k_cumdecay_T, attn_T, k_decay, v_base, g_last, state
        )


def build(batch, heads, chunks, chunk, dim, seed, g_scale):
    torch.manual_seed(seed)
    tokens = chunks * chunk
    q = torch.randn(batch, tokens, heads, dim)
    k = torch.randn(batch, tokens, heads, dim)
    v = torch.randn(batch, tokens, heads, dim)
    g = -torch.rand(batch, tokens, heads) * g_scale
    beta = torch.rand(batch, tokens, heads)

    want, want_state = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=chunk)
    prep = nki_gdn._prepare_chunk_scan(q, k, v, g, beta, chunk, True)
    rows = prep["batch"] * prep["heads"]
    state_in = torch.zeros(rows, prep["k_dim"], prep["v_dim"])
    tensors, _ = nki_gdn.pad_rows_for_lnc(
        [prep[n] for n in ("q_g_T", "k_cumdecay_T", "attn_T", "k_decay",
                           "v_base", "g_last_rep")] + [state_in],
        rows,
    )
    return tensors, prep, want, want_state, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="batch,heads,chunks,chunk,dim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--g-scale", type=float, default=1.0,
                        help="scales g; small values keep the carried state alive")
    parser.add_argument("--device", default="neuron:0")
    parser.add_argument("--tol", type=float, default=2e-3)
    args = parser.parse_args()

    if not nki_gdn.can_use_chunk_scan_kernel(
        torch.zeros(1, 4, 1, 128, device=args.device), 64
    ):
        print("chunk-scan kernel is not enabled; set "
              "VLLM_NEURON_ENABLE_QWEN3_5_SCAN_KERNEL=1", file=sys.stderr)
        return 2

    cases = DEFAULT_CASES
    if args.case:
        cases = [(tuple(int(x) for x in args.case.split(",")), "requested")]

    compiled = torch.compile(ScanModule().to(args.device), backend="neuron",
                             fullgraph=True, dynamic=False)
    failures = 0
    for (batch, heads, chunks, chunk, dim), label in cases:
        tag = f"b{batch} h{heads} c{chunks} w{chunk} d{dim}"
        tensors, prep, want, want_state, rows = build(
            batch, heads, chunks, chunk, dim, args.seed, args.g_scale)
        out, state = compiled(*[t.to(args.device) for t in tensors])
        out, state = out.cpu().float(), state.cpu().float()

        v_dim, k_dim, seq_len = prep["v_dim"], prep["k_dim"], prep["seq_len"]
        got = out[:rows].reshape(batch, heads, -1, v_dim)[:, :, :seq_len]
        got = got.transpose(1, 2).contiguous()
        got_state = state[:rows].reshape(batch, heads, k_dim, v_dim)

        r_out = (got - want).abs().max().item() / max(want.abs().max().item(), 1e-9)
        r_st = (got_state - want_state).abs().max().item() / max(
            want_state.abs().max().item(), 1e-9)
        ok = max(r_out, r_st) < args.tol
        failures += not ok
        print(f"{'MATCH ' if ok else 'DIVERGE'} {tag:26s} out_rel={r_out:.3e} "
              f"state_rel={r_st:.3e}   [{label}]")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
