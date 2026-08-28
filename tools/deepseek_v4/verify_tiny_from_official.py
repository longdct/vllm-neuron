#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a slice built by ``build_tiny_from_official.py`` against its source.

Three independent checks, cheapest first:

1. **Structure** -- every emitted tensor is finite BF16 (or int64 for the routing
   table) and has the shape the sliced config implies.
2. **Dequantization** -- routed-expert weights are re-derived through the
   *official* ``cast_e2m1fn_to_e4m3fn`` from the checkpoint's own
   ``inference/convert.py``, which is a different code path to
   ``quant_formats.dequantize_fp4_blockwise``. Agreement between the two is real
   evidence; agreement of an implementation with itself is not.
3. **Routing** -- the remapped hash table stays in range and keeps every row's
   entries distinct, since ``scores.gather`` would otherwise double-apply an
   expert.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from quant_formats import FP4_TABLE, dequantize

MAX_OFFSET_BITS = 6
FP8_BLOCK = 128
FP4_BLOCK = 32


def official_fp4_reference(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Port of ``cast_e2m1fn_to_e4m3fn`` from the official ``inference/convert.py``.

    It lowers MXFP4 to ``float8_e4m3fn`` plus one ``e8m0`` scale per 128x128 tile,
    a deliberately different factorization from a direct per-32-block multiply.
    Reproducing the same real numbers both ways is the point.
    """
    assert packed.dtype == torch.int8 and packed.ndim == 2
    out_dim, in_dim = packed.size()
    in_dim *= 2
    raw = packed.view(torch.uint8)
    low, high = raw & 0x0F, (raw >> 4) & 0x0F
    values = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1)

    b_out, b_in = out_dim // FP8_BLOCK, in_dim // FP8_BLOCK
    values = values.view(b_out, FP8_BLOCK, b_in, FP8_BLOCK).transpose(1, 2)
    blocked = scale.float().view(b_out, FP8_BLOCK, b_in, -1).transpose(1, 2).flatten(2)
    tile_scale = blocked.amax(dim=-1, keepdim=True) / (2 ** MAX_OFFSET_BITS)
    offset = (blocked / tile_scale).unflatten(-1, (FP8_BLOCK, -1)).repeat_interleave(
        FP4_BLOCK, dim=-1
    )
    mantissa = (values * offset).to(torch.float8_e4m3fn).float()
    return (mantissa * tile_scale.unsqueeze(-1)).transpose(1, 2).reshape(out_dim, in_dim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument("slice_dir", type=Path)
    parser.add_argument("--layers", default="0,2,3")
    parser.add_argument("--samples", type=int, default=4,
                        help="expert tensors per layer to cross-check")
    args = parser.parse_args()
    source_layers = [int(x) for x in args.layers.split(",")]

    config = json.loads((args.slice_dir / "config.json").read_text())
    index = json.loads((args.shard_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    hidden = config["hidden_size"]
    vocab = config["vocab_size"]
    keep = config["n_routed_experts"]
    failures: list[str] = []

    print("=== 1. structure ===")
    with safe_open(str(args.slice_dir / "model.safetensors"), "pt") as sliced:
        names = list(sliced.keys())
        non_finite, wrong_dtype = [], []
        for name in names:
            tensor = sliced.get_tensor(name)
            if name.endswith("tid2eid"):
                if tensor.dtype != torch.int64:
                    wrong_dtype.append(f"{name}:{tensor.dtype}")
                continue
            if tensor.dtype != torch.bfloat16:
                wrong_dtype.append(f"{name}:{tensor.dtype}")
            if not torch.isfinite(tensor.float()).all():
                non_finite.append(name)
        print(f"  tensors            {len(names)}")
        print(f"  non-finite         {len(non_finite)} {non_finite[:3]}")
        print(f"  unexpected dtype   {len(wrong_dtype)} {wrong_dtype[:3]}")
        if non_finite or wrong_dtype:
            failures.append("structure: non-finite or wrong-dtype tensors")

        expected = {
            "embed.weight": (vocab, hidden),
            "head.weight": (vocab, hidden),
            "norm.weight": (hidden,),
        }
        for name, shape in expected.items():
            got = tuple(sliced.get_tensor(name).shape)
            status = "ok" if got == shape else "MISMATCH"
            print(f"  {name:20} {str(got):22} {status}")
            if got != shape:
                failures.append(f"structure: {name} {got} != {shape}")

        for layer in range(config["num_hidden_layers"]):
            gate = tuple(sliced.get_tensor(f"layers.{layer}.ffn.gate.weight").shape)
            n_expert = len({
                n.split(".")[4] for n in names
                if n.startswith(f"layers.{layer}.ffn.experts.")
            })
            ok = gate == (keep, hidden) and n_expert == keep
            print(f"  layer {layer}: gate {str(gate):16} experts {n_expert:<5} "
                  f"{'ok' if ok else 'MISMATCH'}")
            if not ok:
                failures.append(f"structure: layer {layer} router/expert count")

        print()
        print("=== 2. dequantization vs official convert.py ===")
        for destination, layer in enumerate(source_layers):
            shard = weight_map[f"layers.{layer}.ffn.experts.0.w1.weight"]
            with safe_open(str(args.shard_dir / shard), "pt") as raw:
                for expert in range(min(args.samples, keep)):
                    base = f"layers.{layer}.ffn.experts.{expert}.w1"
                    reference = official_fp4_reference(
                        raw.get_tensor(base + ".weight"), raw.get_tensor(base + ".scale")
                    )
                    emitted = sliced.get_tensor(
                        f"layers.{destination}.ffn.experts.{expert}.w1.weight"
                    ).float()
                    delta = (reference - emitted).abs().max().item()
                    if expert == 0:
                        print(f"  layer {layer}->{destination} expert {expert} "
                              f"max|diff| = {delta}")
                    if delta != 0.0:
                        failures.append(
                            f"dequant: layer {layer} expert {expert} differs by {delta}"
                        )
        print(f"  cross-checked {min(args.samples, keep)} experts x "
              f"{len(source_layers)} layers, all exact"
              if not failures else "  MISMATCHES FOUND")

        print()
        print("=== 3. routing table ===")
        for layer in range(config["num_hidden_layers"]):
            name = f"layers.{layer}.ffn.gate.tid2eid"
            if name not in names:
                print(f"  layer {layer}: topk router (no table)")
                continue
            table = sliced.get_tensor(name)
            in_range = bool(((table >= 0) & (table < keep)).all())
            distinct = bool(
                (table.sort(dim=1).values.diff(dim=1) > 0).all()
            )
            print(f"  layer {layer}: shape {tuple(table.shape)} "
                  f"in_range={in_range} rows_distinct={distinct}")
            if not (in_range and distinct):
                failures.append(f"routing: layer {layer} table invalid")
            if tuple(table.shape) != (vocab, config["num_experts_per_tok"]):
                failures.append(f"routing: layer {layer} table shape")

    print()
    if failures:
        print("FAILED")
        for failure in failures:
            print("  -", failure)
        raise SystemExit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
