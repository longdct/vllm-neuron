#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a tiny DeepSeek-V4 checkpoint from real official weights.

``build_tiny_checkpoint.py`` produces a *randomly initialised* three-layer model.
That is enough to find structural bugs but not numerical ones: random weights give
a near-degenerate logit distribution whose top-1/top-2 gaps are the same order as
BF16 noise, so "did the argmax survive?" carries almost no signal.

This builds the same shape of artifact from **real trained tensors**, keeping:

* ``embed.weight`` and the full output stack (``norm``, ``head``, ``hc_head_*``);
* one decoder layer of each attention type, renumbered to be contiguous;
* a subset of the routed experts, with the router remapped to match.

Everything is dequantized to BF16 and re-emitted under the checkpoint's own
native tensor names, so loading the result exercises the real
``weight_loaders.map_checkpoint_name`` path rather than bypassing it.

Note on fidelity: the routed-expert *weights* are real, but when ``--experts`` is
below ``n_routed_experts`` the hash-routing table is necessarily rewritten (see
``remap_tid2eid``). Both this plugin and the ``transformers`` reference read the
same emitted table, so oracle comparisons stay exact -- but the slice is not
token-for-token equivalent to the full model. Use ``--experts 0`` to keep all of
them when that matters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from quant_formats import dequantize, requantize_fp4_to_fp8

#: ``compress_ratios`` entry -> attention implementation.
RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}

#: Tensors outside ``layers.*`` that every slice carries. ``hc_head_*`` collapses
#: the ``hc_mult`` parallel mHC residual streams; without it there is no path from
#: the last decoder layer to the LM head.
OUTPUT_STACK = (
    "embed.weight",
    "norm.weight",
    "head.weight",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
)

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")

_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def remap_tid2eid(table: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Fold a hash-routing table into the first ``n_keep`` experts, keeping each
    row's entries distinct.

    ``DeepseekV4HashRouter`` does ``weights = scores.gather(1, indices)``, so a
    repeated id would apply one expert twice and renormalize against itself. A
    plain modulo collides often enough to matter, hence the linear probe.
    """
    top_k = table.shape[1]
    if n_keep < top_k:
        raise ValueError(f"cannot keep {n_keep} experts with top_k={top_k}")
    source = (table % n_keep).tolist()
    out = []
    for row in source:
        seen: set[int] = set()
        fixed = []
        for value in row:
            while value in seen:
                value = (value + 1) % n_keep
            seen.add(value)
            fixed.append(value)
        out.append(fixed)
    return torch.tensor(out, dtype=table.dtype)


def _shard_readers(shard_dir: Path, index: dict, names: set[str]) -> dict[str, str]:
    """Map each wanted tensor to its shard file, erroring on anything missing."""
    weight_map = index["weight_map"]
    missing = sorted(n for n in names if n not in weight_map)
    if missing:
        raise SystemExit(f"tensors absent from the checkpoint index: {missing[:8]}")
    for shard in {weight_map[n] for n in names}:
        if not (shard_dir / shard).is_file():
            raise SystemExit(
                f"shard {shard} not downloaded -- run fetch_official_shards.py first"
            )
    return {n: weight_map[n] for n in names}


def build(shard_dir: Path, output: Path, source_layers: tuple[int, ...],
          n_experts: int, expert_dtype: str = "bf16") -> None:
    index = json.loads((shard_dir / "model.safetensors.index.json").read_text())
    config = json.loads((shard_dir / "config.json").read_text())
    weight_map = index["weight_map"]
    ratios = config["compress_ratios"]
    n_routed = config["n_routed_experts"]
    n_hash_source = config.get("num_hash_layers", 0)
    keep_experts = n_routed if n_experts in (0, n_routed) else n_experts
    if keep_experts > n_routed:
        raise SystemExit(f"--experts {keep_experts} exceeds n_routed_experts={n_routed}")

    # Collect the source tensor names we need, then resolve them to shards once.
    wanted: set[str] = set(OUTPUT_STACK)
    for layer in source_layers:
        prefix = f"layers.{layer}."
        for name in weight_map:
            if not name.startswith(prefix):
                continue
            expert = _EXPERT_RE.match(name)
            if expert and int(expert.group(2)) >= keep_experts:
                continue  # dropped by the expert subset
            wanted.add(name)
    readers = _shard_readers(shard_dir, index, wanted)

    print(f"source layers {list(source_layers)} -> 0..{len(source_layers) - 1}")
    for destination, layer in enumerate(source_layers):
        kind = RATIO_TO_LAYER_TYPE.get(ratios[layer], f"ratio_{ratios[layer]}")
        router = "hash" if layer < n_hash_source else "topk"
        print(f"  layer {layer} -> {destination}  {kind:30} {router} router")
    print(f"experts: keeping {keep_experts} of {n_routed}")

    tensors: dict[str, torch.Tensor] = {}
    open_files: dict[str, object] = {}
    try:
        for shard in sorted({readers[n] for n in wanted}):
            open_files[shard] = safe_open(str(shard_dir / shard), "pt")

        def source(name: str) -> torch.Tensor:
            return open_files[readers[name]].get_tensor(name)

        def emit(name: str, destination_name: str) -> None:
            scale_name = name.rsplit(".", 1)[0] + ".scale"
            scale = source(scale_name) if scale_name in readers else None
            if expert_dtype == "fp8" and _EXPERT_RE.match(name) is not None:
                if scale is None:
                    raise SystemExit(f"{name} has no scale to requantize from")
                # MXFP4 -> FP8 with one scale per output channel: exact, and the
                # granularity `shard_on_i` consumes. The emitted names match the
                # official ones so the loader's ordinary expert path stacks
                # them; only the dtype and the scale's shape change.
                elements, channel_scale = requantize_fp4_to_fp8(source(name), scale)
                tensors[destination_name] = elements.contiguous()
                tensors[destination_name.rsplit(".", 1)[0] + ".scale"] = (
                    channel_scale.contiguous()
                )
                return
            tensors[destination_name] = dequantize(source(name), scale).contiguous()

        for name in OUTPUT_STACK:
            emit(name, name)

        for destination, layer in enumerate(source_layers):
            prefix, new_prefix = f"layers.{layer}.", f"layers.{destination}."
            for name in sorted(n for n in wanted if n.startswith(prefix)):
                if name.endswith(".scale"):
                    continue  # consumed alongside its weight
                target = new_prefix + name[len(prefix):]

                if name.endswith(".ffn.gate.tid2eid"):
                    table = source(name)
                    tensors[target] = (
                        table if keep_experts == n_routed
                        else remap_tid2eid(table, keep_experts)
                    )
                elif name.endswith(".ffn.gate.weight") or name.endswith(".ffn.gate.bias"):
                    # Router rows are per-expert and must be sliced in lockstep
                    # with the expert set, or scores index absent experts.
                    tensors[target] = source(name)[:keep_experts].contiguous().to(
                        torch.bfloat16
                    )
                else:
                    emit(name, target)
    finally:
        for handle in open_files.values():
            handle.__exit__(None, None, None)

    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.safetensors"
    save_file(tensors, str(model_path))

    slim = dict(config)
    slim["num_hidden_layers"] = len(source_layers)
    # Keep the official spelling: ``compress_ratios`` alone, with no derived
    # ``layer_types``/``compress_rates``. The plugin's normalizer resolves
    # per-layer structure from it, so the slice stays a faithful miniature of the
    # real config rather than one padded with keys upstream does not ship.
    slim["compress_ratios"] = [ratios[layer] for layer in source_layers]
    slim["n_routed_experts"] = keep_experts
    slim["num_hash_layers"] = sum(1 for layer in source_layers if layer < n_hash_source)
    slim["dtype"] = "bfloat16"
    slim["torch_dtype"] = "bfloat16"
    # Weights are dequantized now; leaving these would make the loader look for
    # scale tensors that no longer exist.
    slim.pop("quantization_config", None)
    slim.pop("expert_dtype", None)
    if expert_dtype == "fp8":
        # Everything except the routed experts is dequantized, so the official
        # quantization_config would misdescribe the file. Record the one thing
        # that is still quantized, for a reader; the runtime learns it from
        # neuron_config.quantization rather than from here.
        slim["expert_dtype"] = "fp8"
    (output / "config.json").write_text(json.dumps(slim, indent=2) + "\n")

    for filename in TOKENIZER_FILES:
        candidate = shard_dir / filename
        if candidate.is_file():
            shutil.copy2(candidate, output / filename)

    total = sum(t.numel() * t.element_size() for t in tensors.values())
    tensor_dtypes: dict[str, dict[str, int]] = {}
    for tensor in tensors.values():
        dtype = str(tensor.dtype).removeprefix("torch.")
        summary = tensor_dtypes.setdefault(dtype, {"tensor_count": 0, "bytes": 0})
        summary["tensor_count"] += 1
        summary["bytes"] += tensor.numel() * tensor.element_size()
    manifest = {
        "schema_version": 1,
        "kind": "deepseek_v4_official_weight_slice",
        "source_directory": str(shard_dir.resolve()),
        "source_config_sha256": sha256_file(shard_dir / "config.json"),
        "source_index_sha256": sha256_file(
            shard_dir / "model.safetensors.index.json"
        ),
        "source_experts": n_routed,
        "experts": keep_experts,
        "weight_dtype": "bfloat16",
        "tensor_dtypes": tensor_dtypes,
        "layers": [
            {
                "source_layer": layer,
                "destination_layer": destination,
                "compress_ratio": ratios[layer],
                "attention_type": RATIO_TO_LAYER_TYPE.get(
                    ratios[layer], f"ratio_{ratios[layer]}"
                ),
                "router_type": "hash" if layer < n_hash_source else "topk",
            }
            for destination, layer in enumerate(source_layers)
        ],
        "tensor_count": len(tensors),
        "parameter_bytes": total,
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
    }
    (output / "slice-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"wrote {len(tensors)} tensors, {total / 1e9:.2f} GB -> {output}")
    print(f"manifest and model checksum -> {output / 'slice-manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", type=Path, help="output of fetch_official_shards.py")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--expert-dtype",
        choices=("bf16", "fp8"),
        default="bf16",
        help=(
            "Storage for the routed experts. 'fp8' widens the checkpoint's "
            "MXFP4 exactly and emits one scale per output channel, halving the "
            "checkpoint and matching what quantization='fp8' expects at "
            "runtime. Everything else is dequantized to bf16 either way."
        ),
    )
    parser.add_argument("--layers", default="0,2,3")
    parser.add_argument(
        "--experts", type=int, default=32,
        help="routed experts to keep, or 0 for all (0 needs no router remap)",
    )
    args = parser.parse_args()
    layers = tuple(int(x) for x in re.split(r"[,\s]+", args.layers.strip()) if x)
    build(args.shard_dir, args.output, layers, args.experts, args.expert_dtype)


if __name__ == "__main__":
    main()
