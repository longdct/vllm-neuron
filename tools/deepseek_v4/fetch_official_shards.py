#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download only the safetensors shards a tiny DeepSeek-V4 slice needs.

The official checkpoint is 159.6 GB across 46 shards, but a slice that keeps the
embedding, a handful of decoder layers, and the output stack touches only a few
of them -- layers do not straddle shards. Resolving
``model.safetensors.index.json`` first turns a 159.6 GB pull into roughly 17 GB.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

#: Tensors that live outside ``layers.*`` and are required by any slice.
#: ``hc_head_*`` is the mHC stream collapse: the model expands the embedding into
#: ``hc_mult`` parallel residual streams and this is the only module that folds
#: them back to one before the final norm, so a slice without it cannot reach the
#: LM head at all.
OUTPUT_STACK = (
    "embed.weight",
    "norm.weight",
    "head.weight",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
)

#: Non-weight files needed to load the slice as a model directory.
AUXILIARY_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
)


def shards_for(index: dict, layers: tuple[int, ...]) -> tuple[set[str], list[str]]:
    """Return (shard filenames, missing tensor names) for ``layers`` + output stack."""
    weight_map = index["weight_map"]
    shards: set[str] = set()
    missing: list[str] = []

    for name in OUTPUT_STACK:
        if name in weight_map:
            shards.add(weight_map[name])
        else:
            missing.append(name)

    for layer in layers:
        prefix = f"layers.{layer}."
        matched = [k for k in weight_map if k.startswith(prefix)]
        if not matched:
            missing.append(f"{prefix}*")
        shards.update(weight_map[k] for k in matched)

    return shards, missing


def layer_types(config: dict) -> dict[int, str]:
    """Map layer index -> attention type via ``compress_ratios``.

    The real config carries ``compress_ratios``, not ``layer_types``; the tiny
    synthetic checkpoint carries the reverse. Both spellings reach the plugin's
    normalizer, but only this one exists upstream.
    """
    ratio_names = {
        0: "sliding_attention",
        4: "compressed_sparse_attention",
        128: "heavily_compressed_attention",
    }
    ratios = config.get("compress_ratios") or []
    return {i: ratio_names.get(r, f"ratio_{r}") for i, r in enumerate(ratios)}


def fetch(repo: str, layers: tuple[int, ...], destination: Path, revision: str) -> Path:
    index_path = Path(
        hf_hub_download(
            repo, "model.safetensors.index.json", revision=revision, local_dir=destination
        )
    )
    index = json.loads(index_path.read_text())
    config_path = Path(
        hf_hub_download(repo, "config.json", revision=revision, local_dir=destination)
    )
    config = json.loads(config_path.read_text())

    shards, missing = shards_for(index, layers)
    if missing:
        raise SystemExit(f"tensors absent from {repo} index: {missing}")

    total = index.get("metadata", {}).get("total_size", 0)
    names = layer_types(config)
    print(f"{repo} @ {revision}")
    print(f"  checkpoint: {total / 1e9:.1f} GB across "
          f"{len(set(index['weight_map'].values()))} shards")
    for layer in layers:
        print(f"  layer {layer:<3} {names.get(layer, 'unknown')}")
    print(f"  fetching {len(shards)} shard(s):")
    for shard in sorted(shards):
        print(f"    {shard}")

    snapshot_download(
        repo,
        revision=revision,
        local_dir=destination,
        allow_patterns=[*sorted(shards), *AUXILIARY_FILES],
    )

    # snapshot_download is resumable and silently skips complete files, so verify
    # rather than trust: a truncated shard would surface much later as a confusing
    # dequantization error.
    for shard in sorted(shards):
        path = destination / shard
        if not path.is_file():
            raise SystemExit(f"shard missing after download: {path}")
    print(f"  -> {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repo", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--layers",
        default="0,2,3",
        help="comma-separated source layer indices (default 0,2,3: one per "
        "attention type -- sliding, compressed_sparse, heavily_compressed)",
    )
    args = parser.parse_args()
    layers = tuple(int(x) for x in re.split(r"[,\s]+", args.layers.strip()) if x)
    fetch(args.repo, layers, args.destination, args.revision)


if __name__ == "__main__":
    main()
