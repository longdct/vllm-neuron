#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the software, hardware, and cache provenance for a device run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command(*args: str) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "not-found"
    return (result.stdout + result.stderr).strip()


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    cache_root = args.cache_root or Path(
        os.environ.get("VLLM_CACHE_ROOT", Path.home() / ".cache" / "vllm")
    )
    cache = cache_root / "neuron" / "compile_cache"
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in (
            "torch", "torch-xla", "libtorch-neuronx-lite", "torch-neuronx",
            "neuronx-cc", "neuronxcc", "nki", "vllm",
        )},
        "compiler": command(
            str(Path(sys.executable).with_name("neuronx-cc")),
            "--version",
        ),
        "runtime": command("neuron-ls"),
        "driver": command("modinfo", "neuron"),
        "instance": command("ec2-metadata", "--instance-type"),
        "neuron_ls_json": command("neuron-ls", "-j"),
        "neuron_visible_devices": os.environ.get("NEURON_VISIBLE_DEVICES"),
        "git_revision": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "vllm_cache_root": str(cache_root),
        "compile_cache": str(cache),
        "compile_cache_entries": (
            sum(1 for path in cache.rglob("*") if path.is_file())
            if cache.exists() else 0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
