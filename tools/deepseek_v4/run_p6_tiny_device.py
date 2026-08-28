#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the synthetic structural DeepSeek-V4 model with TorchNeuron Native.

This diagnostic intentionally uses no vLLM import or checkpoint. It verifies
that the portable mHC, attention, compressor and MoE primitives execute through
the Native compiler, but it does not claim scheduler integration.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch_neuronx


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def compile_stack_version() -> str:
    """Report the installed Neuron compile-stack version.

    TorchNeuron Native is supplied by the external ``torch-neuronx`` package.
    """
    for name in ("torch-neuronx",):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not-installed"


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return "not-on-PATH"
    return (result.stdout + result.stderr).strip()


def load_tiny_model_class():
    """Load the model without importing vllm_neuron's vLLM-dependent root."""
    package_paths = {
        "vllm_neuron": REPOSITORY_ROOT / "vllm_neuron",
        "vllm_neuron.model": REPOSITORY_ROOT / "vllm_neuron" / "model",
        "vllm_neuron.model.deepseek_v4": (
            REPOSITORY_ROOT / "vllm_neuron" / "model" / "deepseek_v4"
        ),
    }
    for name, path in package_paths.items():
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    module = importlib.import_module("vllm_neuron.model.deepseek_v4.tiny_model")
    return module.TinyDeepseekV4ForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    torch.manual_seed(7)
    model_class = load_tiny_model_class()
    cpu_model = model_class().eval()
    input_ids = torch.tensor(args.tokens)
    with torch.no_grad():
        expected, _ = cpu_model(input_ids)
    device = torch.device("neuron:0")
    device_model = torch.compile(
        copy.deepcopy(cpu_model).to(device),
        backend="neuron",
        fullgraph=True,
        options={"model_name": "deepseek_v4_structural_tiny"},
    )
    started = time.monotonic()
    with torch.no_grad():
        actual, state = device_model(input_ids.to(device))
    torch_neuronx.synchronize()
    wall_seconds = time.monotonic() - started
    actual = actual.cpu()
    absolute = (actual - expected).abs()
    passed = bool(torch.allclose(actual, expected, rtol=0.025, atol=0.025))

    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "tensors.npz",
        input_ids=input_ids.numpy(),
        expected_fp32=expected.detach().numpy(),
        actual=actual.detach().numpy(),
    )
    artifact = {
        "gate": "P6-direct-torch-neuronx-native-device-sub-gate",
        "scope": "synthetic compiled tiny model; no vLLM and no model weights",
        "does_not_prove": [
            "scheduler-driven cache I/O", "performance",
        ],
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "torch": torch.__version__,
        "torch_neuronx": compile_stack_version(),
        "neuron_ls": command_output(["neuron-ls"]),
        "device": str(device),
        "tokens": args.tokens,
        "shape": list(actual.shape),
        "state_tokens": state.num_tokens,
        "finite": bool(torch.isfinite(actual).all()),
        "max_absolute_error": float(absolute.max().item()),
        "rtol": 0.025,
        "atol": 0.025,
        "wall_seconds": wall_seconds,
        "passed": passed,
    }
    (args.output / "result.json").write_text(json.dumps(artifact, indent=2) + "\n")
    if not artifact["passed"] or not artifact["finite"]:
        raise SystemExit("P6 tiny-model device comparison failed")


if __name__ == "__main__":
    main()
