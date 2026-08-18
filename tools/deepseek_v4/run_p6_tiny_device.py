#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the synthetic structural DeepSeek-V4 model eagerly on Trainium.

This diagnostic intentionally uses no vLLM import or checkpoint. It verifies
that the portable mHC, attention, compressor and MoE primitives execute through
Torch-XLA, but it does not claim captured-graph or scheduler integration.
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
import torch_xla
import torch_xla.runtime as xr


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def compile_stack_version() -> str:
    """Report the installed Neuron compile-stack version.

    Upstream ships this under ``torch-neuronx``. On the 0.24 plugin base it
    lives in ``libtorch-neuronx-lite`` instead (unpinned, resolved against
    vLLM's torch -- see requirements/core.txt), so ``torch-neuronx`` is not
    installed at all in that environment. Try both rather than hard failing
    the whole run after the device work is already done.
    """
    for name in ("torch-neuronx", "libtorch-neuronx-lite"):
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
    device = torch_xla.device()
    if xr.device_type() != "NEURON":
        # This SDK's torch_xla build hardcodes `_found_libneuronxla = False`
        # ("Neuron library initialization is handled by neuronx-cc package
        # directly") -- the classic PJRT auto-detection this script relies on
        # to reach Trainium is compiled out. `torch_xla.device()` silently
        # resolves to the CPU pseudo-device instead of raising, so without this
        # check the run below would "pass" having executed nothing on
        # hardware. Fail loudly rather than mislabel a CPU run as this gate.
        raise SystemExit(
            "torch_xla resolved PJRT_DEVICE="
            f"{xr.device_type()!r}, not NEURON -- this environment cannot "
            "reach Trainium through torch_xla's eager device path. This is "
            "an SDK/environment gap, not a script bug: see "
            "docs/model-dev/deepseek-v4-024-device-validation.md Step 1."
        )
    device_model = copy.deepcopy(cpu_model).to(device)
    started = time.monotonic()
    with torch.no_grad():
        actual, state = device_model(input_ids.to(device))
    torch_xla.sync()
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
        "gate": "P6-direct-torch-xla-device-sub-gate",
        "scope": "synthetic eager tiny model; no vLLM and no model weights",
        "does_not_prove": [
            "captured graph execution", "scheduler-driven cache I/O", "performance",
        ],
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "torch": torch.__version__,
        "torch_xla": importlib.metadata.version("torch-xla"),
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
