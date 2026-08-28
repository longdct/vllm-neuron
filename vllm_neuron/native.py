# SPDX-License-Identifier: Apache-2.0
"""Small adapters around the public TorchNeuron Native surface."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class NativeCapabilityError(RuntimeError):
    """Raised when the installed TorchNeuron build lacks a required feature."""


_RETIRED_ENV_VARS = (
    "VLLM_NEURON_DISABLE_PARALLEL_TRACE",
    "NEURON_LIBTORCH_REMOTE_CACHE",
    "NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND",
)


def reject_retired_configuration() -> None:
    """Reject settings whose lite lifecycle has no Native equivalent."""
    configured = [name for name in _RETIRED_ENV_VARS if name in os.environ]
    configured.extend(
        name
        for name in os.environ
        if name.startswith("NEURON_LIBTORCH_") and name not in configured
    )
    if configured:
        names = ", ".join(sorted(set(configured)))
        raise ValueError(
            f"Retired lite configuration is set: {names}. TorchNeuron Native "
            "uses in-process warmup and its NEFF/HLO caches; remove these settings."
        )


def configure_cache_environment(cache_root: str, rank: int) -> Path:
    """Configure rank-isolated Native compiler caches before package import."""
    rank_root = Path(cache_root) / f"rank_{rank}"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(rank_root / "inductor")
    os.environ["TORCH_NEURONX_HLO_CACHE_DIR"] = str(rank_root / "hlo")
    os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = str(rank_root / "neff")
    os.environ.setdefault("TORCH_NEURONX_HLO_COMPILE_CACHE", "1")
    return rank_root


def native_compile_options(
    *, model_name: str, optimization_level: int | str, use_tensorizer: bool = False
) -> dict[str, Any]:
    """Translate project compile settings to TorchNeuron Native settings."""
    level = str(optimization_level).removeprefix("-O")
    if level not in {"0", "1", "2", "3"}:
        raise ValueError(f"Unsupported Neuron compiler optimization level: {level!r}")
    os.environ["NEURON_COMPILER_OPT_LEVEL"] = f"-O{level}"
    return {
        "model_name": model_name,
        "use_tensorizer_backend": bool(use_tensorizer),
    }


def require_native_capabilities() -> Any:
    """Import TorchNeuron and validate the capabilities used by this project."""
    try:
        import torch
        import torch_neuronx
        import torch_neuronx.distributed
        from torch_neuronx import nki_hop, utils
    except ImportError as exc:
        raise NativeCapabilityError(
            "TorchNeuron Native requires an externally installed torch-neuronx "
            "build compatible with the active PyTorch."
        ) from exc

    missing = []
    if "neuron" not in torch._dynamo.list_backends():
        missing.append("torch.compile backend neuron")
    if not hasattr(nki_hop, "wrap_nki"):
        missing.append("NKI higher-order operator")
    if not hasattr(utils, "get_platform_target"):
        missing.append("platform utilities")
    if not hasattr(torch_neuronx, "device_count"):
        missing.append("PrivateUse1 device API")
    if missing:
        raise NativeCapabilityError(
            "Installed torch-neuronx lacks required Native capabilities: "
            + ", ".join(missing)
        )
    return torch_neuronx
