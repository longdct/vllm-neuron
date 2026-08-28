# SPDX-License-Identifier: Apache-2.0
import os
import shlex

# Memory map needed when compiling on a non Neuron instance
HBM_MEMORY_GB = {
    "trn1": 16,
    "trn1n": 16,
    "trn2": 24,
    "trn3": 36,
    "inf2": 16,
}


def _raise_no_platform():
    raise RuntimeError(
        "Failed to detect host Neuron platform. If compiling on a non Neuron "
        "instance, set the NEURON_PLATFORM_TARGET_OVERRIDE environment variable "
        "to specify the target platform. If you are on a Neuron instance, see "
        "the Neuron Runtime's troubleshooting guide for help on this topic: "
        "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/"
    )


def get_total_available_memory() -> int:
    if "NEURON_PLATFORM_TARGET_OVERRIDE" in os.environ:
        target = os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]
        if target not in HBM_MEMORY_GB:
            raise RuntimeError(
                f"VLLM_NEURON_CPU_COMPILE requires NEURON_PLATFORM_TARGET_OVERRIDE to be one of {list(HBM_MEMORY_GB.keys())}; got '{target}'"
            )
        else:
            return HBM_MEMORY_GB.get(target)
    else:
        raise RuntimeError(
            f"VLLM_NEURON_CPU_COMPILE requires NEURON_PLATFORM_TARGET_OVERRIDE to be set. Must be one of {list(HBM_MEMORY_GB.keys())}."
        )


def _get_target_from_nrt():
    try:
        import torch

        rt = torch.classes.neuron.Runtime()
        info = rt.get_instance_info()
        return info[0] + info[1]
    except Exception:
        _raise_no_platform()


def get_platform_target() -> str:
    """Auto-detect platform target. Precedence: env var > NRT."""
    if "NEURON_PLATFORM_TARGET_OVERRIDE" in os.environ:
        return os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]

    return _get_target_from_nrt()


def resolve_target(compiler_args=None) -> str:
    """Resolve --target with precedence: env var > compiler_args > NRT."""
    if "NEURON_PLATFORM_TARGET_OVERRIDE" in os.environ:
        return os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]

    if compiler_args is not None:
        if isinstance(compiler_args, str):
            compiler_args = shlex.split(compiler_args)
        if "--target" in compiler_args:
            return compiler_args[compiler_args.index("--target") + 1]

    return get_platform_target()


def get_torch_neuronx_version() -> str:
    """Get torch_neuronx framework version."""
    import torch_neuronx

    return torch_neuronx.__version__


def get_neuronxcc_version() -> str:
    """Get neuronxcc compiler version."""
    import neuronxcc

    return neuronxcc.__version__


def get_nki_version() -> str:
    """Get nki version."""
    import nki

    return nki._version.__version__


def get_server_prefix() -> str:
    """Derive a server-scoped prefix from NEURON_VISIBLE_DEVICES.

    Returns a string like "dev0_7" built from the first and last device in
    the NEURON_VISIBLE_DEVICES list.  This uniquely identifies a vLLM server
    on a given host, preventing compile cache collisions between independent
    server processes (e.g., prefill vs decode in DI mode, or non-MoE DP
    engines).

    Falls back to an empty string when the env var is not set (e.g., CPU mode
    or single-device), in which case the legacy "rank<N>" naming is used.
    """
    from vllm_neuron.utils.hardware_config import parse_range_list

    raw = os.environ.get("NEURON_VISIBLE_DEVICES")
    if not raw:
        return ""
    devices = parse_range_list(raw)
    if not devices:
        return ""
    return f"dev{devices[0]}_{devices[-1]}"
