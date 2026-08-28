# SPDX-License-Identifier: Apache-2.0
"""vLLM Neuron plugin module."""

import glob
import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from vllm_neuron import envs as envs

# Initialize logging early so VLLM_NEURON_LOG_LEVEL takes effect
from vllm_neuron.logging_config import setup_logging as _setup_logging

_setup_logging()
import logging

logger = logging.getLogger(__name__)
# Enable prometheus multiprocess mode so that metrics observed in vLLM Neuron
# EngineCore/Worker processes (e.g. scheduler padding metrics) are written to
# shared mmap files and visible at the API server's /metrics endpoint.
# Note: Prometheus requires PROMETHEUS_MULTIPROC_DIR to be set before
# prometheus_client is imported anywhere. Typically a user would set the env
# var manually, but we set it during package init to set this up for the user.
# In case prometheus_client is imported before vLLM Neuron, users can
# set PROMETHEUS_MULTIPROC_DIR manually in their env to enable metrics.
if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
    import tempfile

    os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(
        prefix="vllm_neuron_prometheus_"
    )

# Suppress PyTorch kernel override warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Overriding a previously registered kernel.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*other operators may also be overridden.*"
)


def _is_neuron_dev() -> bool:
    """Detect Neuron device by checking for /dev/neuron* devices."""
    neuron_devices = glob.glob("/dev/neuron*")
    return len(neuron_devices) > 0


def _is_cpu_mode() -> bool:
    """Check if VLLM_NEURON_CPU_MODE is enabled via environment variable."""
    return os.environ.get("VLLM_NEURON_CPU_MODE", "0") == "1"


def _is_cpu_compile() -> bool:
    """Check if VLLM_NEURON_CPU_COMPILE is enabled via environment variable."""
    return os.environ.get("VLLM_NEURON_CPU_COMPILE", "0") == "1"


def _init_backend():
    """Initialize and validate the TorchNeuron Native runtime."""
    from vllm_neuron.backend import get_backend
    from vllm_neuron.native import (
        configure_cache_environment,
        reject_retired_configuration,
        require_native_capabilities,
    )

    get_backend()
    reject_retired_configuration()
    if envs.VLLM_NEURON_CPU_MODE and _is_cpu_compile():
        raise RuntimeError(
            "VLLM_NEURON_CPU_MODE and VLLM_NEURON_CPU_COMPILE are mutually exclusive"
        )
    if envs.VLLM_NEURON_CPU_MODE and os.environ.get("NKI_SIMULATOR") == "1":
        os.environ.setdefault("NKI_PRECISE_FP", "1")

    # Native caches must be local and rank isolated. Set these before importing
    # torch_neuronx because its HLO cache reads configuration at import time.
    rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    configure_cache_environment(envs.get_neuron_compile_cache_dir(), int(rank))
    if _is_cpu_compile():
        os.environ.setdefault("TORCH_NEURONX_COMPILE_ONLY", "1")

    require_native_capabilities()

    # Keep vLLM on its V1 scheduler; this plugin owns its model runner.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")


def register():
    """Register the Neuron platform if Neuron devices are present, else return None.

    An unset backend selector is preferred; ``neuron_native`` is accepted as a
    temporary alias. The former lite/XLA route is no longer available.
    """
    if not _is_cpu_mode() and not _is_cpu_compile() and not _is_neuron_dev():
        warnings.warn(
            "No Neuron devices found. Skipping Neuron plugin registration.",
            category=UserWarning,
        )
        return None

    from vllm_neuron.backend import get_platform_class
    from vllm_neuron.vllm.platform import _patch_dcp_config_validation

    _patch_dcp_config_validation()

    return get_platform_class()


def __getattr__(name):
    import importlib

    if name == "nn":
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Import-time backend init + patches run LAST, after `register` and
# `__getattr__` are bound. vLLM's plugin discovery
# (vllm.plugins.load_general_plugins) can re-enter this module while these
# imports execute; deferring them until `register` exists avoids a circular
# import crash ("partially initialized module 'vllm_neuron' has no attribute
# 'register'") on multi-worker device runs.
# Patches are applied at import time so they survive spawn-mode re-imports
# (the EngineCore subprocess never calls check_and_update_config).
try:
    _init_backend()
except (ImportError, KeyError):
    pass

# port_hold and pin_memory are registered against Phase.IMPORT; applying them
# through the registry keeps them in the same apply-once bookkeeping as every
# other patch. See vllm_neuron.vllm.patches.import_patches.
from vllm_neuron.vllm.patches import Phase, apply_phase

apply_phase(Phase.IMPORT)
