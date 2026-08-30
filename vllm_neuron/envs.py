# SPDX-License-Identifier: Apache-2.0
"""
vLLM Neuron Environment Variables Configuration

This module provides centralized environment variable management for vLLM Neuron

All environment variables are:
- Lazily evaluated when accessed
- Type-safe with proper validation
- Prefixed with VLLM_NEURON_ for namespace isolation
"""

import functools
import logging
import os
import subprocess
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # Core System Variables
    VLLM_NEURON_CPU_MODE: bool = False
    VLLM_NEURON_CPU_COMPILE: bool = False
    VLLM_NEURON_LOG_LEVEL: str = "INFO"
    VLLM_NEURON_DEBUG_MODE: bool = False
    VLLM_NEURON_BARRIER_TIMEOUT: int = 3600
    # Renderer thread-pool size for the EPD Router's HF preprocessing offload.
    # vLLM defaults this to 1; >1 parallelizes the GIL-releasing native image
    # decode/transform. Only safe because the Router disables the mm processor
    # cache (vLLM forbids >1 workers with the cache enabled).
    VLLM_NEURON_EPD_RENDERER_WORKERS: int = 4
    # TODO: Remove VLLM_NEURON_SWITCH_CC and derive topology from instance type.
    VLLM_NEURON_SWITCH_CC: bool = False
    VLLM_NEURON_MIN_KV_BUDGET_GIB: float = 1.0
    VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION: float = 0.30
    VLLM_NEURON_WORKER_TERMINATION_TIMEOUT: int = 5
    VLLM_NEURON_MLP_FORCE_TKG: bool = False
    VLLM_NEURON_DISABLE_NKI_KERNELS: bool = False
    VLLM_NEURON_ENABLE_QWEN3_5_SCAN_KERNEL: bool = False
    VLLM_NEURON_VALIDATE_CACHE_METADATA: bool = False
    VLLM_NEURON_TINY_VALIDATION_DIR: Optional[str] = None
    VLLM_NEURON_SKIP_PREFILL_WARMUP: bool = False
    VLLM_NEURON_SKIP_DECODE_WARMUP: bool = False
    VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP: bool = False
    VLLM_NEURON_SKIP_ENCODER_WARMUP: bool = False
    # Force the STATIC FP8 (non-MX) attention path on TRN3 even when STATIC_MX
    # kernels are available. Used by FP8 model factories as an escape hatch.
    VLLM_NEURON_FORCE_STATIC_FP8: bool = False
    # Snapshot Capture Variables
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE: bool = False
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES: str = "4"
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT: str = "pt"
    VLLM_NEURON_EFA_INSTANCE_FAMILY: str = ""


def maybe_convert_bool(value: str | None) -> bool | None:
    """
    Safely convert string to boolean using numeric conversion.

    Args:
        value: String value to convert ("0", "1") or None

    Returns:
        True if value is "1", False if value is "0", None if value is None

    Raises:
        ValueError: If value cannot be converted to int

    Examples:
        >>> maybe_convert_bool("1")
        True
        >>> maybe_convert_bool("0")
        False
        >>> maybe_convert_bool(None)
        None
        >>> maybe_convert_bool("invalid")  # Raises ValueError
    """
    if value is None:
        return None
    return bool(int(value))


def maybe_convert_int(value: str | None) -> int | None:
    """
    Safely convert string to integer.

    Args:
        value: String value to convert or None

    Returns:
        Integer value if conversion successful, None if value is None

    Raises:
        ValueError: If value cannot be converted to int

    Examples:
        >>> maybe_convert_int("600")
        600
        >>> maybe_convert_int("0")
        0
        >>> maybe_convert_int(None)
        None
        >>> maybe_convert_int("invalid")  # Raises ValueError
    """
    if value is None:
        return None
    return int(value)


def maybe_convert_float(value: str | None) -> float | None:
    """Safely convert string to float."""
    if value is None:
        return None
    return float(value)


environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Core System Variables ==================
    # Enable CPU fallback mode instead of using Neuron accelerators
    "VLLM_NEURON_CPU_MODE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_CPU_MODE")) or False
    ),
    # Enable CPU graph capture and compilation
    "VLLM_NEURON_CPU_COMPILE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_CPU_COMPILE")) or False
    ),
    # Logging level for vLLM Neuron components
    "VLLM_NEURON_LOG_LEVEL": lambda: os.getenv("VLLM_NEURON_LOG_LEVEL", "INFO").upper(),
    # Enable debug mode for verbose output and additional diagnostics
    "VLLM_NEURON_DEBUG_MODE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DEBUG_MODE")) or False
    ),
    # tp_barrier timeout in seconds (raise for long serial warmups)
    "VLLM_NEURON_BARRIER_TIMEOUT": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_BARRIER_TIMEOUT")) or 3600
    ),
    # Renderer thread-pool size for the EPD Router's HF preprocessing offload.
    "VLLM_NEURON_EPD_RENDERER_WORKERS": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_EPD_RENDERER_WORKERS")) or 4
    ),
    # Minimum KV budget (GiB) guardrail
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": lambda: (
        maybe_convert_float(os.getenv("VLLM_NEURON_MIN_KV_BUDGET_GIB"))
        if os.getenv("VLLM_NEURON_MIN_KV_BUDGET_GIB") is not None
        else 1.0
    ),
    # KV cap fraction applied to GMU-scaled total HBM budget.
    # TODO: Interim global safety cap. Replace with a better compile-safe
    # estimator that adapts across model families and hardware generations.
    "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION": lambda: (
        maybe_convert_float(os.getenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION"))
        if os.getenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION") is not None
        else 0.30
    ),
    # Local cache directory for model checkpoints
    "VLLM_NEURON_CHECKPOINT_CACHE": lambda: os.getenv(
        "NXDI_CHECKPOINT_CACHE", "/tmp/vllm_neuron-checkpoints"
    ),
    # Golden cache directory (disk tier)
    "VLLM_NEURON_GOLDEN_CACHE_DIR": lambda: os.path.expandvars(
        os.getenv("VLLM_NEURON_GOLDEN_CACHE_DIR", "/tmp/vllm_neuron-goldens-$USER")
    ),
    # Golden cache S3 URI (secondary tier, empty = disabled)
    "VLLM_NEURON_S3_GOLDENS_URI": lambda: os.getenv("VLLM_NEURON_S3_GOLDENS_URI", ""),
    # TODO: Remove VLLM_NEURON_SWITCH_CC and derive topology from instance type.
    # When True, uses contiguous groups for 8x8 topology instead of custom TRN2 mesh.
    "VLLM_NEURON_SWITCH_CC": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SWITCH_CC")) or False
    ),
    # Timeout in seconds for worker termination (SIGTERM→SIGKILL).
    # Default is 5s (upstream vLLM uses 4s for workers, 5s for API servers).
    # Increase when system profiling is enabled (NEURON_RT_INSPECT_ENABLE=1),
    # as the Neuron runtime needs time to flush profiling data after SIGTERM.
    "VLLM_NEURON_WORKER_TERMINATION_TIMEOUT": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_WORKER_TERMINATION_TIMEOUT")) or 5
    ),
    # Force TKG mode for the NKI MLP kernel (bypass CTE). Workaround for
    # nkilib CTE nc_transpose FP8 dtype mismatch on gen3+ (trn3).
    "VLLM_NEURON_MLP_FORCE_TKG": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_MLP_FORCE_TKG")) or False
    ),
    # Disable NKI kernels — forces can_run_kernel() to return False
    "VLLM_NEURON_DISABLE_NKI_KERNELS": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_NKI_KERNELS")) or False
    ),
    # Opt in to the Qwen3.5 Gated DeltaNet chunk-scan NKI kernel, which is
    # correct in the NKI simulator but produces garbage on device -- see
    # docs/model-dev/qwen3-5-real-checkpoint-bringup.md. Off by default so the
    # model is slow and right rather than fast and wrong; set it to debug the
    # kernel, not to serve.
    "VLLM_NEURON_ENABLE_QWEN3_5_SCAN_KERNEL": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_ENABLE_QWEN3_5_SCAN_KERNEL"))
        or False
    ),
    # Expensive device-to-host validation for correctness qualification runs.
    "VLLM_NEURON_VALIDATE_CACHE_METADATA": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_VALIDATE_CACHE_METADATA")) or False
    ),
    # Optional directory for per-step CPU logits captured before sampling.
    "VLLM_NEURON_TINY_VALIDATION_DIR": lambda: os.getenv(
        "VLLM_NEURON_TINY_VALIDATION_DIR", None
    ),
    # Skip prefill warmup/compilation without requiring kv-transfer-config.
    # Useful for decode-only profiling workflows.
    "VLLM_NEURON_SKIP_PREFILL_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_PREFILL_WARMUP")) or False
    ),
    # Skip both prefill and decode (language-model) warmup without requiring a
    # transfer config. Used by EPD vision-only (VE) pools, which have no
    # language model (mirrors the prefill/decode levers).
    "VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP")) or False
    ),
    # Skip vision-encoder warmup without requiring a transfer config. Used by
    # EPD language-only (PD) pools, which have no vision encoder.
    "VLLM_NEURON_SKIP_ENCODER_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_ENCODER_WARMUP")) or False
    ),
    # Skip decode warmup/compilation without requiring kv-transfer-config.
    # Useful for prefill-only profiling workflows.
    "VLLM_NEURON_SKIP_DECODE_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_DECODE_WARMUP")) or False
    ),
    # Force the STATIC FP8 (non-MX) attention path on TRN3 even when STATIC_MX
    # kernels are available (FP8 model factory escape hatch).
    "VLLM_NEURON_FORCE_STATIC_FP8": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_FORCE_STATIC_FP8")) or False
    ),
    # ================== Snapshot Capture Variables ==================
    # Master switch for NRT-boundary input snapshot capture; off by default.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE"))
        or False
    ),
    # Comma-separated 0-based call indices to capture (-1 = all); unset selects
    # the first post-warmup call.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL", None
    ),
    # Comma-separated generated-token indices to capture (decode only); the
    # forward whose inputs *produce* the token is captured. Unset disables it.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN", None
    ),
    # Comma-separated request_ids to capture; fires on any forward whose batch
    # contains the request. Matched against the caller's base id, ignoring the
    # unique suffix vLLM appends. Unset disables it.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST", None
    ),
    # Hard cap on captured calls per process, so a broad rule cannot fill disk
    # or stall the engine. Defaults to 4; a non-positive or malformed override
    # falls back to that default.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES", "4"
    ),
    # Comma-separated tp_rank set to capture; unset captures all ranks (each
    # worker writes its own global-rank directory). Filtered on tp_rank, so
    # with DP>1 a value selects that tp position in every DP group.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS", None
    ),
    # On-disk artifact format the serialize op writes for each tensor{i}:
    # "pt" (pickled, any dtype) or "npy" (raw element bytes, value-preserving;
    # bf16/fp8 stored as |V2/|V1). "npy" pulls in a numpy read dependency on
    # replay. Defaults "pt". Validated against the write_tensors op's accepted
    # formats, so an unsupported value fails at startup.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT", "pt"
    ),
    # Explicit instance-family override. When set, get_instance_family()
    # returns this verbatim instead of resolving from the sysfs product_name
    # (which defaults any trn3* to trn3pds and any trn2* to trn2).
    "VLLM_NEURON_EFA_INSTANCE_FAMILY": lambda: os.getenv(
        "VLLM_NEURON_EFA_INSTANCE_FAMILY", ""
    ),
}


def __getattr__(name: str) -> Any:
    """
    Gets environment variables lazily.

    Args:
        name: Name of the environment variable to retrieve

    Returns:
        Value of the environment variable (type depends on variable)

    Raises:
        AttributeError: If environment variable name is not defined

    Examples:
        >>> import vllm_neuron.envs as envs
        >>> cpu_mode = envs.VLLM_NEURON_CPU_MODE  # Returns bool
        >>> log_level = envs.VLLM_NEURON_LOG_LEVEL  # Returns str
        >>> envs.VLLM_NEURON_NONEXISTENT  # Raises AttributeError
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Return list of available environment variables.

    Returns:
        List of all defined environment variable names

    Examples:
        >>> import vllm_neuron.envs as envs
        >>> variables = dir(envs)
        >>> 'VLLM_NEURON_CPU_MODE' in variables
        True
    """
    return list(environment_variables.keys())


def is_set(name: str) -> bool:
    """
    Check if an environment variable is explicitly set.

    Args:
        name: Name of the environment variable to check

    Returns:
        True if environment variable is explicitly set, False otherwise

    Raises:
        AttributeError: If environment variable name is not defined

    Examples:
        >>> import os
        >>> import vllm_neuron.envs as envs
        >>> os.environ['VLLM_NEURON_CPU_MODE'] = '1'
        >>> envs.is_set('VLLM_NEURON_CPU_MODE')
        True
        >>> envs.is_set('VLLM_NEURON_LOG_LEVEL')  # Not set, uses default
        False
        >>> envs.is_set('VLLM_NEURON_NONEXISTENT')  # Raises AttributeError
    """
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_native_backend() -> bool:
    """Validate that TorchNeuron Native is selected."""
    value = os.getenv("VLLM_NEURON_BACKEND", "").lower()
    if value == "vllm_neuron":
        raise ValueError(
            "VLLM_NEURON_BACKEND=vllm_neuron is retired; install torch-neuronx "
            "and unset VLLM_NEURON_BACKEND."
        )
    if value not in ("", "neuron_native"):
        raise ValueError(f"Unsupported VLLM_NEURON_BACKEND value: {value!r}")
    return True


def get_compile_backend_name() -> str:
    """Return the TorchNeuron Native torch.compile backend."""
    is_native_backend()
    return "neuron"


def get_dist_backend() -> str:
    """Dispatch CPU and Neuron collectives to their Native backends."""
    is_native_backend()
    return "cpu:gloo,neuron:neuron"


def get_neuron_compile_cache_dir() -> str:
    """Return the local compile cache directory derived from VLLM_CACHE_ROOT.

    If ``VLLM_CACHE_ROOT`` is explicitly set by the user, returns
    ``$VLLM_CACHE_ROOT/neuron/compile_cache`` unconditionally — the user's
    explicit choice is always respected.

    If ``VLLM_CACHE_ROOT`` is not set (default ``~/.cache/vllm``), probes
    the home directory filesystem type.  On multi-node clusters,the home directory
    can be NFS-mounted, which is incompatible with FileLock semantics.  In that case
    the function falls back to ``/tmp/vllm_neuron_wdir_$USER/neuron/compile_cache`` and logs a warning.

    Returns:
        str: Absolute path to the local Neuron compile cache directory.

    Examples:
        >>> import os
        >>> os.environ["VLLM_CACHE_ROOT"] = "/tmp/my_cache"
        >>> get_neuron_compile_cache_dir()
        '/tmp/my_cache/neuron/compile_cache'
    """
    # Cache by VLLM_CACHE_ROOT so the FS probe + NFS-fallback warning only
    # runs once per cache root per process.
    return _resolve_neuron_compile_cache_dir(os.environ.get("VLLM_CACHE_ROOT"))


@functools.lru_cache
def _resolve_neuron_compile_cache_dir(cache_root_override: Optional[str]) -> str:
    """Cached resolver for ``get_neuron_compile_cache_dir``.

    Keyed on the caller-supplied ``VLLM_CACHE_ROOT`` value so different
    overrides each get their own cache slot.
    """
    if cache_root_override is not None:
        # User made an explicit choice — respect it unconditionally.
        return os.path.join(cache_root_override, "neuron/compile_cache")

    # Probe the home directory to detect remote filesystem mounts (NFS or Lustre).
    # We probe ~ (guaranteed to exist) rather than ~/.cache/vllm (may not exist yet).
    if _path_is_remote_filesystem(os.path.expanduser("~")):
        fallback = os.path.join(
            os.path.expandvars("/tmp/vllm_neuron_wdir_$USER"), "neuron/compile_cache"
        )
        logger.warning(
            "Default compile cache path ~/.cache/vllm/neuron/compile_cache is on a remote filesystem (NFS or Lustre). "
            "Falling back to %s.",
            fallback,
        )
        return fallback

    return os.path.join(os.path.expanduser("~/.cache/vllm"), "neuron/compile_cache")


def _path_is_remote_filesystem(path: str) -> bool:
    """Return True if *path* is on a remote filesystem (NFS or Lustre).

    Args:
        path: An existing filesystem path to check.

    Returns:
        bool: True if the filesystem is NFS (any variant) or Lustre
        (including FSx for Lustre), False otherwise.
    """
    result = subprocess.check_output(
        ["stat", "-f", "-c", "%T", path],
        stderr=subprocess.STDOUT,
        text=True,
    )
    fs_type = result.strip().lower()
    return fs_type.startswith("nfs") or fs_type == "lustre"


def get_neuron_snapshot_dir() -> str:
    """Return the snapshot bundle root, derived like the compile cache dir.

    Snapshots reference the HLO/NEFF that live under the compile cache, so they
    resolve off the same ``VLLM_CACHE_ROOT`` base and inherit the same NFS/Lustre
    fallback. The two trees sit side by side (``neuron/compile_cache`` and
    ``neuron/snapshots``) under one root.

    Examples:
        >>> import os
        >>> os.environ["VLLM_CACHE_ROOT"] = "/tmp/my_cache"
        >>> get_neuron_snapshot_dir()
        '/tmp/my_cache/neuron/snapshots'
    """
    return _resolve_neuron_snapshot_dir(os.environ.get("VLLM_CACHE_ROOT"))


@functools.lru_cache
def _resolve_neuron_snapshot_dir(cache_root_override: Optional[str]) -> str:
    """Cached resolver for ``get_neuron_snapshot_dir``.

    Mirrors ``_resolve_neuron_compile_cache_dir``: an explicit ``VLLM_CACHE_ROOT``
    is honored unconditionally, otherwise the default home location is used with
    a ``/tmp`` fallback when home is on a remote filesystem.
    """
    if cache_root_override is not None:
        return os.path.join(cache_root_override, "neuron/snapshots")

    if _path_is_remote_filesystem(os.path.expanduser("~")):
        fallback = os.path.join(
            os.path.expandvars("/tmp/vllm_neuron_wdir_$USER"), "neuron/snapshots"
        )
        logger.warning(
            "Default snapshot path ~/.cache/vllm/neuron/snapshots is on a remote filesystem (NFS or Lustre). "
            "Falling back to %s.",
            fallback,
        )
        return fallback

    return os.path.join(os.path.expanduser("~/.cache/vllm"), "neuron/snapshots")
