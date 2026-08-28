# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import errno
import hashlib
import os
import socket
import time
import sys
import shutil
import logging
import json
import psutil
import copy
import re
from filelock import FileLock, Timeout
from typing import TYPE_CHECKING, Dict, List, Optional, Callable, Any

import torch
import torch.distributed as dist

from vllm_neuron import envs
from vllm_neuron.compile.platform import (
    get_neuronxcc_version,
    get_nki_version,
    get_torch_neuronx_version,
    resolve_target,
)

if TYPE_CHECKING:
    from vllm_neuron.compile.artifacts import CompilationArtifacts
    from vllm_neuron.compile.schema import ArtifactMetadata


logger = logging.getLogger(__name__)


# Configuration defaults
VLLM_NEURON_DEFAULT_POLL_INTERVAL = 1  # 1 second
_METADATA_READ_RETRIES = 10
_METADATA_READ_INTERVAL = 0.2  # seconds

# File name constants
VLLM_NEURON_COMPILATION_METADATA_FILE = ".compilation_metadata"
VLLM_NEURON_COMPILATION_COMPLETE_FILE = ".compilation_complete"
VLLM_NEURON_GRAPH_NEFF_FILE = "graph.neff"
VLLM_NEURON_GRAPH_HLO_FILE = "graph.hlo"
VLLM_NEURON_LOCK_FILE_SUFFIX = ".lock"
VLLM_NEURON_CACHE_ACCESSED_FILE = ".cache_accessed"


def get_neff_filename(cache_dir: str) -> str:
    """Return the NEFF filename for a given cache directory.

    Uses the hash (directory basename) in the filename so that
    NRT profiling tools can distinguish multiple NEFFs in a flat directory.
    """
    hash_key = os.path.basename(cache_dir.rstrip("/"))
    return f"graph_{hash_key}.neff"


def get_local(
    hash_key: str,
    local_cache_dir: str,
) -> Optional[CompilationArtifacts]:
    """Check if cached artifacts exist locally and return them (fast path only).

    Args:
        hash_key: Precomputed cache hash key
        local_cache_dir: Local cache root directory (never NFS)

    Returns:
        CompilationArtifacts if local cache hit, None otherwise
    """
    cache_dir = os.path.join(local_cache_dir, hash_key)

    if _is_cache_complete(cache_dir):
        # Mark cache access for CI observability (conftest detects cache hits
        # by checking .cache_accessed mtime changes between test boundaries).
        try:
            marker = os.path.join(cache_dir, VLLM_NEURON_CACHE_ACCESSED_FILE)
            with open(marker, "w") as f:
                f.write(str(int(time.time())))
                f.flush()
        except OSError:
            pass  # best-effort CI marker — compilation succeeded regardless

        return _build_cache_artifact(cache_dir, hash_key)

    logger.info(f"Local cache miss for key: {hash_key}")
    return None


def fetch_remote_or_compile(
    hash_key: str,
    local_cache_dir: str,
    compile_fn: Callable[[], CompilationArtifacts],
    timeout: int,
    invalidate: bool = False,
    remote_cache_dir: Optional[str] = None,
    assert_cache_hits: bool = False,
) -> CompilationArtifacts:
    """Populate the local cache on a miss: try remote fetch first, then compile.

    This function orchestrates the complete cache miss workflow in multi-process
    environments, ensuring that only one process populates the local cache for a
    given key while others wait for completion.

    Args:
        hash_key: Precomputed cache hash key
        compile_fn: Callable that returns CompilationArtifacts when invoked
        local_cache_dir: Local cache root directory (never NFS)
        timeout: Timeout in seconds for waiting for shared compilation
        invalidate: If True, forcefully recompile even on cache hit (default: False)
        remote_cache_dir: Optional path to NFS/FSx remote cache store.  When set,
            the lock winner attempts a remote fetch before falling back to
            compilation. (default: None)
        assert_cache_hits: If enabled, remote cache miss is treated as an error.

    Returns:
        CompilationArtifacts: Compiled artifacts
    """
    with _lock(hash_key, local_cache_dir) as lock_ctx:
        if lock_ctx.acquired():
            cache_dir = os.path.join(local_cache_dir, hash_key)

            # Optionally invalidate existing artifacts first
            if invalidate:
                _remove_completion_signal(hash_key, local_cache_dir)

            if not invalidate and _is_cache_complete(cache_dir):
                logger.info(
                    f"Lock acquired but local cache already populated: {hash_key}"
                )
                return _build_cache_artifact(cache_dir, hash_key)

            # Try remote fetch before compiling
            if remote_cache_dir and not invalidate:
                if _fetch_from_remote(hash_key, remote_cache_dir, local_cache_dir):
                    return _build_cache_artifact(cache_dir, hash_key)
                if assert_cache_hits:
                    raise RuntimeError(
                        f"VLLM_NEURON_ASSERT_CACHE_HIT: remote compile cache miss for key "
                        f"{hash_key}. All graphs must be a cache hit."
                    )

            # Fall back to local compilation
            artifacts = compile_fn()
            _store_local(hash_key, artifacts, local_cache_dir)
            return artifacts
        else:
            # Wait for the lock winner (fetching or compiling) to complete
            return _wait_for_completion(hash_key, local_cache_dir, timeout)


def save_cache(local_cache_dir: str, remote_cache_dir: str, hash_key: str) -> None:
    """Promote a locally-compiled cache entry to a shared remote cache directory.

    Call this function after compilation returns to make the
    newly-compiled Neuron artifacts available to other nodes or future runs.
    Internally it copies the local entry into a per-process staging directory
    inside ``remote_cache_dir`` and then atomically renames it to the final
    destination, ensuring that no reader ever sees a partially-written entry.

    This also promotes any locally-cached NKI compile results to the remote
    cache, so that other nodes skip NKI BIR compilation on warm start.

    This function is safe to call concurrently from multiple nodes targeting
    the same ``remote_cache_dir``:

    * If this node wins the race, the entry is promoted and a log message is
      emitted at INFO level.
    * If another node already promoted the same entry, this call is a silent
      no-op — the remote entry is left intact and no error is raised.

    If the local entry is not yet complete (missing the ``.compilation_complete``
    sentinel), a :exc:`RuntimeError` is raised to signal that compilation has
    not finished successfully.

    Args:
        local_cache_dir: Root directory of the local compilation cache.
            Obtain the correct value with
            :func:`vllm_neuron.envs.get_neuron_compile_cache_dir`.
        remote_cache_dir: Root directory of the shared remote cache that other
            nodes will read from.  Must be writable by this process.
        hash_key: The cache key identifying the specific compiled entry to
            promote.  Corresponds to a subdirectory name inside
            ``local_cache_dir``.

    Examples:
        After running inference, promote all locally-compiled entries to a
        shared remote cache so that other nodes can skip recompilation:

        >>> import os
        >>> from vllm import LLM, SamplingParams
        >>> from vllm_neuron.compile import save_cache
        >>> from vllm_neuron.envs import get_neuron_compile_cache_dir
        >>>
        >>> local_cache_dir = get_neuron_compile_cache_dir()
        >>> remote_cache_dir = "/mnt/fsx/shared-neuron-cache"
        >>>
        >>> llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", ...)
        >>> outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=10))
        >>>
        >>> # Promote locally-compiled entry to the shared remote cache.
        >>> save_cache(local_cache_dir, remote_cache_dir, hash_key)
    """
    entry_dir = os.path.join(local_cache_dir, hash_key)
    if not _is_cache_complete(entry_dir):
        raise RuntimeError(
            f"Cannot save cache: local entry is incomplete at {entry_dir}. "
            "Ensure compilation has completed successfully before calling save_cache()."
        )
    _promote_to_remote(hash_key, local_cache_dir, remote_cache_dir)

    # Also promote NKI compile cache entries (lazy import to avoid pulling
    # in the NKI SDK at module load time for environments without it).
    from vllm_neuron.nki.nki_cache import save_nki_cache_to_remote

    save_nki_cache_to_remote(local_cache_dir, remote_cache_dir)


def create_cache_hash(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    options: dict = {},
) -> str:
    """Create deterministic 32-character hash for shared cache scenarios.

    Generates a deterministic hash that includes graph structure,
    input tensor metadata, dependency versions, compiler args, and platform
    target for safe shared caching.
    The hash ensures that cached compilation artifacts are only reused when
    all relevant components (graph, inputs, dependencies, compiler flags) are identical.

    For multi-process scenarios, neuron device indices are normalized to neuron:0
    to ensure consistent hashing across different processes with varying device indices.

    Args:
        gm: The PyTorch FX GraphModule to hash
        example_inputs: Example inputs used for tracing the model
        options: Compilation options dict (may contain ``compiler_args`` and
            ``compiler_workdir``). Defaults to an empty dict.

    Returns:
        str: 32-character deterministic hash
    """
    from vllm_neuron.compile.backend import _parse_compiler_args

    normalized_gm = _normalize_neuron_devices_for_hashing(gm)

    normalized_gm = _add_kernel_backend_config_for_hashing(normalized_gm)

    hash_components = [str(normalized_gm.graph)]

    # Include resolved replica group ranks so that different process group
    # configurations produce distinct cache keys.
    replica_groups_str = _get_collective_replica_groups_for_hashing(normalized_gm)
    if replica_groups_str:
        hash_components.append(replica_groups_str)

    # Add input tensor metadata
    for inp in example_inputs:
        shape_str = "x".join(map(str, inp.shape))
        dtype_str = str(inp.dtype).split(".")[-1]
        stride_str = "x".join(map(str, inp.stride()))
        hash_components.append(f"{dtype_str}_{shape_str}_{stride_str}")

    # Add version information for cache safety
    torch_neuronx_version = get_torch_neuronx_version()
    neuronxcc_version = get_neuronxcc_version()
    nki_version = get_nki_version()
    hash_components.append(f"torch_neuronx:{torch_neuronx_version}")
    hash_components.append(f"neuronxcc:{neuronxcc_version}")
    hash_components.append(f"nki:{nki_version}")

    # Normalize compiler args into a canonical dict
    raw_compiler_args = options.get("compiler_args", "")
    compiler_args = _parse_compiler_args(raw_compiler_args)
    canonical = _normalize_compiler_args(compiler_args)
    # Add sorted canonical args to hash
    sorted_args_str = " ".join(f"{k}={v}" for k, v in sorted(canonical.items()))
    hash_components.append(f"compiler_args:{sorted_args_str}")

    combined_hash = hashlib.md5("|".join(hash_components).encode()).hexdigest()[:32]

    # Log the cache hash along with all components that made up this hash
    input_summaries = [
        f"{str(inp.dtype).split('.')[-1]}_{('x'.join(map(str, inp.shape)))}_{('x'.join(map(str, inp.stride())))}"
        for inp in example_inputs
    ]
    logger.debug(
        f"Compilation cache key: {combined_hash} | Components: "
        f"graph_len={len(str(normalized_gm.graph))}, "
        f"inputs={input_summaries}, "
        f"torch_neuronx={torch_neuronx_version}, "
        f"neuronxcc={neuronxcc_version}, "
        f"nki={nki_version}, "
        f"compiler_args_canonical={sorted_args_str}"
    )

    return combined_hash


def save_artifact_metadata(cache_dir: str, metadata: "ArtifactMetadata") -> None:
    """Save validated artifact metadata to cache directory.

    Args:
        cache_dir: Cache directory path
        metadata: Pydantic metadata object to save
    """
    if not metadata:
        return

    filepath = os.path.join(
        cache_dir, _get_artifact_metadata_filename(metadata.version)
    )

    try:
        with open(filepath, "w") as f:
            f.write(metadata.model_dump_json(indent=2))
            f.flush()
        logger.debug(f"Saved artifact metadata to {filepath}")
    except (OSError, TypeError) as e:
        logger.warning(f"Failed to save artifact metadata to {filepath}: {e}")


class CompilationLock:
    """Context manager for compilation locking"""

    def __init__(self, hash_key: str, local_cache_dir: str):
        self.hash_key = hash_key
        self.local_cache_dir = local_cache_dir
        self.cache_dir = os.path.join(local_cache_dir, hash_key)
        self.file_lock = None
        self._acquired = False
        self.metadata_file_path = os.path.join(
            local_cache_dir, f".{hash_key}{VLLM_NEURON_COMPILATION_METADATA_FILE}"
        )
        self.lock_file_path = self.metadata_file_path + VLLM_NEURON_LOCK_FILE_SUFFIX

    def __enter__(self) -> "CompilationLock":
        _del_stale_lock(self.hash_key, self.local_cache_dir)

        os.makedirs(self.local_cache_dir, exist_ok=True)
        self.file_lock = FileLock(self.lock_file_path)

        try:
            self.file_lock.acquire(timeout=0.001)
            self._acquired = True
            _write_compilation_metadata(self.metadata_file_path)
            logger.info(f"Acquired compilation lock for cache: {self.cache_dir}")
        except Timeout:
            # Another process has the lock
            self._acquired = False
            logger.debug(
                f"Failed to acquire lock - another process is compiling: {self.cache_dir}"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._acquired and self.file_lock:
            self.file_lock.release()
            logger.debug("Released compilation lock")

        if exc_type is not None:
            logger.error(
                f"Compilation failed - terminating process for cleanup: {exc_val}"
            )
            sys.exit(f"FATAL: {exc_val}")

    def acquired(self) -> bool:
        """Check if lock was successfully acquired."""
        return self._acquired


def _get_collective_replica_groups_for_hashing(
    gm: torch.fx.GraphModule,
) -> Optional[str]:
    """Run CollectiveReplicaGroupsPass on a copy and return a deterministic
    string encoding the resolved replica groups for all collective nodes.

    Returns None when torch.distributed is not initialised or no collectives
    are present.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return None

    try:
        from vllm_neuron.fx_passes.collective_replica_groups_pass import (
            CollectiveReplicaGroupsPass,
        )

        gm, metadata = CollectiveReplicaGroupsPass().run(gm)
        if metadata.get("resolved_count", 0) == 0:
            return None

        # Build a deterministic string from the resolved groups in node order.
        parts = []
        for node in gm.graph.nodes:
            groups = node.meta.get("replica_groups")
            if groups is not None:
                parts.append(f"{node.name}:{groups}")
        return "|".join(parts) if parts else None
    except Exception as e:
        logger.debug(f"Failed to resolve collective replica groups for hashing: {e}")
        return None


def _add_kernel_backend_config_for_hashing(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    from vllm_neuron.fx_passes.backend_config_pass import (
        NkiKernelWriteBackendConfigPass,
    )
    from vllm_neuron.nki.nki_hop import NKIKernelWrapper

    gm, _ = NkiKernelWriteBackendConfigPass().run(gm)

    # ``kernel_idx`` is an integer assigned by ``NKIRegistry`` in
    # registration order. The exact value depends on which kernels have
    # been registered so far in the running process — so the same FX
    # node ends up with a different ``kernel_idx`` value across phases
    # (e.g. fork-based parallel-trace child vs the parent's warmup) even
    # when the kernel itself is identical. The canonical kernel identity
    # lives in the ``backend_config`` blob (func_name, version, dtypes,
    # grid, etc.), which is already part of the hashed FX. Drop
    # ``kernel_idx`` before hashing so registry-order doesn't perturb
    # the cache key. Safe on the hashing copy: callers operate on a
    # deepcopy so the live graph is never mutated.
    #
    # We ALSO normalize the volatile ``klir_binary.binary`` artifact PATH
    # out of ``backend_config`` before hashing (defense-in-depth derived from
    # the same root cause below). The NKI SDK sets that field to the absolute
    # on-disk path of the compiled BIR artifact, whose location is keyed off the
    # NKI compile-cache key. For most kernels that key is deterministic, but for
    # a kernel that takes a compile-time config OBJECT (rotational_topk's
    # ``RotationalTopkConfig``), the NKI cache key embeds ``str(config)`` — and
    # that dataclass's repr includes its ``_shared_const_cache`` of
    # per-trace-random NamedTemporaryFile ``.npy`` paths (its custom __hash__
    # excludes them, but __repr__ does not). So the path can vary between the
    # first trace and the warmup re-trace, which would make the SAME bucket hash
    # to two different keys → cache miss → every NEFF compiled twice (the
    # "Expected 7 compilations, found 14" double-compile regression — the
    # on-device repro of which tracks to the ``_shared_const_cache`` channel
    # below; this path normalization closes the same leak at the backend_config
    # layer). The path is an artifact LOCATION, not kernel identity —
    # identity is fully captured by ``func_name``/``grid``/``mac_count`` in the
    # blob plus the input dtypes/shapes (example_inputs) and dep versions already
    # in the hash — so dropping it is safe and makes the key deterministic.
    #
    # The two volatile channels live on TWO DIFFERENT ``call_function`` nodes,
    # handled by the two branches below in a single pass:
    #   * Branch A — the NKI kernel-call node (``target`` is the NKIKernelWrapper
    #     singleton): drop ``kernel_idx`` and normalize ``backend_config``.
    #   * Branch B — the compile-time config-CONSTRUCTOR node (``target`` is the
    #     config class, e.g. ``RotationalTopkConfig``, that Dynamo reconstructs
    #     in the graph): blank its ``_shared_const_cache`` kwarg — a dict of
    #     per-trace-random NamedTemporaryFile ``.npy`` paths that ``str(gm.graph)``
    #     renders verbatim, leaking into the cache key exactly like the
    #     klir_binary path. The shared-constant file PATHS are not part of the
    #     kernel's semantic identity (the shapes/k/n_stages that determine the
    #     constants are), so blanking them is safe.
    # These node sets are DISJOINT — the kernel-call node has no
    # ``_shared_const_cache`` kwarg and the constructor node's target is a class,
    # not the NKIKernelWrapper instance — so the branches are mutually exclusive.
    # Branch B is intentionally NOT scoped to NKIKernelWrapper: that would skip
    # the constructor node entirely and reintroduce the double-compile regression
    # (the "Expected 7, found 14" recompile).
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if type(node.target) == NKIKernelWrapper:
            new_kwargs = dict(node.kwargs)
            new_kwargs.pop("kernel_idx", None)
            bc = new_kwargs.get("backend_config")
            if isinstance(bc, str) and bc:
                new_kwargs["backend_config"] = _normalize_backend_config_for_hashing(bc)
            node.kwargs = new_kwargs
        elif "_shared_const_cache" in node.kwargs:
            new_kwargs = dict(node.kwargs)
            new_kwargs["_shared_const_cache"] = {}
            node.kwargs = new_kwargs

    return gm


def _normalize_backend_config_for_hashing(backend_config_b64: str) -> str:
    """Strip the volatile NKI artifact path from a base64 ``backend_config``.

    Returns a deterministic string capturing the kernel's identity (everything
    in the blob except the non-deterministic ``klir_binary.binary`` path). On any
    decode/parse failure, returns the input unchanged so hashing never breaks.

    The decoded ``backend_config`` JSON (produced by the NKI SDK's
    ``build_config()``) looks like::

        {
          "func_name": "rotational_topk",
          "platform_target": "<nc target>",
          "kernel_format": "bir",
          "kernel_version": 1,
          "klir_binary": {
            "binary": "<absolute path to the on-disk NKI BIR artifact>",
            "input_names": ["in0"],
            "output_names": ["out0", "out1"],
            "version_identifier": "",
            "aliases": []
          },
          "grid": 1,
          "has_collectives": false,
          "mac_count": 12345
        }

    ``klir_binary.binary`` is the ONLY volatile field: the NKI SDK sets it to
    ``descriptor.kernel_json_path``, an absolute path to the on-disk BIR JSON
    artifact whose location varies per trace (see the leak explanation in
    ``_add_kernel_backend_config_for_hashing``). Everything else — ``func_name``,
    ``grid``, ``mac_count``, the input/output names — is stable kernel identity,
    so we pop only ``binary`` and keep the rest.
    """
    import base64

    try:
        decoded = base64.b64decode(backend_config_b64)
        config = json.loads(decoded)
        klir = config.get("klir_binary")
        if isinstance(klir, dict):
            # Drop the artifact path; keep the semantic input/output names.
            klir.pop("binary", None)
        # Canonical, order-independent serialization for a stable hash input.
        return json.dumps(config, sort_keys=True)
    except Exception:
        return backend_config_b64


def _normalize_neuron_devices_for_hashing(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Create a copy of GraphModule with all neuron:X devices normalized to neuron:0 for consistent hashing.

    This function creates a deep copy of the GraphModule and normalizes all 'neuron:X' device references
    to 'neuron:0' to ensure consistent cache hashing across different processes in multi-process setups.

    Args:
        gm: The PyTorch FX GraphModule to normalize

    Returns:
        torch.fx.GraphModule: A copy with normalized neuron devices
    """
    gm_copy = copy.deepcopy(gm)

    modifications_made = 0
    nodes_to_replace = []

    # First pass: identify nodes that need normalization
    for node in gm_copy.graph.nodes:
        if node.kwargs and "device" in node.kwargs:
            device = node.kwargs["device"]
            normalized_device = _normalize_device_reference(device)
            logger.debug(f"Normalized device: {device} -> {normalized_device}")

            if normalized_device != device:
                new_kwargs = dict(node.kwargs)
                new_kwargs["device"] = normalized_device
                nodes_to_replace.append((node, new_kwargs))

    # Second pass: replace nodes that need normalization
    for node, new_kwargs in nodes_to_replace:
        # Create new node with normalized device kwargs
        with node.graph.inserting_after(node):
            # Preserve the original FX op/target while only changing kwargs["device"].
            # This avoids op-specific branching and prevents mismatches such as
            # recreating call_method nodes as call_function nodes.
            new_node = node.graph.create_node(
                op=node.op,
                target=node.target,
                args=node.args,
                kwargs=new_kwargs,
                name=node.name,
                type_expr=node.type,
            )
            new_node.meta = dict(node.meta)
            node.replace_all_uses_with(new_node)

        # Remove the old node
        node.graph.erase_node(node)
        modifications_made += 1

    # Recompile the graph if modifications were made
    if modifications_made > 0:
        gm_copy.recompile()

    return gm_copy


def _normalize_device_reference(device) -> Any:
    """Normalize device references for consistent hashing.

    Handles string device references and torch.device objects.
    Converts neuron:X devices to neuron:0 for consistent hashing.

    When CPU Compilation mode is enabled, meta devices are converted
    to neuron:0 for consistent hashing.

    Args:
        device: Device information (str, torch.device, or other)

    Returns:
        Normalized device with neuron:X/meta converted to neuron:0
    """
    if device is None:
        return device

    # Convert device to string for normalization
    device_str = None
    if isinstance(device, str):
        device_str = device.strip()
    elif hasattr(device, "type"):
        # Handle torch.device objects
        device_str = str(device)
    elif hasattr(device, "__str__"):
        # Handle other device objects that can be converted to string
        device_str = str(device).strip()
    else:
        logger.debug(
            f"Device type {type(device)} not supported for normalization: {device}"
        )
        return device

    logger.debug(
        f"Normalizing device string: '{device_str}' (original type: {type(device)})"
    )

    if device_str:
        # Convert meta device to neuron:0 for consistent hashing
        # when CPU compilation is enabled.
        if re.match(r"^meta(:\d+)?$", device_str) and envs.VLLM_NEURON_CPU_COMPILE:
            normalized = "neuron:0"
        else:
            # Use regex to match neuron:X pattern and replace with neuron:0
            normalized = re.sub(r"^neuron:\d+$", "neuron:0", device_str)

        # If we had a torch.device object originally, convert back to torch.device
        if hasattr(device, "type") and isinstance(device, torch.device):
            if normalized != device_str:
                return torch.device(normalized)

        return normalized

    return device


def _lock(hash_key: str, local_cache_dir: str) -> "CompilationLock":
    """Get a compilation lock for the given hash.

    Args:
        hash_key: Cache hash key
        local_cache_dir: Local cache root directory (never NFS)

    Returns:
        CompilationLock context manager
    """
    return CompilationLock(hash_key, local_cache_dir)


def _store_local(
    hash_key: str, artifacts: CompilationArtifacts, local_cache_dir: str
) -> None:
    """Store compilation artifacts in the local cache directory.

    Args:
        hash_key: Cache hash key
        artifacts: Compilation artifacts to store
        local_cache_dir: Local cache root directory (never NFS)
    """
    cache_dir = os.path.join(local_cache_dir, hash_key)

    # 1. Verify that all required files exists
    # Verify HLO file exists (written by neuronx-cc into the workdir)
    hlo_path = os.path.join(cache_dir, VLLM_NEURON_GRAPH_HLO_FILE)
    if not os.path.exists(hlo_path):
        raise FileNotFoundError(
            f"Compilation completed but HLO file not found: {hlo_path}"
        )

    # Verify NEFF file exists (written by neuronx-cc into the workdir)
    if not os.path.exists(artifacts.neff_filename):
        raise FileNotFoundError(
            f"Compilation completed but NEFF file not found: {artifacts.neff_filename}"
        )

    # 2. Store artifact metadata
    save_artifact_metadata(cache_dir, artifacts.metadata)

    # 3. Write .compilation_complete LAST — signals all waiters that all prior
    #    files are fully written and coherent on this host.
    complete_file = os.path.join(cache_dir, VLLM_NEURON_COMPILATION_COMPLETE_FILE)
    completion_info = f"completed:{time.time()}\n"
    completion_info += f"neff_size:{os.path.getsize(artifacts.neff_filename)}\n"
    if dist.is_initialized():
        completion_info += f"completed_by_rank:{dist.get_rank()}\n"

    with open(complete_file, "w") as f:
        f.write(completion_info)
        f.flush()

    logger.info(f"Stored compilation artifacts in local cache: {cache_dir}")


def _fetch_from_remote(
    hash_key: str,
    remote_cache_dir: str,
    local_cache_dir: str,
) -> bool:
    """Fetch compilation artifacts from remote.

    Args:
        hash_key: Cache hash key identifying the artifact set to fetch.
        remote_cache_dir: Path to the NFS/FSx remote cache root directory.
        local_cache_dir: Local cache root directory (never NFS).

    Returns:
        bool: ``True`` if artifacts are now present in the local cache;
              ``False`` if the remote directory does not exist, is incomplete,
              or an unrecoverable I/O error occurred.
    """
    remote_dir = os.path.join(remote_cache_dir, hash_key)
    local_dir = os.path.join(local_cache_dir, hash_key)
    tmp_local_dir = local_dir + f".tmp.{os.getpid()}"

    try:
        if not os.path.exists(remote_dir):
            logger.info(
                f"Remote cache miss for key {hash_key} — remote directory does not exist: {remote_dir}"
            )
            return False

        if not _is_cache_complete(remote_dir):
            logger.info(
                f"Remote cache miss for key {hash_key} — remote artifacts incomplete: {remote_dir}"
            )
            return False

        logger.info(f"Remote cache hit for key {hash_key} — copying from {remote_dir}")
        shutil.copytree(
            remote_dir,
            tmp_local_dir,
            ignore=shutil.ignore_patterns(
                VLLM_NEURON_COMPILATION_METADATA_FILE,
                VLLM_NEURON_COMPILATION_METADATA_FILE + VLLM_NEURON_LOCK_FILE_SUFFIX,
                VLLM_NEURON_CACHE_ACCESSED_FILE,
            ),
        )

        # If a partial workdir exists from a previous failed compilation, move it
        # aside atomically before placing the complete staged artifacts.
        trash_dir = local_dir + f".trash.{os.getpid()}"
        if os.path.exists(local_dir):
            os.rename(local_dir, trash_dir)

        # Atomically place the complete staged artifacts into the final location.
        os.rename(tmp_local_dir, local_dir)
        logger.info(f"Remote artifacts fetched to local cache: {local_dir}")
        return True

    except OSError as e:
        shutil.rmtree(tmp_local_dir, ignore_errors=True)
        logger.warning(
            f"Remote fetch failed for key {hash_key}: {e} "
            "— falling back to compile and store path"
        )
        return False
    finally:
        # Best-effort cleanup of the trash dir
        trash_dir = local_dir + f".trash.{os.getpid()}"
        shutil.rmtree(trash_dir, ignore_errors=True)


def _wait_for_completion(
    hash_key: str, local_cache_dir: str, timeout: int
) -> CompilationArtifacts:
    """Wait for another process to complete compilation.

    Polls the local cache directory until all required artifacts are present,
    the lock holder process dies, or the timeout expires.

    Args:
        hash_key: Cache hash key
        local_cache_dir: Local cache root directory (never NFS)
        timeout: Timeout in seconds for waiting

    Returns:
        CompilationArtifacts once compilation completes
    """
    cache_dir = os.path.join(local_cache_dir, hash_key)
    start_time = time.monotonic()

    # Read initial lock holder information.
    metadata_file_path = os.path.join(
        local_cache_dir, f".{hash_key}{VLLM_NEURON_COMPILATION_METADATA_FILE}"
    )
    initial_metadata = None
    for attempt in range(_METADATA_READ_RETRIES):
        initial_metadata = _read_lock_metadata(metadata_file_path)
        if initial_metadata:
            break
        logger.debug(
            f"Lock metadata not yet visible for {hash_key} "
            f"(attempt {attempt + 1}/{_METADATA_READ_RETRIES}) — retrying"
        )
        time.sleep(_METADATA_READ_INTERVAL)

    if not initial_metadata:
        raise RuntimeError(f"Cannot read lock metadata for {cache_dir}")

    lock_holder_pid = initial_metadata.get("pid")
    lock_holder_name = initial_metadata.get("process_name")
    lock_holder_cmdline = initial_metadata.get("cmdline")

    logger.info(
        f"Waiting for shared compilation by process {lock_holder_pid} ({lock_holder_name})"
    )

    while time.monotonic() - start_time < timeout:
        # Check if all required artifacts are present (sentinel + neff + hlo + metadata)
        if _is_cache_complete(cache_dir):
            cached_artifacts = _build_cache_artifact(cache_dir, hash_key)
            if cached_artifacts:
                logger.info(f"Shared compilation completed: {cache_dir}")
                return cached_artifacts
            else:
                raise RuntimeError(
                    f"Compilation completed but failed to load artifacts from {cache_dir}"
                )

        # Check if lock holder process is still alive and the same
        process_status = _is_same_process(
            lock_holder_pid, lock_holder_name, lock_holder_cmdline
        )

        if process_status is False:
            # Process died or changed - invalidate cache, clean up lock files, and retry.
            # The workdir is preserved to aid debugging of the failed compilation.
            logger.info(
                f"Lock holder process {lock_holder_pid} died or changed - "
                "invalidating cache entry and cleaning up lock files"
            )
            _remove_completion_signal(hash_key, local_cache_dir)
            _cleanup_lock_files(
                metadata_file_path, metadata_file_path + VLLM_NEURON_LOCK_FILE_SUFFIX
            )
            # Signal that caller should retry compilation from the beginning
            raise RuntimeError(
                "Lock holder process died - cache invalidated, please retry compilation"
            )

        time.sleep(VLLM_NEURON_DEFAULT_POLL_INTERVAL)

    elapsed = time.monotonic() - start_time
    raise TimeoutError(
        f"Shared compilation timeout after {elapsed:.1f}s in {cache_dir}"
    )


def _build_cache_artifact(
    cache_dir: str, hash_key: str
) -> Optional[CompilationArtifacts]:
    """Build CompilationArtifacts from cached files.

    Verifies that the ``cache_key`` embedded in the artifact metadata matches
    the ``hash_key`` used to locate this directory.  A mismatch indicates that
    the directory was renamed, copied incorrectly, or the metadata was written
    by a different compilation run.

    Args:
        cache_dir: Absolute path to the per-hash-key cache directory.
        hash_key: The 32-char deterministic hash that was used to locate this
            cache entry (i.e. the directory basename).

    Returns:
        CompilationArtifacts if artifacts load successfully.

    Raises:
        RuntimeError: If metadata is missing or the cache_key does not match
            ``hash_key``.
    """
    try:
        from vllm_neuron.compile.artifacts import CompilationArtifacts

        hlo_path = os.path.join(cache_dir, VLLM_NEURON_GRAPH_HLO_FILE)
        neff_filename = os.path.join(cache_dir, get_neff_filename(cache_dir))

        # Load and validate metadata
        metadata = _load_artifact_metadata(cache_dir)
        if metadata is None:
            raise RuntimeError(
                f"Required artifact metadata missing from cache: {cache_dir}"
            )

        # Verify the embedded cache key matches the directory we looked up
        if metadata.cache_key != hash_key:
            raise RuntimeError(
                f"Cache key mismatch: expected '{hash_key}', "
                f"found '{metadata.cache_key}' in {cache_dir}. "
                "The cache entry may have been corrupted or moved."
            )

        return CompilationArtifacts(
            hlo_filename=hlo_path,
            neff_filename=neff_filename,
            metadata=metadata,
        )

    except RuntimeError:
        raise  # Re-raise metadata/key-mismatch errors
    except Exception as e:
        # Non-critical errors return None for cache miss handling
        logger.warning(f"Failed to load cached artifacts from {cache_dir}: {e}")
        return None


def _del_stale_lock(hash_key: str, local_cache_dir: str) -> None:
    """Delete stale locks from dead processes.

    Args:
        hash_key: Cache hash key
        local_cache_dir: Local cache root directory (never NFS)
    """
    metadata_file_path = os.path.join(
        local_cache_dir, f".{hash_key}{VLLM_NEURON_COMPILATION_METADATA_FILE}"
    )
    cache_dir = os.path.join(local_cache_dir, hash_key)

    if not os.path.exists(metadata_file_path):
        return  # No metadata exists

    metadata = _read_lock_metadata(metadata_file_path)
    if not metadata:
        return

    pid = metadata.get("pid")
    process_name = metadata.get("process_name")
    cmdline = metadata.get("cmdline")

    if not all([pid, process_name, cmdline]):
        logger.warning(
            f"Incomplete lock metadata in {metadata_file_path} - cleaning up"
        )
        _remove_completion_signal(hash_key, local_cache_dir)
        _cleanup_lock_files(
            metadata_file_path, metadata_file_path + VLLM_NEURON_LOCK_FILE_SUFFIX
        )
        return

    # Check if the process that created the lock is still alive
    process_status = _is_same_process(pid, process_name, cmdline)

    if process_status is False:
        # Process died or PID was reused - invalidate cache and clean up lock files.
        # The workdir is preserved to aid debugging of the failed compilation.
        logger.info(
            f"Lock holder process {pid} ({process_name}) died or changed - "
            "invalidating cache entry and cleaning up lock files"
        )
        _remove_completion_signal(hash_key, local_cache_dir)
        _cleanup_lock_files(
            metadata_file_path, metadata_file_path + VLLM_NEURON_LOCK_FILE_SUFFIX
        )
    elif process_status is True:
        logger.info(
            f"Lock holder process {pid} ({process_name}) still alive - keeping existing lock"
        )
    else:
        logger.debug(f"Cannot verify process {pid} - keeping existing state")


def _is_cache_complete(cache_dir: str) -> bool:
    """Check if cache directory contains complete, coherent artifacts.

    Returns ``True`` only when **all** of the following are present:

    * ``graph.neff``   — NEFF binary written by ``neuronx-cc``
    * ``graph.hlo``    — serialised HLO module
    * ``.compilation_complete`` — the coherency sentinel written **last** by
      ``_store_local()``.  Any reader that observes this file is guaranteed to
      also see all other files fully written
    * at least one ``.artifact_metadata_vN.json`` — versioned metadata with
      checksums

    Args:
        cache_dir: Path to the per-hash-key cache directory to inspect.

    Returns:
        bool: ``True`` if all required files are present, ``False`` otherwise.
    """
    required_files = [
        VLLM_NEURON_GRAPH_HLO_FILE,
        get_neff_filename(cache_dir),
        VLLM_NEURON_COMPILATION_COMPLETE_FILE,
    ]

    # Check that all basic required files exist
    if not all(os.path.exists(os.path.join(cache_dir, f)) for f in required_files):
        return False

    # Check that at least one artifact metadata file exists (any version)
    if not os.path.exists(cache_dir):
        return False

    try:
        for filename in os.listdir(cache_dir):
            if filename.startswith(".artifact_metadata_v") and filename.endswith(
                ".json"
            ):
                return True
        return False  # No metadata file found
    except OSError:
        return False


def _get_current_process_info() -> Dict[str, any]:
    """Get current process information for lock metadata."""
    current_process = psutil.Process()
    return {
        "pid": current_process.pid,
        "name": current_process.name(),
        "cmdline": current_process.cmdline(),
    }


def _is_same_process(
    pid: int, expected_name: str, expected_cmdline: List[str]
) -> Optional[bool]:
    """Check if PID matches expected process name and command line."""
    try:
        process = psutil.Process(pid)
        actual_name = process.name()
        actual_cmdline = process.cmdline()

        # Check both name and command line match
        name_match = actual_name == expected_name
        cmdline_match = actual_cmdline == expected_cmdline

        if name_match and cmdline_match:
            logger.debug(f"Process {pid} verified: same process still alive")
            return True
        else:
            logger.debug(
                f"Process {pid} mismatch - name: {actual_name} vs {expected_name}, "
                f"cmdline: {actual_cmdline} vs {expected_cmdline}"
            )
            return False

    except psutil.NoSuchProcess:
        logger.debug(f"Process {pid} no longer exists")
        return False
    except (psutil.AccessDenied, psutil.ZombieProcess, OSError) as e:
        logger.debug(f"Cannot access process {pid}: {e} - assuming dead")
        return False


def _read_lock_metadata(metadata_file_path: str) -> Optional[Dict[str, any]]:
    """Read lock metadata from metadata file."""
    if not os.path.exists(metadata_file_path):
        return None

    try:
        with open(metadata_file_path, "r") as f:
            content = f.read()
    except (OSError, IOError) as e:
        logger.debug(f"Failed to read metadata file {metadata_file_path}: {e}")
        return None

    metadata = {}
    for line in content.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            if key == "pid":
                metadata["pid"] = int(value)
            elif key == "cmdline":
                import ast

                metadata["cmdline"] = ast.literal_eval(value)
            else:
                metadata[key] = value

    return metadata


def _write_compilation_metadata(metadata_file_path: str) -> None:
    """Write lock metadata with process information to metadata file.

    Args:
        metadata_file_path: Path to the metadata file to write
    """
    process_info = _get_current_process_info()

    metadata_content = f"pid:{process_info['pid']}\n"
    metadata_content += f"process_name:{process_info['name']}\n"
    metadata_content += f"cmdline:{process_info['cmdline']}\n"

    if dist.is_initialized():
        metadata_content += f"rank:{dist.get_rank()}\n"
    metadata_content += f"timestamp:{time.time()}\n"

    with open(metadata_file_path, "w") as f:
        f.write(metadata_content)
        f.flush()


def _cleanup_lock_files(metadata_file_path: str, lock_file_path: str) -> None:
    """Clean up lock-related files on compilation failure.

    Args:
        metadata_file_path: Path to the metadata file to remove
        lock_file_path: Path to the lock file to remove
    """
    for file_path, file_desc in [
        (metadata_file_path, "metadata file"),
        (lock_file_path, "lock file"),
    ]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up {file_desc} on failure: {file_path}")
            except OSError as e:
                logger.warning(f"Error removing {file_desc}: {e}")


def _remove_completion_signal(hash_key: str, local_cache_dir: str) -> None:
    """Remove the compilation completion signal to force recompilation.

    Deletes the ``.compilation_complete`` file from the cache directory so
    that ``_is_cache_complete()`` returns ``False`` and any in-progress
    waiters will not load stale artifacts.  All other artifact files
    (HLO, NEFF, metadata) are left in place and will be overwritten by
    the subsequent ``_store_local()`` call.

    Args:
        hash_key: Cache hash key
        local_cache_dir: Local cache root directory (never NFS)
    """
    cache_dir = os.path.join(local_cache_dir, hash_key)
    complete_file = os.path.join(cache_dir, VLLM_NEURON_COMPILATION_COMPLETE_FILE)
    if os.path.exists(complete_file):
        os.remove(complete_file)
        logger.debug(f"Removed completion signal for forced recompilation: {cache_dir}")


def _cleanup_directory(cache_dir: str, reason: str):
    """Clean up the entire compilation directory."""
    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info(f"Cleaned up compilation directory {cache_dir}: {reason}")
    except OSError as e:
        logger.warning(f"Failed to cleanup directory {cache_dir}: {e}")


def _get_artifact_metadata_filename(version: int) -> str:
    """Get standardized filename for artifact metadata of given version.

    Args:
        version: Metadata schema version

    Returns:
        str: Filename for the metadata file
    """
    return f".artifact_metadata_v{version}.json"


def _load_artifact_metadata(cache_dir: str) -> Optional["ArtifactMetadata"]:
    """Load artifact metadata from cache directory with validation.

    Args:
        cache_dir: Cache directory path

    Returns:
        ArtifactMetadata: Validated metadata object if found, None otherwise
    """
    # Try latest versions first, fallback to older
    for version in [0]:  # Start with v0, add new versions as [1, 0] later
        artifact_metadata_file = _get_artifact_metadata_filename(version)
        filepath = os.path.join(cache_dir, artifact_metadata_file)

        if os.path.exists(filepath):
            try:
                from vllm_neuron.compile.schema import parse_metadata_dict

                with open(filepath, "r") as f:
                    data = json.load(f)
                return parse_metadata_dict(data)
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.debug(f"Failed to load metadata from {filepath}: {e}")
                continue

    return None  # No valid metadata found


def _normalize_compiler_args(compiler_args: List[str]) -> Dict[str, str]:
    """Parse compiler_args into a canonical dict with the resolved platform target.

    Converts ``--key value``, ``--key=value``, and standalone flag forms into a
    dict mapping argument name to value (standalone flags map to ``"True"``).
    The ``--target`` key is always resolved via ``resolve_target()`` with
    precedence: env var > explicit ``--target`` in args > NRT auto-detect.
    This ensures that passing
    ``["--target", "trn2"]`` explicitly and relying on auto-detection on a
    trn2 host both produce the identical canonical dict — and therefore the
    same cache hash.

    Args:
        compiler_args: List of compiler argument strings, as returned by
            ``_parse_compiler_args``.

    Returns:
        Dict[str, str]: Canonical mapping of argument name
    """
    result: Dict[str, str] = {}
    i = 0
    while i < len(compiler_args):
        arg = compiler_args[i]
        if arg.startswith("--") and "=" in arg:
            # --key=value form
            key, val = arg.split("=", 1)
            result[key] = val
        elif (
            arg.startswith("-")
            and i + 1 < len(compiler_args)
            and not compiler_args[i + 1].startswith("-")
        ):
            # --key value form (two tokens)
            result[arg] = compiler_args[i + 1]
            i += 1
        else:
            # Standalone flag (e.g. -O1, --verbose)
            result[arg] = "True"
        i += 1

    # Always resolve and set --target canonically so that explicit and
    # auto-detected targets produce the same dict entry.
    # Precedence: env var > --target from args > NRT auto-detect.
    result["--target"] = resolve_target(compiler_args)
    return result


def _cleanup_staging(staging: str) -> None:
    """Best-effort removal of a staging directory.

    The cleanup is best-effort: remote filesystem transient errors (``ESTALE``, ``EIO``, ``EACCES``)
    are logged as warnings rather than propagated, because a leftover staging dir is not fatal.

    Args:
        staging: Absolute path to the staging directory to remove.
    """
    if os.path.exists(staging):
        try:
            shutil.rmtree(staging)
        except OSError as e:
            logger.warning(f"Failed to cleanup staging dir {staging}: {e}")


def _promote_to_remote(
    hash_key: str, local_cache_dir: str, remote_cache_dir: str
) -> None:
    """Atomically promote a local cache entry to the remote store via staging + rename.

    Copies ``local_cache_dir/<hash_key>/`` into a per-process staging directory
    on the same remote parent (named ``<hash_key>.tmp.<hostname>.<pid>``), then
    performs an atomic ``os.rename`` to the final destination.  If the remote
    entry already exists (either detected by the fast-path pre-check or via an
    ``OSError(EEXIST)`` from ``os.rename``), the call is a silent no-op — the
    remote entry is left intact.

    The staging directory is always cleaned up in a ``finally`` block via
    :func:`_cleanup_staging`.

    Args:
        hash_key: Cache hash key identifying the entry to promote.
        local_cache_dir: Local cache root directory.
        remote_cache_dir: Remote root directory to promote into.
    """
    local_entry = os.path.join(local_cache_dir, hash_key)
    remote_entry = os.path.join(remote_cache_dir, hash_key)
    staging = os.path.join(
        remote_cache_dir, f"{hash_key}.tmp.{socket.gethostname()}.{os.getpid()}"
    )

    # Fast path: skip all remote I/O if the entry already exists.
    if os.path.exists(remote_entry):
        logger.info(f"Remote entry already exists, skipping promotion: {remote_entry}")
        return

    os.makedirs(remote_cache_dir, exist_ok=True)
    try:
        shutil.copytree(
            local_entry,
            staging,
            ignore=shutil.ignore_patterns(
                VLLM_NEURON_CACHE_ACCESSED_FILE,
            ),
        )
        os.rename(staging, remote_entry)
        logger.info(f"Promoted cache entry to remote: {remote_entry}")
    except OSError as e:
        if e.errno in (errno.EEXIST, errno.ENOTEMPTY):
            logger.debug(
                f"Remote entry already exists (concurrent promoter won): {remote_entry}"
            )
        else:
            logger.warning(f"Failed to promote cache entry {hash_key} to remote: {e}")
    finally:
        _cleanup_staging(staging)


def _upgrade_metadata_schema(
    data: Dict[str, Any], target_version: int
) -> Dict[str, Any]:
    """Upgrade metadata schema to target version.

    Args:
        data: Metadata dictionary to upgrade
        target_version: Target schema version

    Returns:
        Upgraded metadata dictionary
    """
    current_version = data.get("version", 0)

    return data
