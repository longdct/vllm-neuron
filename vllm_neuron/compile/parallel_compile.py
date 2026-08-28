# SPDX-License-Identifier: Apache-2.0
import concurrent.futures
import logging
import os
import shutil
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from vllm_neuron import envs
from vllm_neuron.compile.backend import (
    neuroncc_compile,
    _parse_compiler_args,
)
from vllm_neuron.compile.cache import (
    get_neff_filename,
    save_cache,
    _is_cache_complete,
    VLLM_NEURON_GRAPH_HLO_FILE,
    VLLM_NEURON_COMPILATION_COMPLETE_FILE,
)
from vllm_neuron.compile.platform import get_server_prefix
from vllm_neuron.utils.timer import timer

logger = logging.getLogger(__name__)


def parallel_compile(
    options: dict = {},
    rank=None,
    world_size=None,
    remote_cache_dir: Optional[str] = None,
):
    """Compile captured HLOs to NEFFs in parallel.

    When rank/world_size are provided (distributed mode):
        Assumes a barrier was called before invocation so the cache directory
        contains the globally complete set of HLOs. All the HLOs that
        require compilation are passed on to rank 0 to compile.
        Compilation happens inside the rank subdirectory chosen by
        _find_rank_subdir to avoid race conditions when multiple servers
        share the same cache directory. After compilation, files are
        atomically moved to the hash root directory.

    When rank is None (non-distributed mode):
        Compiles all HLOs found in the cache directory on this single process.
        Compilation happens directly in the hash root directory since there
        is no contention.

    Only HLOs captured by this server (identified by the server prefix derived
    from NEURON_VISIBLE_DEVICES) are compiled. This prevents race conditions
    when multiple independent vLLM servers share the same compile cache
    directory (e.g., DI prefill/decode servers, or non-MoE DP engines).
    Once compilation is complete, this saves all the compiled artifacts
    to the remote cache directory if provided as input.
    """
    # Mirror the injection the main compile() path does before hashing, so
    # stand-alone parallel compiles honor the same platform defaults.
    from vllm_neuron.compile.backend import _apply_platform_compiler_args

    options = _apply_platform_compiler_args(options)

    max_workers = envs.VLLM_NEURON_PARALLEL_COMPILE_WORKERS

    compile_base_dir = options.get(
        "compiler_workdir", envs.get_neuron_compile_cache_dir()
    )

    server_prefix = get_server_prefix()

    is_distributed = rank is not None and world_size is not None

    # Find all HLO hash dirs that need compilation (have HLO but no NEFF)
    all_hlo = sorted(
        d
        for d in os.listdir(compile_base_dir)
        if os.path.isdir(os.path.join(compile_base_dir, d))
        and _has_hlo(os.path.join(compile_base_dir, d), server_prefix, is_distributed)
        and not _is_cache_complete(os.path.join(compile_base_dir, d))
    )

    # Only rank 0 compiles in a distributed setup.
    if is_distributed:
        if rank == 0:
            my_hlos = all_hlo
            logger.info(
                "rank %d: compile — %d total HLOs, %d assigned to this rank",
                rank,
                len(all_hlo),
                len(my_hlos),
            )
        else:
            logger.info(
                "rank %d: Waiting for rank0 to compile all %d graphs",
                rank,
                len(all_hlo),
            )
            return
    else:
        my_hlos = all_hlo
        logger.info("compile — %d HLOs to compile", len(my_hlos))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _compile_single,
                compile_base_dir,
                hlo,
                options,
                server_prefix,
                is_distributed,
            )
            for hlo in my_hlos
        ]
        concurrent.futures.wait(futures)
    for future in futures:
        future.result()

    # Clean up any remaining rank subdirs for this server's prefix.
    # This handles hash dirs that were already compiled by another prefix
    # (skipped above because _is_cache_complete was True) but still have
    # this server's rank subdirs lingering.
    if is_distributed:
        for d in os.listdir(compile_base_dir):
            hash_dir = os.path.join(compile_base_dir, d)
            if os.path.isdir(hash_dir):
                _cleanup_prefix_subdirs(hash_dir, server_prefix)

    # Post Compilation, save artifacts to remote cache
    if remote_cache_dir:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(save_cache, compile_base_dir, remote_cache_dir, hlo)
                for hlo in my_hlos
            ]
            concurrent.futures.wait(futures)
        for future in futures:
            future.result()


def _find_rank_subdir(hlo_dir, server_prefix=""):
    """Find the smallest rank subdirectory that contains a graph.hlo file.

    When server_prefix is provided, only considers subdirectories matching
    that prefix (e.g., "dev0_7.rank0"). When empty, falls back to
    "rank<N>" naming.

    Returns:
        The rank subdirectory path, or None if no matching rank subdirectory exists.
    """
    if server_prefix:
        prefix_pattern = f"{server_prefix}.rank"
    else:
        prefix_pattern = "rank"

    rank_dirs = sorted(
        entry
        for entry in os.listdir(hlo_dir)
        if entry.startswith(prefix_pattern)
        and os.path.isdir(os.path.join(hlo_dir, entry))
    )
    for entry in rank_dirs:
        if os.path.exists(os.path.join(hlo_dir, entry, "graph.hlo")):
            return os.path.join(hlo_dir, entry)

    return None


def _has_hlo(hlo_dir, server_prefix="", is_distributed=False):
    """Check if a hash directory contains a graph.hlo relevant to this server.

    - Non-distributed (is_distributed=False): checks for graph.hlo directly
      in the hash root directory only.
    - Distributed (is_distributed=True): checks for graph.hlo inside a rank
      subdirectory matching server_prefix only.
    """
    if not is_distributed:
        return os.path.exists(os.path.join(hlo_dir, "graph.hlo"))
    return _find_rank_subdir(hlo_dir, server_prefix) is not None


def _cleanup_prefix_subdirs(workdir, server_prefix, exclude_dir=None):
    """Remove all rank subdirectories matching this server's prefix.

    Args:
        workdir: The hash root directory.
        server_prefix: The server prefix to match against.
        exclude_dir: Optional absolute path of a rank subdir to keep.
    """
    if server_prefix:
        cleanup_pattern = f"{server_prefix}.rank"
    else:
        cleanup_pattern = "rank"

    for entry in os.listdir(workdir):
        path = os.path.join(workdir, entry)
        if entry.startswith(cleanup_pattern) and os.path.isdir(path):
            if exclude_dir and os.path.abspath(path) == os.path.abspath(exclude_dir):
                continue
            shutil.rmtree(path)


def _atomic_move_to_root(src_dir, dest_dir):
    """Recursively move all files from src_dir to dest_dir using atomic renames.

    Walks the source tree, mirrors the directory structure in dest_dir via
    os.makedirs(exist_ok=True), and atomically renames each file. On Linux,
    os.rename on a file silently replaces an existing destination, so if
    another server prefix already moved an identical file, it is harmlessly
    overwritten (contents are identical for the same hash).
    """
    for dirpath, dirnames, filenames in os.walk(src_dir):
        rel_dir = os.path.relpath(dirpath, src_dir)
        dest_subdir = os.path.join(dest_dir, rel_dir) if rel_dir != "." else dest_dir
        os.makedirs(dest_subdir, exist_ok=True)
        for filename in filenames:
            src_path = os.path.join(dirpath, filename)
            dest_path = os.path.join(dest_subdir, filename)
            os.rename(src_path, dest_path)


def _compile_single(
    cache_dir, hlo_key, options, server_prefix="", is_distributed=False
):
    """Compile a single HLO to NEFF.

    Non-distributed mode:
        Compiles directly in the hash root directory. No rank subdirs involved.

    Distributed mode:
        1. Finds the compilation rank subdir via _find_rank_subdir.
        2. Cleans up all other rank subdirs for this server's prefix.
        3. Compiles inside the chosen rank subdir.
        4. Atomically moves compiled artifacts to the hash root.
        5. Deletes the rank subdir used for compilation.
    """
    workdir = os.path.join(cache_dir, hlo_key)

    if not is_distributed:
        # Non-distributed: compile directly in the hash root
        _do_compile(workdir, options, hlo_key)
        return

    # Distributed: compile inside the chosen rank subdir
    compile_dir = _find_rank_subdir(workdir, server_prefix)
    if compile_dir is None:
        logger.warning(
            "No rank subdir found for HLO %s with prefix %s, skipping",
            hlo_key,
            server_prefix,
        )
        return

    # Clean up all other rank subdirs for this prefix
    _cleanup_prefix_subdirs(workdir, server_prefix, exclude_dir=compile_dir)

    # Set --output so the NEFF is written with the hash-based name expected
    # by cache lookup, rather than one derived from the rank subdir basename.
    neff_filename = os.path.join(compile_dir, get_neff_filename(workdir))
    compiler_args = options.get("compiler_args", "")
    if isinstance(compiler_args, list):
        compiler_args = compiler_args + ["--output", neff_filename]
    else:
        compiler_args = compiler_args + f" --output {neff_filename}"
    options = {**options, "compiler_args": compiler_args}

    _do_compile(compile_dir, options, hlo_key)

    # Atomically move compiled artifacts to the hash root
    _atomic_move_to_root(compile_dir, workdir)

    # Remove the rank subdir now that everything is at the root
    shutil.rmtree(compile_dir)


def _do_compile(compile_dir, options, hlo_key):
    """Run neuroncc compilation in the given directory.

    Expects graph.hlo to exist in compile_dir. Writes graph.neff and
    the completion marker file into compile_dir.
    """
    compiler_args = _parse_compiler_args(options.get("compiler_args", ""))
    hlo_path = os.path.join(compile_dir, VLLM_NEURON_GRAPH_HLO_FILE)

    from vllm_neuron.compile.hlo import load_hlo_module

    hlo_module = load_hlo_module(hlo_path)

    with timer() as compile_timer:
        neff_filename = neuroncc_compile(
            hlo_module.SerializeToString(),
            compile_dir,
            compiler_args=compiler_args,
        )
    compilation_time = compile_timer()

    completion_info = f"completed:{time.time()}\n"
    completion_info += f"neff_size:{os.path.getsize(neff_filename)}\n"
    with open(
        os.path.join(compile_dir, VLLM_NEURON_COMPILATION_COMPLETE_FILE), "w"
    ) as f:
        f.write(completion_info)
        f.flush()

    logger.info("Compiled HLO %s in %.1fs", hlo_key, compilation_time)
