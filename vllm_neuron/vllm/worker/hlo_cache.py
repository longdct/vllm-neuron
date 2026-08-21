# SPDX-License-Identifier: Apache-2.0
"""Lifecycle helpers for graph-capture entries in the Neuron compile cache."""

import logging
import os
from pathlib import Path
import time


logger = logging.getLogger(__name__)

_COMPILATION_COMPLETE = ".compilation_complete"
_GRAPH_HLO = "graph.hlo"
_QUARANTINE_DIR = ".incomplete_hlo_quarantine"


def quarantine_incomplete_hlo_captures(
    compile_base_dir: str | os.PathLike[str],
    server_prefix: str,
    tp_rank: int,
) -> list[tuple[str, Path]]:
    """Quarantine stale captures owned by this server and TP rank.

    ``parallel_compile`` discovers HLOs in immediate
    ``<server-prefix>.rank<N>`` children of each cache-key directory. Moving an
    incomplete capture one level below the hidden quarantine directory keeps
    all of its diagnostics while removing it from compiler discovery. The
    rename is atomic because source and destination are under the same hash.

    Completed cache entries and captures belonging to other servers or ranks
    are deliberately left untouched.
    """
    base_dir = Path(compile_base_dir)
    if not base_dir.is_dir():
        return []

    capture_name = f"{server_prefix}.rank{tp_rank}"
    quarantined: list[tuple[str, Path]] = []
    for hash_dir in base_dir.iterdir():
        if not hash_dir.is_dir() or (hash_dir / _COMPILATION_COMPLETE).exists():
            continue

        capture_dir = hash_dir / capture_name
        if not (capture_dir / _GRAPH_HLO).is_file():
            continue

        quarantine_dir = hash_dir / _QUARANTINE_DIR
        quarantine_dir.mkdir(exist_ok=True)
        destination = quarantine_dir / (
            f"{capture_name}.{os.getpid()}.{time.time_ns()}"
        )
        try:
            capture_dir.rename(destination)
        except FileNotFoundError:
            # Another lifecycle owner may have handled the same abandoned
            # capture between discovery and rename.
            continue

        quarantined.append((hash_dir.name, destination))
        logger.warning(
            "Quarantined incomplete Neuron HLO cache key %s at %s",
            hash_dir.name,
            destination,
        )

    return quarantined
