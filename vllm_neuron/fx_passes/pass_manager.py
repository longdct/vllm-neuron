# SPDX-License-Identifier: Apache-2.0
"""Pass manager for executing FX passes with debugging support."""

import logging
import os
import time

import torch

from .base import FXPass


class FXPassManager:
    """Manages execution of FX passes with timing logs and graph dumping.

    The pass manager executes passes sequentially and provides debugging
    capabilities including execution timing and graph state dumping.
    """

    def __init__(self):
        """Initialize the pass manager."""
        self.passes: list[FXPass] = []
        self.logger = logging.getLogger(__name__)

    def add_pass(self, pass_obj: FXPass) -> None:
        """Add a pass to the execution pipeline.

        Args:
            pass_obj: The FX pass to add
        """
        self.passes.append(pass_obj)

    def run_passes(
        self, gm: torch.fx.GraphModule, **kwargs
    ) -> tuple[torch.fx.GraphModule, dict]:
        """Execute all passes sequentially with timing and debugging.

        Args:
            gm: The PyTorch FX GraphModule to transform
            **kwargs: Additional arguments passed to each pass

        Returns:
            tuple[torch.fx.GraphModule, dict]: The transformed GraphModule and collected metadata

        Raises:
            RuntimeError: If any pass fails during execution
        """
        compiler_workdir = kwargs.get("compiler_workdir")

        # Dump original graph
        if compiler_workdir:
            self._dump_graph(gm, compiler_workdir, 0, "original")

        current_gm = gm
        all_metadata = {}

        for idx, pass_obj in enumerate(self.passes, 1):
            try:
                start_time = time.perf_counter()
                current_gm, pass_metadata = pass_obj.run(
                    current_gm, **kwargs, **all_metadata
                )
                current_gm.recompile()
                elapsed_time = time.perf_counter() - start_time

                # Collect metadata from this pass
                all_metadata[pass_obj.name] = pass_metadata

                self.logger.debug(
                    f"FX Pass '{pass_obj.name}' completed in {elapsed_time:.4f}s"
                )

                # Dump graph after pass
                if compiler_workdir:
                    self._dump_graph(current_gm, compiler_workdir, idx, pass_obj.name)

            except Exception as e:
                error_msg = f"FX Pass '{pass_obj.name}' failed: {str(e)}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        return current_gm, all_metadata

    def _dump_graph(
        self, gm: torch.fx.GraphModule, compiler_workdir: str, idx: int, pass_name: str
    ) -> None:
        """Dump GraphModule to file for debugging.

        Args:
            gm: The GraphModule to dump
            compiler_workdir: Base directory for compiler work files
            idx: Pass index (0 for original, 1+ for after passes)
            pass_name: Name of the pass (or "original")
        """
        try:
            passes_dir = os.path.join(compiler_workdir, "passes")
            os.makedirs(passes_dir, exist_ok=True)

            if idx == 0:
                filename = f"{idx:02d}_{pass_name}.txt"
            else:
                filename = f"{idx:02d}_after_{pass_name}.txt"

            filepath = os.path.join(passes_dir, filename)

            with open(filepath, "w") as f:
                f.write(f"GraphModule after {pass_name}:\n")
                f.write("=" * 50 + "\n\n")

                replica_header = _format_replica_groups_header(gm)
                if replica_header:
                    f.write(replica_header)
                    f.write("\n")

                f.write(str(gm.graph))
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("GraphModule code:\n")
                f.write(gm.code)

            self.logger.debug(f"Dumped graph to {filepath}")

        except Exception as e:
            self.logger.warning(f"Failed to dump graph for {pass_name}: {str(e)}")


def _format_replica_groups_header(gm: torch.fx.GraphModule) -> str:
    """Build a header string showing replica groups for collective nodes.

    Scans the graph for nodes whose ``meta`` contains ``replica_groups``
    (populated by :class:`CollectiveReplicaGroupsPass`) and returns a
    human-readable mapping of ``group_name → [ranks]``.

    Returns:
        A multi-line string like::

            Replica Groups
            --------------
            group_name_0: [0, 1, 2, 3]
            group_name_1: [0, 1]

        or an empty string if no replica groups are present.
    """
    # Collect unique group_name → ranks, preserving first-seen order.
    seen: dict[str, list[int]] = {}
    for node in gm.graph.nodes:
        groups = node.meta.get("replica_groups")
        if groups is None:
            continue
        # group_name is always the last positional arg for _c10d_functional collectives
        if node.args:
            group_name = node.args[-1]
            if isinstance(group_name, str) and group_name not in seen:
                seen[group_name] = groups[0]

    if not seen:
        return ""

    lines = ["Replica Groups", "-" * 14]
    for name, ranks in seen.items():
        lines.append(f"{name}: {ranks}")
    lines.append("")
    return "\n".join(lines)
