# SPDX-License-Identifier: Apache-2.0
"""FX pass that writes replica group ranks into collective call nodes."""

import logging
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group

from .base import FXPass

logger = logging.getLogger(__name__)

# _c10d_functional collective ops that carry a group_name argument
COLLECTIVE_OPS = {
    torch.ops._c10d_functional.all_reduce,
    torch.ops._c10d_functional.all_gather_into_tensor,
    torch.ops._c10d_functional.reduce_scatter_tensor,
    torch.ops._c10d_functional.all_to_all_single,
    torch.ops._c10d_functional.broadcast,
}


class CollectiveReplicaGroupsPass(FXPass):
    """Resolve process group names in collective call nodes to explicit replica group ranks.

    Collective ops in the FX graph reference process groups by an opaque
    ``group_name`` string.  This pass resolves each ``group_name`` to the
    concrete list of ranks via ``torch.distributed`` and writes the result
    into ``node.meta["replica_groups"]`` so downstream consumers (e.g. HLO
    lowering) can access the rank information without re-resolving at runtime.
    """

    @property
    def name(self) -> str:
        return "collective_replica_groups"

    def run(
        self, gm: torch.fx.GraphModule, **kwargs
    ) -> Tuple[torch.fx.GraphModule, Dict]:
        resolved_count = 0
        group_cache: Dict[str, List[int]] = {}

        for node in gm.graph.nodes:
            if node.op != "call_function" or node.target not in COLLECTIVE_OPS:
                continue

            group_name = self._extract_group_name(node)
            if group_name is None:
                continue

            if group_name not in group_cache:
                group_cache[group_name] = self._resolve_ranks(group_name)

            ranks = group_cache[group_name]
            node.meta["replica_groups"] = [ranks]
            resolved_count += 1
            logger.debug(
                f"Wrote replica_groups={ranks} for {node.target} (group={group_name})"
            )

        return gm, {"resolved_count": resolved_count}

    @staticmethod
    def _extract_group_name(node: torch.fx.Node):
        """Return the group_name argument from a collective call node."""
        # group_name is always the last positional arg for all _c10d_functional collectives
        if node.args:
            return node.args[-1]
        return None

    @staticmethod
    def _resolve_ranks(group_name: str) -> List[int]:
        """Resolve a process group name to its list of ranks."""
        group = _resolve_process_group(group_name)
        return dist.get_process_group_ranks(group)
