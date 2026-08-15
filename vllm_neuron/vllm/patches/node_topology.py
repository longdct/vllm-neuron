# SPDX-License-Identifier: Apache-2.0
"""Single implementation of the ``in_the_same_node_as`` replacement.

vLLM's ``parallel_state.in_the_same_node_as()`` calls
``torch.distributed.barrier(group=pg)`` to discover which ranks share memory.
PyTorch's ``barrier()`` unconditionally calls ``torch._C._get_accelerator()``
*before* dispatching to the backend, which raises on Neuron (PrivateUse1 with no
``PrivateUse1HooksInterface`` registered) even though the group is Gloo and Gloo's
barrier needs no device information. The call chain that breaks::

    init_model_parallel_group()
      -> GroupCoordinator.__init__()
      -> MessageQueue.create_from_process_group()
      -> in_the_same_node_as(pg)
      -> torch.distributed.barrier(group=pg)
      -> torch._C._get_accelerator()          # raises

The replacement derives node membership arithmetically instead of probing, so
``MessageQueue`` can still use shared memory and group init completes.

Either proper fix removes the need for this: register
``PrivateUse1HooksInterface`` for Neuron (arriving with Torch Eager
consolidation), or ship the plugin so PrivateUse1 is never bound.

This module exists because the replacement was previously duplicated in
``NeuronWorker._patch_in_same_node_as_function`` and
``neuron_parallel_state._ensure_vllm_parallel_state``. The two inner functions
were identical, but they derived ``ranks_per_node`` differently -- the worker from
``ParallelConfig`` (with a floor of 1), the parallel-state path from the live
``torch.distributed`` world. Both derivations are legitimate for where they run,
so the caller supplies ``ranks_per_node`` and only the topology maths is shared.
"""

import torch.distributed as dist

from vllm_neuron.vllm.patches.guards import require_attr, require_params
from vllm_neuron.vllm.patches.registry import mark_applied

PATCH_NAME = "node_topology:in_the_same_node_as"


def make_in_the_same_node_as(ranks_per_node: int):
    """Build a barrier-free ``in_the_same_node_as`` for a fixed node size.

    Group-local ranks are mapped back to global ranks before being assigned to a
    node, so subgroups (TP/DP/EP) report same-node membership correctly however
    their ranks are laid out across nodes. With DP>1 an EP/cross-DP group spans
    all nodes, which is exactly the case a group-local calculation gets wrong.
    """
    if ranks_per_node < 1:
        raise ValueError(f"ranks_per_node must be >= 1, got {ranks_per_node}")

    def patched_in_the_same_node_as(pg, source_rank: int = 0) -> list[bool]:
        global_ranks = [
            dist.get_global_rank(pg, r)
            for r in range(dist.get_world_size(group=pg))
        ]
        source_node = global_ranks[source_rank] // ranks_per_node
        return [g // ranks_per_node == source_node for g in global_ranks]

    return patched_in_the_same_node_as


def install_in_the_same_node_as(ranks_per_node: int) -> None:
    """Guard and install the replacement onto ``vllm.distributed.parallel_state``.

    Parameterized, so it is not a zero-arg registry entry; the matching existence
    guard is registered at :attr:`~vllm_neuron.vllm.patches.registry.Phase.
    DISTRIBUTED_INIT`. Idempotent, and records itself via ``mark_applied`` so
    startup assertions can see it.
    """
    import vllm.distributed.parallel_state as parallel_state

    existing = require_attr(
        parallel_state, "in_the_same_node_as", patch="node_topology"
    )
    if getattr(existing, "_neuron_node_topology_patched", False):
        return
    require_params(existing, "pg", patch="node_topology")

    patched = make_in_the_same_node_as(ranks_per_node)
    patched._neuron_node_topology_patched = True  # type: ignore[attr-defined]
    parallel_state.in_the_same_node_as = patched
    mark_applied(PATCH_NAME)
