# SPDX-License-Identifier: Apache-2.0
"""Parallel topology used by the DeepSeek-V4 serving implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepseekV4ParallelTopology:
    """Resolved TP/DP/EP placement for one rank.

    EP partitions experts over the TP x DP world.  Ranks in the same EP
    partition hold identical experts and tensor-shard their intermediate
    dimension.  Keeping this arithmetic in a value object makes checkpoint
    ownership independent of process-group implementation details.
    """

    tp_degree: int = 1
    dp_degree: int = 1
    ep_degree: int = 1
    tp_rank: int = 0
    dp_rank: int = 0
    ep_rank: int = 0
    expert_tp_rank: int = 0

    @property
    def world_size(self) -> int:
        return self.tp_degree * self.dp_degree

    @property
    def expert_tp_degree(self) -> int:
        """Number of ranks that column-shard one expert's intermediate dim.

        With EP enabled the expert-TP group is the TP x DP world partitioned
        into ``ep_degree`` disjoint expert shards, so each partition has
        ``world_size // ep_degree`` ranks.

        With EP disabled (``ep_degree == 1``) there is no cross-replica expert
        group at all: every DP replica carries a complete model and there is no
        collective spanning replicas, so the ranks that column-shard an expert
        are exactly the TP group.  Deriving this as ``world_size`` instead would
        claim DP replicas share a shard -- and since ``_forward_*`` only ever
        all-reduces over the TP group, those extra shards would never be summed.
        """
        if self.ep_degree == 1:
            return self.tp_degree
        return self.world_size // self.ep_degree

    def validate(self, *, num_heads: int, output_groups: int,
                 num_experts: int, expert_intermediate_size: int) -> None:
        values = (self.tp_degree, self.dp_degree, self.ep_degree)
        if any(value < 1 for value in values):
            raise ValueError(f"parallel degrees must be positive, got {values}")
        if self.world_size % self.ep_degree:
            raise ValueError(
                f"ep_degree={self.ep_degree} must divide TPxDP world_size={self.world_size}"
            )
        checks = (
            (num_heads, self.tp_degree, "num_attention_heads", "tp_degree"),
            (output_groups, self.tp_degree, "o_groups", "tp_degree"),
            (num_experts, self.ep_degree, "num_experts", "ep_degree"),
            (expert_intermediate_size, self.expert_tp_degree,
             "expert_intermediate_size", "expert_tp_degree"),
        )
        for size, degree, size_name, degree_name in checks:
            if size % degree:
                raise ValueError(
                    f"{size_name}={size} must be divisible by {degree_name}={degree}"
                )
        if not 0 <= self.ep_rank < self.ep_degree:
            raise ValueError(f"ep_rank={self.ep_rank} is outside ep_degree={self.ep_degree}")
        if not 0 <= self.expert_tp_rank < self.expert_tp_degree:
            raise ValueError(
                f"expert_tp_rank={self.expert_tp_rank} is outside "
                f"expert_tp_degree={self.expert_tp_degree}"
            )

    def local_expert_interval(self, num_experts: int) -> tuple[int, int]:
        if num_experts % self.ep_degree:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by ep_degree={self.ep_degree}"
            )
        count = num_experts // self.ep_degree
        start = self.ep_rank * count
        return start, start + count


def resolve_parallel_topology() -> DeepseekV4ParallelTopology:
    """Resolve vLLM/Neuron groups, degrading to a single CPU rank in tests."""
    try:
        from vllm.config import get_current_vllm_config
        from vllm.distributed.parallel_state import get_tp_group

        config = get_current_vllm_config().parallel_config
        tp_group = get_tp_group()
        tp_degree = tp_group.world_size
        tp_rank = tp_group.rank_in_group
        dp_degree = int(config.data_parallel_size)
        ep_enabled = bool(config.enable_expert_parallel)
    except (AssertionError, RuntimeError, AttributeError, ValueError):
        return DeepseekV4ParallelTopology()

    if not ep_enabled:
        # ``expert_tp_rank`` MUST be plumbed here.  Routed-expert weights are
        # column-sharded across ``expert_tp_degree`` ranks by
        # ``weight_loaders.load_checkpoint_weights``, which narrows the
        # checkpoint tensor at ``expert_tp_rank * shard_size``.  Leaving it at
        # its default of 0 made every rank load shard 0, so the TP all-reduce
        # in ``DeepseekV4MoE.forward`` summed ``tp_degree`` copies of the same
        # partial while the other shards were never computed -- a silently
        # wrong MoE output at any ``tp_degree > 1``.  With EP off the expert-TP
        # group is the TP group, so the column index is the TP rank.
        return DeepseekV4ParallelTopology(
            tp_degree=tp_degree,
            dp_degree=dp_degree,
            tp_rank=tp_rank,
            expert_tp_rank=tp_rank,
        )

    from vllm_neuron.parallel.neuron_parallel_state import (
        get_neuron_ep_degree,
        get_neuron_ep_rank,
        get_neuron_ep_tp_group,
    )

    ep_degree = get_neuron_ep_degree()
    if ep_degree <= 1:
        # ``--enable-expert-parallel`` with a degenerate EP degree: no EP group
        # exists (``get_neuron_ep_tp_group`` asserts ``ep_degree > 1``), so this
        # is the plain TP geometry above.
        return DeepseekV4ParallelTopology(
            tp_degree=tp_degree,
            dp_degree=dp_degree,
            tp_rank=tp_rank,
            expert_tp_rank=tp_rank,
        )
    ep_tp_group = get_neuron_ep_tp_group()
    dp_rank = 0
    try:
        from vllm.distributed.parallel_state import get_dp_group
        dp_rank = get_dp_group().rank_in_group
    except (AssertionError, RuntimeError, AttributeError, ValueError):
        pass
    return DeepseekV4ParallelTopology(
        tp_degree=tp_degree,
        dp_degree=dp_degree,
        ep_degree=ep_degree,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        ep_rank=get_neuron_ep_rank(),
        expert_tp_rank=ep_tp_group.rank_in_group,
    )
