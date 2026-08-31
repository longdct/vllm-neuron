# SPDX-License-Identifier: Apache-2.0
"""Parallel topology used by the DeepSeek-V4 serving implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputProjectionPartition:
    """One TP rank's slice of the grouped low-rank output projection.

    At TP degrees up to the number of output groups, ranks own whole groups.
    Above that point, several consecutive ranks split the input columns of one
    group and replicate that group's ``o_b`` columns.  The ordinary TP
    all-reduce then sums both the within-group column partials and the
    contributions from distinct groups.
    """

    group_start: int
    group_count: int
    ranks_per_group: int
    input_offset: int
    input_width: int


def resolve_output_projection_partition(
    *,
    tp_degree: int,
    tp_rank: int,
    output_groups: int,
    total_input_width: int,
) -> OutputProjectionPartition:
    """Resolve a grouped output-projection shard for one TP rank.

    ``total_input_width`` is ``num_attention_heads * head_dim``.  The official
    checkpoint stores ``o_a`` as ``output_groups`` row blocks, each with the
    full input width for one group.  Consequently TP can either divide the
    groups (TP <= groups) or divide each group's input columns (TP > groups).
    """
    if tp_degree < 1 or output_groups < 1:
        raise ValueError(
            "tp_degree and output_groups must be positive, got "
            f"tp_degree={tp_degree}, output_groups={output_groups}"
        )
    if not 0 <= tp_rank < tp_degree:
        raise ValueError(f"tp_rank={tp_rank} is outside tp_degree={tp_degree}")
    if total_input_width % output_groups:
        raise ValueError(
            f"total_input_width={total_input_width} must be divisible by "
            f"output_groups={output_groups}"
        )

    group_input_width = total_input_width // output_groups
    if tp_degree <= output_groups:
        if output_groups % tp_degree:
            raise ValueError(
                f"output_groups={output_groups} must be divisible by "
                f"tp_degree={tp_degree} when tp_degree <= output_groups"
            )
        group_count = output_groups // tp_degree
        return OutputProjectionPartition(
            group_start=tp_rank * group_count,
            group_count=group_count,
            ranks_per_group=1,
            input_offset=0,
            input_width=group_input_width,
        )

    if tp_degree % output_groups:
        raise ValueError(
            f"tp_degree={tp_degree} must be divisible by "
            f"output_groups={output_groups} when tp_degree > output_groups"
        )
    ranks_per_group = tp_degree // output_groups
    if group_input_width % ranks_per_group:
        raise ValueError(
            f"input width per output group={group_input_width} must be divisible "
            f"by ranks_per_group={ranks_per_group}"
        )
    input_width = group_input_width // ranks_per_group
    return OutputProjectionPartition(
        group_start=tp_rank // ranks_per_group,
        group_count=1,
        ranks_per_group=ranks_per_group,
        input_offset=(tp_rank % ranks_per_group) * input_width,
        input_width=input_width,
    )


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
            (num_experts, self.ep_degree, "num_experts", "ep_degree"),
            (expert_intermediate_size, self.expert_tp_degree,
             "expert_intermediate_size", "expert_tp_degree"),
        )
        for size, degree, size_name, degree_name in checks:
            if size % degree:
                raise ValueError(
                    f"{size_name}={size} must be divisible by {degree_name}={degree}"
                )
        # Above ``output_groups`` the groups are replicated across rank subsets
        # while their input columns are split. The attention constructor, which
        # also knows ``head_dim``, validates the resulting input width.
        if self.tp_degree <= output_groups:
            output_size, degree = output_groups, self.tp_degree
            size_name, degree_name = "o_groups", "tp_degree"
        else:
            output_size, degree = self.tp_degree, output_groups
            size_name, degree_name = "tp_degree", "o_groups"
        if output_size % degree:
            raise ValueError(
                f"{size_name}={output_size} must be divisible by "
                f"{degree_name}={degree}"
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
