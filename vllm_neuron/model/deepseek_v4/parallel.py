# SPDX-License-Identifier: Apache-2.0
"""Parallel topology used by the DeepSeek-V4 serving implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputProjectionPartition:
    """One TP rank's slice of the grouped low-rank output projection."""

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
    """Assign whole groups up to TP8, then split each group's input columns."""
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
        return DeepseekV4ParallelTopology(
            tp_degree=tp_degree, dp_degree=dp_degree, tp_rank=tp_rank
        )

    from vllm_neuron.parallel.neuron_parallel_state import (
        get_neuron_ep_degree,
        get_neuron_ep_rank,
        get_neuron_ep_tp_group,
    )

    ep_degree = get_neuron_ep_degree()
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
