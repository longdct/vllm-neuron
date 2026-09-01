# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.parallel import (
    DeepseekV4ParallelTopology,
    resolve_output_projection_partition,
)
from vllm_neuron.model.deepseek_v4.weight_loaders import load_checkpoint_weights


@pytest.mark.parametrize(
    "tp,dp,ep,expert_tp",
    [(1, 1, 1, 1), (2, 1, 1, 2), (2, 1, 2, 1), (2, 2, 2, 2), (2, 2, 4, 1)],
)
def test_valid_topologies(tp, dp, ep, expert_tp):
    topology = DeepseekV4ParallelTopology(tp, dp, ep)
    topology.validate(
        num_heads=64,
        output_groups=8,
        num_experts=32,
        expert_intermediate_size=2048,
    )
    assert topology.expert_tp_degree == expert_tp


@pytest.mark.parametrize("tp,ep", [(8, 1), (32, 2), (64, 4)])
def test_production_tp_ep_topologies_accept_grouped_output_projection(tp, ep):
    topology = DeepseekV4ParallelTopology(tp_degree=tp, ep_degree=ep)
    topology.validate(
        num_heads=64,
        output_groups=8,
        num_experts=256,
        expert_intermediate_size=2048,
    )
    assert topology.expert_tp_degree == (16 if tp > 8 else 8)


@pytest.mark.parametrize(
    ("tp", "rank", "expected"),
    [
        (1, 0, (0, 8, 1, 0, 4096)),
        (8, 7, (7, 1, 1, 0, 4096)),
        (16, 3, (1, 1, 2, 2048, 2048)),
        (32, 17, (4, 1, 4, 1024, 1024)),
        (64, 63, (7, 1, 8, 3584, 512)),
    ],
)
def test_output_projection_partition_covers_tp64(tp, rank, expected):
    partition = resolve_output_projection_partition(
        tp_degree=tp,
        tp_rank=rank,
        output_groups=8,
        total_input_width=64 * 512,
    )
    assert (
        partition.group_start,
        partition.group_count,
        partition.ranks_per_group,
        partition.input_offset,
        partition.input_width,
    ) == expected


@pytest.mark.parametrize("tp_degree", [1, 8, 16, 32, 64])
def test_output_projection_tp_partials_sum_to_unsharded_result(tp_degree):
    """The existing TP all-reduce reconstructs the official projection."""
    torch.manual_seed(0)
    tokens, output_groups, group_width, lora_rank, hidden = 3, 8, 16, 5, 7
    total_width = output_groups * group_width
    x = torch.randn(tokens, total_width, dtype=torch.float64)
    o_a = torch.randn(
        output_groups * lora_rank, group_width, dtype=torch.float64
    )
    o_b = torch.randn(hidden, output_groups * lora_rank, dtype=torch.float64)
    grouped = x.view(tokens, output_groups, group_width)
    projected = torch.cat(
        [
            torch.nn.functional.linear(
                grouped[:, group],
                o_a[group * lora_rank : (group + 1) * lora_rank],
            )
            for group in range(output_groups)
        ],
        dim=-1,
    )
    expected = torch.nn.functional.linear(projected, o_b)

    partials = []
    rank_width = total_width // tp_degree
    for rank in range(tp_degree):
        partition = resolve_output_projection_partition(
            tp_degree=tp_degree,
            tp_rank=rank,
            output_groups=output_groups,
            total_input_width=total_width,
        )
        local_x = x[:, rank * rank_width : (rank + 1) * rank_width].view(
            tokens, partition.group_count, partition.input_width
        )
        local_projected = []
        for local_group in range(partition.group_count):
            group = partition.group_start + local_group
            rows = o_a[group * lora_rank : (group + 1) * lora_rank]
            columns = rows[
                :,
                partition.input_offset : (
                    partition.input_offset + partition.input_width
                ),
            ]
            local_projected.append(
                torch.nn.functional.linear(local_x[:, local_group], columns)
            )
        local_projected = torch.cat(local_projected, dim=-1)
        start = partition.group_start * lora_rank
        end = start + partition.group_count * lora_rank
        partials.append(
            torch.nn.functional.linear(local_projected, o_b[:, start:end])
        )

    torch.testing.assert_close(sum(partials), expected)


@pytest.mark.parametrize(
    "kwargs,sizes,match",
    [
        ({"tp_degree": 3}, (8, 6, 9, 12), "num_attention_heads"),
        ({"tp_degree": 2, "dp_degree": 1, "ep_degree": 4}, (8, 8, 8, 8), "world_size"),
        ({"tp_degree": 3, "ep_degree": 3}, (6, 6, 8, 12), "num_experts"),
        ({"tp_degree": 4}, (8, 8, 8, 10), "expert_intermediate_size"),
    ],
)
def test_invalid_topologies_fail_before_loading(kwargs, sizes, match):
    topology = DeepseekV4ParallelTopology(**kwargs)
    with pytest.raises(ValueError, match=match):
        topology.validate(
            num_heads=sizes[0],
            output_groups=sizes[1],
            num_experts=sizes[2],
            expert_intermediate_size=sizes[3],
        )


class _Moe(torch.nn.Module):
    def __init__(self, ep_rank, expert_tp_rank):
        super().__init__()
        self.local_start = ep_rank * 2
        self.local_end = self.local_start + 2
        self.expert_tp_degree = 2
        self.expert_tp_rank = expert_tp_rank
        self.routed_gate_up = torch.nn.Parameter(torch.zeros(2, 3, 2, 2))
        self.routed_down = torch.nn.Parameter(torch.zeros(2, 2, 3))


class _Layer(torch.nn.Module):
    def __init__(self, ep_rank, expert_tp_rank):
        super().__init__()
        self.moe = _Moe(ep_rank, expert_tp_rank)


class _Inner(torch.nn.Module):
    def __init__(self, ep_rank, expert_tp_rank):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(ep_rank, expert_tp_rank)])


class _Model(torch.nn.Module):
    def __init__(self, ep_rank, expert_tp_rank):
        super().__init__()
        self.model = _Inner(ep_rank, expert_tp_rank)


def test_ep_filter_and_expert_tp_slices_reconstruct_checkpoint():
    sources = []
    originals = {}
    for expert in range(4):
        for leaf, offset in (("w1", 0), ("w3", 100), ("w2", 200)):
            shape = (4, 3) if leaf != "w2" else (3, 4)
            tensor = torch.arange(12.0).reshape(shape) + offset + expert * 1000
            originals[expert, leaf] = tensor
            sources.append((f"layers.0.ffn.experts.{expert}.{leaf}.weight", tensor))

    ranks = [[_Model(ep, tp) for tp in range(2)] for ep in range(2)]
    for row in ranks:
        for model in row:
            load_checkpoint_weights(model, sources)

    for ep, row in enumerate(ranks):
        for tp, model in enumerate(row):
            moe = model.model.layers[0].moe
            for local in range(2):
                expert = ep * 2 + local
                start = tp * 2
                torch.testing.assert_close(
                    moe.routed_gate_up[local, :, 0],
                    originals[expert, "w1"].T[:, start : start + 2],
                )
                torch.testing.assert_close(
                    moe.routed_gate_up[local, :, 1],
                    originals[expert, "w3"].T[:, start : start + 2],
                )
                torch.testing.assert_close(
                    moe.routed_down[local],
                    originals[expert, "w2"].T[start : start + 2],
                )
