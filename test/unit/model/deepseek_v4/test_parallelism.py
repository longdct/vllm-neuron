# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.parallel import DeepseekV4ParallelTopology
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
