# SPDX-License-Identifier: Apache-2.0

from types import MethodType

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.model import DeepseekV4MoE
from vllm_neuron.model.deepseek_v4.parallel import DeepseekV4ParallelTopology


class _ReduceGroup:
    def __init__(self, size, rank=0):
        self.size = size
        self.rank_in_group = rank
        self.all_reduce_calls = 0
        self.all_gather_calls = 0

    def all_reduce(self, value):
        self.all_reduce_calls += 1
        return value * self.size

    def all_gather(self, value, dim=0):
        self.all_gather_calls += 1
        return torch.cat([value] * self.size, dim=dim)


class _Shared(torch.nn.Module):
    def __init__(self, tp_degree, tp_group):
        super().__init__()
        self.tp_degree = tp_degree
        self.tp_group = tp_group
        self.swiglu_limit = 7.0

    def forward_local(self, hidden):
        return hidden * 3

    def forward(self, hidden):
        local = self.forward_local(hidden)
        return self.tp_group.all_reduce(local) if self.tp_degree > 1 else local


def _moe(topology, tp_group):
    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.kind = "routed_moe"
    moe.topk = 1
    moe.num_experts = 4
    moe.routed_scaling_factor = 1.0
    moe.ep_degree = topology.ep_degree
    moe.topology = topology
    moe.tp_group = tp_group
    moe.gate = torch.nn.Linear(2, 4, bias=False)
    moe.correction_bias = torch.nn.Parameter(torch.zeros(4))
    moe.shared_experts = _Shared(topology.tp_degree, tp_group)
    moe._forward_portable = MethodType(
        lambda self, hidden, affinities: hidden * 2, moe
    )
    return moe


@pytest.mark.parametrize("tp,ep", [(1, 1), (8, 1), (32, 2), (64, 4)])
def test_fused_moe_matches_legacy_two_reductions_with_one_collective(
    monkeypatch, tp, ep
):
    monkeypatch.setattr(
        "vllm_neuron.model.deepseek_v4.model.can_run_kernel", lambda tensor: False
    )
    topology = DeepseekV4ParallelTopology(tp_degree=tp, ep_degree=ep)
    group = _ReduceGroup(tp)
    moe = _moe(topology, group)
    hidden = torch.tensor([[1.0, -2.0]])
    actual = moe(hidden, torch.tensor([1]))
    # Legacy result: all_reduce(2h) + all_reduce(3h).
    expected = hidden * 5 * tp
    torch.testing.assert_close(actual, expected)
    assert group.all_reduce_calls == (0 if tp == 1 else 1)


def test_cross_dp_ep_keeps_separate_wide_ep_and_shared_tp_reductions(monkeypatch):
    monkeypatch.setattr(
        "vllm_neuron.model.deepseek_v4.model.can_run_kernel", lambda tensor: False
    )
    topology = DeepseekV4ParallelTopology(tp_degree=2, dp_degree=2, ep_degree=4)
    tp_group = _ReduceGroup(2)
    dp_group = _ReduceGroup(2)
    wide_ep_group = _ReduceGroup(4)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dp_group", lambda: dp_group
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_wide_ep_group",
        lambda: wide_ep_group,
        raising=False,
    )
    moe = _moe(topology, tp_group)
    hidden = torch.tensor([[1.0, -2.0]])
    actual = moe(hidden, torch.tensor([1]))
    # Routed: 2h reduced over wide EP (x4), then local row slice. Shared: 3h
    # reduced separately over TP (x2).
    torch.testing.assert_close(actual, hidden * 14)
    assert wide_ep_group.all_reduce_calls == 1
    assert tp_group.all_reduce_calls == 1
    assert dp_group.all_gather_calls == 2
