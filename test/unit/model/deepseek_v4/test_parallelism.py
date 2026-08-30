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


class _FakeGroup:
    def __init__(self, world_size, rank_in_group):
        self.world_size = world_size
        self.rank_in_group = rank_in_group


class _FakeParallelConfig:
    def __init__(self, data_parallel_size, enable_expert_parallel):
        self.data_parallel_size = data_parallel_size
        self.enable_expert_parallel = enable_expert_parallel


class _FakeVllmConfig:
    def __init__(self, parallel_config):
        self.parallel_config = parallel_config


def _resolve_as_rank(monkeypatch, *, tp_degree, tp_rank, dp_degree=1,
                     enable_expert_parallel=False):
    """Run ``resolve_parallel_topology`` as if this process were ``tp_rank``."""
    import vllm.config
    import vllm.distributed.parallel_state

    from vllm_neuron.model.deepseek_v4.parallel import resolve_parallel_topology

    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: _FakeVllmConfig(
            _FakeParallelConfig(dp_degree, enable_expert_parallel)
        ),
    )
    monkeypatch.setattr(
        vllm.distributed.parallel_state,
        "get_tp_group",
        lambda: _FakeGroup(tp_degree, tp_rank),
    )
    return resolve_parallel_topology()


@pytest.mark.parametrize("tp_degree", [2, 8])
def test_expert_tp_rank_is_distinct_per_rank_without_expert_parallel(
    monkeypatch, tp_degree
):
    """Every rank must own a *different* routed-expert column shard.

    ``load_checkpoint_weights`` narrows the routed expert tensors at
    ``expert_tp_rank * shard_size``.  This used to be left at its default of 0
    on every rank when ``--enable-expert-parallel`` was off, so all ranks loaded
    shard 0 and the TP all-reduce in ``DeepseekV4MoE.forward`` summed
    ``tp_degree`` copies of one partial -- a silently wrong MoE at TP>1, with
    attention and the shared expert (which do plumb ``tp_rank``) still correct.
    """
    ranks = [
        _resolve_as_rank(monkeypatch, tp_degree=tp_degree, tp_rank=rank)
        for rank in range(tp_degree)
    ]
    assert [t.expert_tp_rank for t in ranks] == list(range(tp_degree))
    assert all(t.expert_tp_degree == tp_degree for t in ranks)
    # The shard indices must tile the intermediate dimension exactly once.
    assert sorted({t.expert_tp_rank for t in ranks}) == list(range(tp_degree))


def test_expert_tp_degree_ignores_data_parallel_replicas_without_ep(monkeypatch):
    """DP replicas each hold a whole model; they do not share an expert shard.

    ``DeepseekV4MoE.forward`` only all-reduces over the TP group when EP is off,
    so claiming ``expert_tp_degree == tp_degree * dp_degree`` would drop the
    shards held by the other replicas from every sum.
    """
    topology = _resolve_as_rank(
        monkeypatch, tp_degree=4, tp_rank=3, dp_degree=2
    )
    assert topology.expert_tp_degree == 4
    assert topology.expert_tp_rank == 3


def test_routed_expert_shards_tile_the_checkpoint_under_resolved_topology(
    monkeypatch,
):
    """End-to-end: resolved topology -> loader -> shards reconstruct the whole.

    ``test_ep_filter_and_expert_tp_slices_reconstruct_checkpoint`` passes
    ``expert_tp_rank`` in by hand, so it stayed green while the value the real
    code path produced was always 0.  This drives the loader from
    ``resolve_parallel_topology`` instead.
    """
    sources = []
    originals = {}
    for expert in range(2):
        for leaf, offset in (("w1", 0), ("w3", 100), ("w2", 200)):
            shape = (4, 3) if leaf != "w2" else (3, 4)
            tensor = torch.arange(12.0).reshape(shape) + offset + expert * 1000
            originals[expert, leaf] = tensor
            sources.append((f"layers.0.ffn.experts.{expert}.{leaf}.weight", tensor))

    gate_shards, up_shards, down_shards = [], [], []
    for rank in range(2):
        topology = _resolve_as_rank(monkeypatch, tp_degree=2, tp_rank=rank)
        model = _Model(topology.ep_rank, topology.expert_tp_rank)
        model.model.layers[0].moe.expert_tp_degree = topology.expert_tp_degree
        load_checkpoint_weights(model, sources)
        moe = model.model.layers[0].moe
        gate_shards.append(moe.routed_gate_up[:, :, 0].clone())
        up_shards.append(moe.routed_gate_up[:, :, 1].clone())
        down_shards.append(moe.routed_down.clone())

    for expert in range(2):
        torch.testing.assert_close(
            torch.cat([s[expert] for s in gate_shards], dim=-1),
            originals[expert, "w1"].T,
        )
        torch.testing.assert_close(
            torch.cat([s[expert] for s in up_shards], dim=-1),
            originals[expert, "w3"].T,
        )
        torch.testing.assert_close(
            torch.cat([s[expert] for s in down_shards], dim=0),
            originals[expert, "w2"].T,
        )


def test_column_sharded_experts_sum_to_the_unsharded_result():
    """The all-reduce of per-rank partials must equal the TP=1 MoE output.

    This is the numerical statement the device run violated: with every rank
    holding shard 0, the sum was ``tp_degree`` copies of one partial instead of
    the concatenated whole.
    """
    torch.manual_seed(0)
    tokens, hidden, intermediate, tp_degree = 5, 6, 8, 4
    x = torch.randn(tokens, hidden, dtype=torch.float64)
    gate_w = torch.randn(hidden, intermediate, dtype=torch.float64)
    up_w = torch.randn(hidden, intermediate, dtype=torch.float64)
    down_w = torch.randn(intermediate, hidden, dtype=torch.float64)

    def expert(g, u, d):
        gate = (x @ g).clamp(max=10.0)
        up = (x @ u).clamp(min=-10.0, max=10.0)
        return (torch.nn.functional.silu(gate) * up) @ d

    reference = expert(gate_w, up_w, down_w)
    shard = intermediate // tp_degree
    partials = [
        expert(
            gate_w[:, r * shard : (r + 1) * shard],
            up_w[:, r * shard : (r + 1) * shard],
            down_w[r * shard : (r + 1) * shard],
        )
        for r in range(tp_degree)
    ]
    torch.testing.assert_close(sum(partials), reference)

    # And the shape the bug produced is genuinely wrong, so the test above is
    # not vacuously satisfied by any sharding.
    all_rank_zero = sum(partials[0] for _ in range(tp_degree))
    assert not torch.allclose(all_rank_zero, reference)
