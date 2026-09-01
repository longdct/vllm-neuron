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
    [
        (1, 1, 1, 1),
        (2, 1, 1, 2),
        (2, 1, 2, 1),
        (2, 2, 2, 2),
        (2, 2, 4, 1),
        (16, 1, 1, 16),
        (32, 1, 2, 16),
        (64, 1, 4, 16),
    ],
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


@pytest.mark.parametrize(
    "tp_rank,tp_degree,expected",
    [
        (0, 1, (0, 8, 1, 0, 4096)),
        (7, 8, (7, 1, 1, 0, 4096)),
        (0, 16, (0, 1, 2, 0, 2048)),
        (1, 16, (0, 1, 2, 2048, 2048)),
        (15, 16, (7, 1, 2, 2048, 2048)),
        (31, 32, (7, 1, 4, 3072, 1024)),
        (63, 64, (7, 1, 8, 3584, 512)),
    ],
)
def test_output_projection_partition(tp_rank, tp_degree, expected):
    partition = resolve_output_projection_partition(
        tp_degree=tp_degree,
        tp_rank=tp_rank,
        output_groups=8,
        total_input_width=32768,
    )
    assert (
        partition.group_start,
        partition.group_count,
        partition.ranks_per_group,
        partition.input_offset,
        partition.input_width,
    ) == expected


def test_output_projection_rejects_incompatible_tp_degree():
    with pytest.raises(ValueError, match="tp_degree=12 must be divisible"):
        resolve_output_projection_partition(
            tp_degree=12,
            tp_rank=0,
            output_groups=8,
            total_input_width=96,
        )


@pytest.mark.parametrize("tp_degree", [1, 8, 16, 32, 64])
def test_output_projection_tp_partials_sum_to_unsharded_result(tp_degree):
    """The TP all-reduce must reconstruct the official grouped projection."""
    torch.manual_seed(0)
    tokens, output_groups, group_width, lora_rank, hidden = 3, 8, 16, 5, 7
    total_width = output_groups * group_width
    x = torch.randn(tokens, total_width, dtype=torch.float64)
    o_a = torch.randn(
        output_groups * lora_rank, group_width, dtype=torch.float64
    )
    o_b = torch.randn(hidden, output_groups * lora_rank, dtype=torch.float64)

    grouped = x.view(tokens, output_groups, group_width)
    reference_a = torch.cat(
        [
            torch.nn.functional.linear(
                grouped[:, group],
                o_a[group * lora_rank : (group + 1) * lora_rank],
            )
            for group in range(output_groups)
        ],
        dim=-1,
    )
    reference = torch.nn.functional.linear(reference_a, o_b)

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
        local_a = []
        for local_group in range(partition.group_count):
            group = partition.group_start + local_group
            rows = o_a[group * lora_rank : (group + 1) * lora_rank]
            columns = rows[
                :,
                partition.input_offset : (
                    partition.input_offset + partition.input_width
                ),
            ]
            local_a.append(
                torch.nn.functional.linear(local_x[:, local_group], columns)
            )
        local_a = torch.cat(local_a, dim=-1)
        start = partition.group_start * lora_rank
        end = start + partition.group_count * lora_rank
        partials.append(torch.nn.functional.linear(local_a, o_b[:, start:end]))

    torch.testing.assert_close(sum(partials), reference)


class _OutputProjectionAttention(torch.nn.Module):
    def __init__(self, *, tp_degree, tp_rank, total_width=64, output_groups=8):
        super().__init__()
        self.world_size = tp_degree
        self.topology = DeepseekV4ParallelTopology(
            tp_degree=tp_degree,
            tp_rank=tp_rank,
            expert_tp_rank=tp_rank,
        )
        self.output_partition = resolve_output_projection_partition(
            tp_degree=tp_degree,
            tp_rank=tp_rank,
            output_groups=output_groups,
            total_input_width=total_width,
        )
        self.o_lora_rank = 3
        self.heads_per_rank = total_width // tp_degree
        self.q_b_proj = torch.nn.Linear(2, self.heads_per_rank, bias=False)
        self.o_a_proj = torch.nn.Linear(
            self.output_partition.input_width,
            self.output_partition.group_count * self.o_lora_rank,
            bias=False,
        )
        self.o_b_proj = torch.nn.Linear(
            self.output_partition.group_count * self.o_lora_rank, 5, bias=False
        )
        self.sinks = torch.nn.Parameter(torch.zeros(self.heads_per_rank))

        from vllm_neuron.model.deepseek_v4.model import DeepseekV4Attention

        DeepseekV4Attention._setup_weight_loaders(self)


class _OutputProjectionLayer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.attention = _OutputProjectionAttention(**kwargs)


class _OutputProjectionInner(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList([_OutputProjectionLayer(**kwargs)])


class _OutputProjectionModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = _OutputProjectionInner(**kwargs)


@pytest.mark.parametrize(
    "tp_degree,tp_rank",
    [
        (16, 0),
        (16, 1),
        (16, 15),
        (32, 0),
        (32, 3),
        (32, 31),
        (64, 0),
        (64, 7),
        (64, 63),
    ],
)
def test_large_tp_output_projection_loaders_slice_rows_and_columns(
    tp_degree, tp_rank
):
    """Ranks sharing a group replicate o_b while splitting o_a columns."""
    output_groups, lora_rank, group_width = 8, 3, 8
    full_o_a = torch.arange(
        output_groups * lora_rank * group_width, dtype=torch.float32
    ).view(output_groups * lora_rank, group_width)
    full_o_b = torch.arange(
        5 * output_groups * lora_rank, dtype=torch.float32
    ).view(5, output_groups * lora_rank)
    model = _OutputProjectionModel(tp_degree=tp_degree, tp_rank=tp_rank)
    load_checkpoint_weights(
        model,
        [
            ("layers.0.attn.wo_a.weight", full_o_a),
            ("layers.0.attn.wo_b.weight", full_o_b),
        ],
    )

    partition = model.model.layers[0].attention.output_partition
    row_start = partition.group_start * lora_rank
    row_end = row_start + lora_rank
    column_start = partition.input_offset
    column_end = column_start + partition.input_width
    torch.testing.assert_close(
        model.model.layers[0].attention.o_a_proj.weight,
        full_o_a[row_start:row_end, column_start:column_end],
    )
    torch.testing.assert_close(
        model.model.layers[0].attention.o_b_proj.weight,
        full_o_b[:, row_start:row_end],
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


@pytest.mark.parametrize("tp_degree", [16, 32, 64])
def test_output_projection_shards_tile_checkpoint_under_resolved_topology(
    monkeypatch, tp_degree
):
    """End-to-end resolved TP ranks cover every grouped o_a column once."""
    output_groups, total_width = 8, 512
    group_width = total_width // output_groups
    coverage = torch.zeros(output_groups, group_width, dtype=torch.int64)

    for rank in range(tp_degree):
        topology = _resolve_as_rank(
            monkeypatch, tp_degree=tp_degree, tp_rank=rank
        )
        partition = resolve_output_projection_partition(
            tp_degree=topology.tp_degree,
            tp_rank=topology.tp_rank,
            output_groups=output_groups,
            total_input_width=total_width,
        )
        group_end = partition.group_start + partition.group_count
        input_end = partition.input_offset + partition.input_width
        coverage[
            partition.group_start:group_end,
            partition.input_offset:input_end,
        ] += 1

    torch.testing.assert_close(coverage, torch.ones_like(coverage))


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
