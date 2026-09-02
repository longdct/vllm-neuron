# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.functional.sampling import sample
from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig
from vllm_neuron.nn.sampler import Sampler


def _deterministic_oracle(logits, temperature, top_k, top_p, max_top_k=256):
    """Small CPU oracle for the device sampler's fixed-quantile policy."""
    if temperature < 1e-5:
        return int(torch.argmax(logits))
    active = min(max_top_k, logits.numel())
    values, indices = torch.topk(logits, active)
    if top_k > 0:
        values[top_k:] = -3000.0
    probs = torch.softmax(values / temperature, dim=-1)
    remove = probs.cumsum(-1) > top_p
    remove[0] = False
    probs[remove] = 0
    probs /= probs.sum()
    sampled = int((0.5 > probs.cumsum(-1)).sum())
    return int(indices[sampled])


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p"),
    [(0.0, -1, 1.0), (0.8, 3, 1.0), (0.8, -1, 0.7), (1.3, -1, 1.0)],
)
def test_generic_device_sampling_matches_fixed_cpu_oracle(
    temperature, top_k, top_p
):
    logits = torch.tensor([[1.1, -0.4, 2.7, 0.3, 1.8, -2.0]])
    actual = sample(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        deterministic=True,
        all_greedy=False,
        max_top_k=256,
    )
    expected = _deterministic_oracle(
        logits[0].clone(), temperature, top_k, top_p
    )
    assert actual.tolist() == [expected]


def test_mixed_sampling_parameters_share_one_generic_graph_contract():
    logits = torch.tensor(
        [
            [1.0, 4.0, 4.0, 0.0, -1.0, 2.0],
            [0.1, 0.2, 3.0, 2.0, 1.0, -1.0],
            [2.0, 1.0, 0.5, 0.2, -0.5, -1.0],
            [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        ]
    )
    temperatures = torch.tensor([0.0, 0.8, 0.9, 1.2])
    top_ks = torch.tensor([-1, 2, -1, 4], dtype=torch.int32)
    top_ps = torch.tensor([1.0, 1.0, 0.7, 0.85])
    actual = sample(
        logits,
        temperature=temperatures,
        top_k=top_ks,
        top_p=top_ps,
        deterministic=True,
        all_greedy=False,
        max_top_k=256,
    )
    expected = [
        _deterministic_oracle(
            logits[row].clone(),
            float(temperatures[row]),
            int(top_ks[row]),
            float(top_ps[row]),
        )
        for row in range(logits.shape[0])
    ]
    assert actual.tolist() == expected


def test_generic_greedy_uses_cpu_argmax_tie_breaking():
    logits = torch.tensor([[5.0, 5.0, 5.0, 1.0]])
    token = sample(
        logits,
        temperature=torch.tensor([0.0]),
        deterministic=True,
        all_greedy=False,
    )
    assert token.tolist() == [0]


def test_sampler_masks_padded_vocabulary_rows():
    sampler = Sampler(
        OnDeviceSamplingConfig(all_greedy=True, max_top_k=256), vocab_size=5
    )
    # Padding would win without the explicit vocabulary mask.
    logits = torch.tensor([[-5.0, -4.0, -3.0, -2.0, -1.0, 9.0, 8.0, 7.0]])
    assert sampler(logits).tolist() == [4]


def test_sampler_masks_only_padding_on_the_last_vocabulary_shard(monkeypatch):
    import vllm_neuron.nn.sampler as sampler_module

    captured = {}

    def inspect_logits(logits, **kwargs):
        captured["logits"] = logits
        return torch.argmax(logits, dim=-1)

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(sampler_module, "sample", inspect_logits)
    sampler = Sampler(
        OnDeviceSamplingConfig(all_greedy=True, max_top_k=256),
        process_group=object(),
        vocab_size=6,
    )
    logits = torch.tensor([[-5.0, -4.0, 9.0, 8.0]])
    assert sampler(logits, tp_rank=torch.tensor(1)).tolist() == [1]
    assert captured["logits"][0, :2].tolist() == [-5.0, -4.0]
    assert torch.isneginf(captured["logits"][0, 2:]).all()


def _patch_gather(monkeypatch, world_size, replacement=None):
    """Patch the collective the distributed reductions use.

    Records the gather dims so the negative-dim normalization stays pinned.
    """
    import importlib

    argmax_module = importlib.import_module("vllm_neuron.functional.argmax")
    gathered_dims = []

    def fake_all_gather(tensor, dim, group):
        gathered_dims.append(dim)
        if replacement is not None:
            return replacement
        return torch.cat([tensor] * world_size, dim=dim)

    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group: world_size
    )
    monkeypatch.setattr(argmax_module, "all_gather_tensor", fake_all_gather)
    return argmax_module, gathered_dims


def test_distributed_argmax_normalizes_negative_gather_dimension(monkeypatch):
    """A negative gather dim must not reach the collective.

    ``_maybe_view_chunk_cat``, which the collective reconstructs its result
    with, is not negative-dim safe -- it re-appends the whole shape and raises.
    """
    argmax_module, gathered_dims = _patch_gather(monkeypatch, world_size=8)
    logits = torch.tensor(
        [
            [1.0, 3.0, 2.0, 0.0],
            [4.0, 2.0, 1.0, 0.0],
        ]
    )
    actual = argmax_module.argmax(
        logits, dim=-1, gather_dim=-1, process_group=object()
    )
    assert gathered_dims == [1]
    assert actual.tolist() == [1, 0]


def test_distributed_argmax_returns_the_globally_winning_index(monkeypatch):
    """The index must come from whichever shard actually holds the maximum.

    This is the regression the on-device defect produced: each rank reduced
    over only the shards on its own physical Neuron device, so it returned an
    index from its own half of the vocabulary rather than the global one. Here
    the winner is planted in the third shard of four, where a half-blind
    reduction cannot find it.
    """
    shard, shards = 10, 4
    gathered = torch.zeros(1, shard * shards)
    gathered[0, 2 * shard + 2] = 9.0  # the only maximum, in shard 2
    argmax_module, _ = _patch_gather(monkeypatch, world_size=shards, replacement=gathered)

    actual = argmax_module.argmax(
        torch.zeros(1, shard), dim=-1, gather_dim=-1, process_group=object()
    )
    assert actual.tolist() == [2 * shard + 2]


def test_distributed_argmax_is_a_plain_argmax_over_the_gathered_shards(monkeypatch):
    """Ties resolve to the lowest global index, matching torch.argmax."""
    shard, shards = 4, 2
    gathered = torch.tensor([[0.0, 5.0, 1.0, 2.0, 0.0, 5.0, 1.0, 2.0]])
    argmax_module, _ = _patch_gather(monkeypatch, world_size=shards, replacement=gathered)

    actual = argmax_module.argmax(
        torch.zeros(1, shard), dim=-1, gather_dim=-1, process_group=object()
    )
    assert actual.tolist() == [1]
