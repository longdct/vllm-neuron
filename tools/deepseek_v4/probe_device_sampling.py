#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the compiled on-device sampler actually return the token it chose?

On-device sampling on DeepSeek-V4 returns token ids far outside the vocabulary
(`967439869`, `-1167771948`). Read as raw int32 those are float32 bit patterns
for `+3.24e-4` and `-8.74e-4` -- inside the `(-1e-3, 1e-3)` interval vLLM's
dummy-weight initializer draws from. That is not a miscomputed index; it is a
buffer nobody wrote, read through an int32 view.

The full model cannot tell apart the three ways that can happen: a wrong
sampler, an output the NEFF never writes, or an async readback of recycled
memory. This probe removes the model. It compiles *only* the sampler over
logits with a known unique maximum, so a mismatch here convicts the sampler
itself and a match sends the investigation to the graph's output plumbing.

The mismatch report bit-casts whatever came back to float32, because that
reinterpretation is what identified the defect in the first place and is worth
having automatically rather than by hand.

At ``--tensor-parallel-size > 1`` the probe runs one process per rank, each
holding one vocabulary shard, and is the direct regression test for the
collective defect: a narrow all-gather across a TP group spanning physical
Neuron devices returned a buffer in which each rank saw only its own device's
shards. The tell is not just a wrong token -- it is that the ranks *disagree*,
which cannot happen when the reduction spans the group. Every rank therefore
writes its own answer and the launcher compares them.

Distributed mode mirrors ``NeuronWorker``'s startup exactly (per-rank
``NEURON_RT_VISIBLE_CORES``, ``init_neuron_distributed_environment``,
``rendezvous_ccom_bootstrap``, ``torch_neuronx.set_device``): a group built any
other way is not the group the model uses, and a bare ``ProcessGroup`` from a
plain ``init_process_group`` is not even traceable by dynamo here.

Distributed mode requires the TP group's Neuron devices to be **connected in
the interconnect**, which is a real constraint and not a quirk of this script.
On a disconnected group the runtime aborts in ``encd_mesh_add_wr_barrier``
("Assertion `event->evt_type == EVT_SYNC` failed") before producing anything.
That is the same defect the sampler hit, in a louder form: a small collective
NEFF on such a group aborts, while a full model NEFF on it silently returns
partial data. Measured on trn2:

===================  ===========  ============  ==========================
cores                devices      connected?    probe
===================  ===========  ============  ==========================
``12-15``            3            single        passes
``12-19``            3, 4         **no**        runtime abort
``16-23``            4, 5         yes           passes
``20-27``            5, 6         yes           passes
``16-31``            4, 5, 6, 7   yes (ring)    passes
===================  ===========  ============  ==========================

``neuron-ls -j`` reports each device's neighbours; devices 3 and 4 are not in
each other's lists. Pick cores whose devices form one connected component.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import torch

from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig
from vllm_neuron.nn.sampler import Sampler


def build_logits(batch: int, vocab: int, seed: int) -> torch.Tensor:
    """Logits whose per-row argmax is unique and known.

    A unique maximum matters: tie-breaking between the device reduce and
    ``torch.argmax`` is a legitimate difference, and would muddy the signal
    this probe exists to produce.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = torch.rand((batch, vocab), generator=generator) * 2.0 - 1.0
    winners = torch.randint(0, vocab, (batch,), generator=generator)
    logits[torch.arange(batch), winners] = 10.0
    return logits, winners


def as_float_bits(token: int) -> float | None:
    """Reinterpret an int32 token id as float32, or None if it does not fit."""
    try:
        return struct.unpack("<f", struct.pack("<i", int(token)))[0]
    except (struct.error, OverflowError, ValueError):
        return None


class RuntimeSampler(torch.nn.Module):
    def __init__(self, sampler: Sampler) -> None:
        super().__init__()
        self.sampler = sampler

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.sampler(logits)


def _init_distributed(rank: int, world_size: int, local_rank: int) -> None:
    """Bring up the Neuron distributed environment the way NeuronWorker does.

    The ordering is load-bearing and taken from
    ``NeuronWorker._init_neuron_distributed_environment_and_runtime``: gloo
    first, then the CCOM rendezvous that pins ``NEURON_RT_ROOT_COMM_ID``, then
    ``set_device`` so the native runtime latches that id.
    """
    import torch_neuronx

    from vllm_neuron.envs import get_dist_backend
    from vllm_neuron.parallel.neuron_parallel_state import (
        init_neuron_distributed_environment,
    )
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap

    # The same runtime environment NeuronWorker establishes. These are not
    # tuning knobs here: without the execution-barrier setting the runtime
    # aborts inside `encd_mesh_add_wr_barrier`, and a probe running under a
    # different runtime configuration from the model is not evidence about the
    # model.
    from vllm_neuron.vllm.platform import NeuronPlatform

    # Each process has one visible core, so torch reports one device; vLLM
    # needs the node-wide count. NeuronWorker sets this before touching the
    # runtime and the collective resources do not build without it.
    NeuronPlatform.set_device_count(world_size)

    os.environ["NEURON_RT_MAP_HBM"] = "1"
    os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", "32")
    os.environ.setdefault("NEURON_RT_IO_RING_CACHE_SIZE", "32")
    os.environ.setdefault("NEURON_RT_DISABLE_EXECUTION_BARRIER", "1")

    init_neuron_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend=get_dist_backend(),
        tensor_parallel_size=world_size,
    )
    rendezvous_ccom_bootstrap(data_parallel_index=0)
    # The device index is the local rank, not 0: torch_neuronx indexes by
    # position within NEURON_VISIBLE_DEVICES, so rank 7 must use neuron:7 even
    # though NEURON_RT_VISIBLE_CORES pins this process to a single core
    # ("Device index 0 is out of range. Valid range is 7 to 7").
    torch_neuronx.set_device(local_rank)


def run_distributed(args) -> int:
    """One rank of a sharded-vocabulary sampling probe."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if args.vocab % world_size:
        raise SystemExit(f"vocab {args.vocab} is not divisible by TP {world_size}")

    # The logits are produced by a matmul inside the compiled graph rather than
    # handed in as an input. A graph containing nothing but a collective is not
    # schedulable here ("Invalid NEFF, instruction, or input"), and a sampler
    # fed a constant is not the shape of the thing being tested anyway.
    hidden_size = args.hidden
    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.randn((args.batch, hidden_size), generator=generator)
    weight = torch.randn((args.vocab, hidden_size), generator=generator) * 0.02

    # Plant one dominant row per batch entry so the global maximum is
    # unambiguous. Without it, two logits can land within a float32 ulp of each
    # other and CPU/device argmax may legitimately disagree, which would look
    # exactly like the defect under test.
    winners = [
        (i + 1) * (args.vocab // (args.batch + 1)) for i in range(args.batch)
    ]
    for row, target in enumerate(winners):
        weight[target] = hidden[row] * 5.0

    full_logits = torch.nn.functional.linear(hidden, weight)
    expected = torch.argmax(full_logits, dim=-1).tolist()
    if expected != winners:
        raise SystemExit(
            f"probe construction failed: argmax {expected} != planted {winners}"
        )

    shard = args.vocab // world_size
    local_weight = weight[rank * shard : (rank + 1) * shard].contiguous()

    # vLLM's parallel state and the Sampler's CustomOps both read the ambient
    # VllmConfig; without this context they assert at construction time.
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    # Name the device rather than letting DeviceConfig infer it: platform
    # auto-detection resolves to Unspecified in a bare process like this one,
    # even though the neuron plugin is registered.
    context = set_current_vllm_config(
        VllmConfig(device_config=DeviceConfig(device="neuron"))
    )
    context.__enter__()

    _init_distributed(rank, world_size, local_rank)

    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    sampler = Sampler(
        OnDeviceSamplingConfig(
            all_greedy=args.greedy == "all-greedy",
            max_top_k=256,
            deterministic=True,
        ),
        process_group=tp_group.device_group,
        vocab_size=args.vocab,
    )

    device = torch.device(f"neuron:{local_rank}")

    class _ShardedHead(torch.nn.Module):
        """This rank's slice of a column-parallel head, plus the sampler.

        The all-reduce is not arithmetic -- it adds an all-zero tensor. It is
        here because a NEFF whose only collective is the sampler's all-gather
        trips a runtime assertion in `encd_mesh_add_wr_barrier` when the replica
        group spans physical devices. A real decode graph always carries the
        model's own reductions alongside the sampler, so this keeps the probe on
        the shape of graph the runtime actually executes.

        No `tp_rank` is passed: the sampler only needs it to mask padded
        vocabulary rows, and this probe's vocabulary divides evenly across the
        group, so there is no padding. Feeding one in also made an SPMD pass
        report "conflicting shard indices" and skip its injection.
        """

        def __init__(self, sampler, weight, tp_group):
            super().__init__()
            self.sampler = sampler
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.tp_group = tp_group

        def forward(self, hidden_states):
            hidden_states = hidden_states + self.tp_group.all_reduce(
                torch.zeros_like(hidden_states)
            )
            logits = torch.nn.functional.linear(hidden_states, self.weight)
            return self.sampler(logits)

    compiled = torch.compile(
        _ShardedHead(sampler, local_weight, tp_group).to(device),
        backend="neuron",
        dynamic=False,
    )
    tokens = compiled(hidden.to(device))
    torch.neuron.synchronize()
    tokens = tokens.to("cpu").tolist()

    report = {
        "rank": rank,
        "world_size": world_size,
        "vocab": args.vocab,
        "shard_width": shard,
        "hidden_size": hidden_size,
        "greedy_path": args.greedy,
        "expected_tokens": expected,
        "device_tokens": tokens,
        "matches_expected": tokens == expected,
        "owning_shard_of_each_token": [t // shard for t in tokens],
        "device_tokens_as_float32": [as_float_bits(t) for t in tokens],
        "in_vocab_range": [0 <= t < args.vocab for t in tokens],
    }
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.output is not None:
        out = args.output.parent / f"{args.output.stem}-rank{rank}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return 0 if report["matches_expected"] else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=129280)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Width of the synthetic head that produces the shard logits.",
    )
    parser.add_argument(
        "--greedy",
        choices=("all-greedy", "generic"),
        default="all-greedy",
        help=(
            "'all-greedy' takes sampling.py's argmax fast path; 'generic' goes "
            "through top-k/top-p with a zero temperature, which is the path a "
            "real request with temperature=0 takes."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "At >1 this process is one rank of a sharded-vocabulary probe and "
            "expects RANK/WORLD_SIZE/LOCAL_RANK in the environment; use "
            "launch_device_sampling_probe.sh rather than running it directly."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.tensor_parallel_size > 1:
        raise SystemExit(run_distributed(args))

    logits, winners = build_logits(args.batch, args.vocab, args.seed)
    expected = torch.argmax(logits, dim=-1)
    assert torch.equal(expected, winners), "probe construction is wrong"

    sampler = Sampler(
        OnDeviceSamplingConfig(
            all_greedy=args.greedy == "all-greedy",
            max_top_k=256,
            deterministic=True,
        ),
        process_group=None,
        vocab_size=args.vocab,
    )
    cpu_tokens = RuntimeSampler(sampler)(logits).tolist()

    device = torch.device(f"neuron:{local_rank}")
    compiled = torch.compile(
        RuntimeSampler(sampler).to(device), backend="neuron", dynamic=False
    )
    device_tokens = compiled(logits.to(device))
    torch.neuron.synchronize()
    device_tokens = device_tokens.to("cpu").tolist()

    matches = device_tokens == expected.tolist()
    report = {
        "batch": args.batch,
        "vocab": args.vocab,
        "greedy_path": args.greedy,
        "expected_tokens": expected.tolist(),
        "cpu_sampler_tokens": cpu_tokens,
        "device_sampler_tokens": device_tokens,
        "device_matches_expected": matches,
        "cpu_matches_expected": cpu_tokens == expected.tolist(),
    }
    if not matches:
        report["device_tokens_as_float32"] = [
            as_float_bits(token) for token in device_tokens
        ]
        report["in_vocab_range"] = [
            0 <= token < args.vocab for token in device_tokens
        ]

    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    raise SystemExit(0 if matches else 1)


if __name__ == "__main__":
    main()
