#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the tiny Qwen3.5 checkpoint through the real vLLM engine on Neuron.

This is the device path, not the CPU oracle: CPU mode substitutes torch
fallbacks for the NKI kernels, so agreement there says nothing about what the
device compiles (see the plan's section 4.4).

Device selection is explicit and mandatory: the run aborts unless
``NEURON_VISIBLE_DEVICES`` is set, so it cannot wander onto a device that is
already serving a model. That variable is named for devices but parsed as
logical *core* IDs, and it needs exactly ``tensor_parallel_size`` entries.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

# TorchNeuron Native must register its PrivateUse1 device before vLLM resolves
# the active platform; importing vLLM first can leave current_platform as the
# unspecified fallback in editable development installs.
import vllm_neuron  # noqa: F401
import vllm.platforms as _vllm_platforms

if not _vllm_platforms.current_platform.device_type:
    _vllm_platforms._current_platform = None

from vllm import LLM, ModelRegistry, SamplingParams  # noqa: E402


class _Qwen3_5TextShim(nn.Module):
    """Parent-process stand-in that declares "text generation, not multimodal".

    Never instantiated: vLLM's parent only *inspects* the registered class,
    and ``NeuronWorker`` registers the real implementation over this name
    before any model is built.

    It is needed because the parent resolves the architecture against vLLM's
    own registry, where Qwen3.5 exists only as a multimodal model. Left alone,
    the parent builds a Qwen3-VL preprocessing stack and demands an image
    processor, a video processor and a tokenizer -- none of which a text-only
    decode run will ever use. The real Neuron class cannot be registered here
    instead: its ``__init__`` takes ``(hf_config, neuron_config)`` and so fails
    vLLM's ``VllmModel`` protocol, which wants ``vllm_config``.

    Every other Neuron model shadows an architecture vLLM already ships, so
    the parent finds a valid class without help. Qwen3.5's text decoder is the
    first that does not, and fixing that properly is a plugin-level decision
    about registration -- deliberately not made here, in a bring-up tool.
    """

    def __init__(self, vllm_config, prefix: str = "") -> None:
        super().__init__()
        raise AssertionError(
            "the parent-process shim must never be instantiated; the worker "
            "registers the real Qwen3.5 model over this name"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor): ...

    def compute_logits(self, hidden_states: torch.Tensor): ...


def register_parent_shim() -> None:
    for arch in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"):
        ModelRegistry.register_model(arch, _Qwen3_5TextShim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--output", type=Path)
    # Without this vLLM sizes the KV cache to fill HBM -- ~21 GiB of a 24 GiB
    # logical core for an 8-layer tiny model -- and the compile then OOMs
    # asking for a few GiB more. A handful of blocks covers max_model_len.
    parser.add_argument("--num-gpu-blocks-override", type=int, default=16)
    args = parser.parse_args()

    if "NEURON_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit(
            "refusing to run unpinned: set NEURON_VISIBLE_DEVICES to the logical "
            "core IDs this run may use, so it cannot land on a device that is "
            "already serving a model."
        )

    register_parent_shim()

    config = json.loads((args.checkpoint / "config.json").read_text())
    # Released checkpoints nest the decoder under text_config; a bare text
    # config is also accepted.
    vocab_size = config.get("text_config", config)["vocab_size"]

    llm = LLM(
        model=str(args.checkpoint),
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        block_size=args.block_size,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        async_scheduling=False,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [args.max_model_len],
                "on_device_sampling_config": None,
            }
        },
    )

    prompt_ids = [(i * 7 + 3) % vocab_size for i in range(args.prompt_length)]
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    token_ids = list(outputs[0].outputs[0].token_ids)

    if len(token_ids) != args.max_tokens:
        raise SystemExit(f"expected {args.max_tokens} tokens, got {token_ids}")
    if any(t < 0 or t >= vocab_size for t in token_ids):
        raise SystemExit(f"token out of vocabulary range: {token_ids}")

    result = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "visible_cores": os.environ["NEURON_VISIBLE_DEVICES"],
        "prompt_length": len(prompt_ids),
        "token_ids": token_ids,
    }
    print("RESULT " + json.dumps(result))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
