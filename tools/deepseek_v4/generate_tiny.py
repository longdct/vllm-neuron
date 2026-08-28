#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the deterministic tiny checkpoint through the real vLLM engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# TorchNeuron Native must register its PrivateUse1 device before vLLM resolves
# the active platform.  Importing vLLM first can leave current_platform as the
# unspecified fallback in source/editable development environments.
import vllm_neuron  # noqa: F401
import vllm.platforms as _vllm_platforms

# torch_neuronx currently causes vLLM's lazy platform object to be touched
# while the Neuron plugin is still importing.  Retry discovery once plugin
# initialization has completed (development TorchNeuron Native environments).
if not _vllm_platforms.current_platform.device_type:
    _vllm_platforms._current_platform = None
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument(
        "--ep-degree",
        type=int,
        help="Variable EP degree; defaults to the TP x DP world when EP is enabled.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Run without graph capture (CPU-mode oracle diagnostics only).",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        help="Capture decoder-layer and lm-head outputs for accuracy diagnosis.",
    )
    parser.add_argument(
        "--capture-attention-internals-layer",
        type=int,
        choices=range(3),
        metavar="{0,1,2}",
        help="Capture only one layer plus its attention projection internals.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=64,
        help="Model length for focused diagnostics (default: 64).",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        help="Independent scheduler/prefill chunk limit (defaults to max-model-len).",
    )
    parser.add_argument(
        "--prefill-segment-buckets",
        default=None,
        help="Comma-separated compiled prefill buckets, e.g. 512,2048,4096.",
    )
    parser.add_argument(
        "--decode-context-buckets",
        default=None,
        help="Comma-separated decode history buckets, e.g. 4096,32768,131072.",
    )
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=256,
        help=(
            "KV cache blocks (default: 256). The floor is the largest per-group value in "
            "the runner's logged max_num_blocks_per_req (16 for this config) "
            "plus a null block, so 32 is the smallest safe value here."
        ),
    )
    parser.add_argument(
        "--load-format",
        default="dummy",
        help=(
            "vLLM load format (default: dummy). The synthetic gate runs on "
            "random weights; pass 'auto' to load a real checkpoint, e.g. one "
            "built by build_tiny_from_official.py."
        ),
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Comma-separated prompt token ids (default: 1..8). Real checkpoints "
            "need ids from their own tokenizer, not the synthetic 1..8."
        ),
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        help="Materialize a synthetic prompt of exactly this many token ids.",
    )
    parser.add_argument(
        "--workload-lengths",
        help="Comma-separated synthetic prompt lengths to execute sequentially.",
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help=(
            "KV cache dtype (default: auto, i.e. the model's). The CPU path runs "
            "the model in FP32 but still stores BF16 K/V, which is the only thing "
            "that separates its logits from the transformers reference; "
            "'float32' removes that quantization so the two can be compared exactly."
        ),
    )
    args = parser.parse_args()
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if args.ep_degree is not None:
        if not args.enable_expert_parallel:
            parser.error("--ep-degree requires --enable-expert-parallel")
        if args.ep_degree < 1:
            parser.error("--ep-degree must be positive")
    if args.prompt and (args.prompt_length or args.workload_lengths):
        parser.error("--prompt cannot be combined with synthetic prompt lengths")
    if args.prompt_length is not None and args.workload_lengths:
        parser.error("--prompt-length and --workload-lengths are mutually exclusive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.max_model_len < 12:
        parser.error("--max-model-len must fit the 8-token prompt and 4 outputs")
    if args.num_gpu_blocks_override < 32:
        parser.error("--num-gpu-blocks-override must be at least 32")

    def parse_buckets(value: str | None, option: str) -> list[int] | None:
        if value is None:
            return None
        try:
            buckets = [int(item) for item in value.split(",")]
        except ValueError:
            parser.error(f"{option} must be a comma-separated list of integers")
        if not buckets or any(item <= 0 for item in buckets):
            parser.error(f"{option} entries must be positive")
        if buckets != sorted(set(buckets)):
            parser.error(f"{option} entries must be unique and increasing")
        return buckets

    max_num_batched_tokens = args.max_num_batched_tokens or args.max_model_len
    prefill_buckets = parse_buckets(
        args.prefill_segment_buckets, "--prefill-segment-buckets"
    ) or sorted({8, max_num_batched_tokens})
    decode_buckets = parse_buckets(
        args.decode_context_buckets, "--decode-context-buckets"
    )
    if prefill_buckets[-1] != max_num_batched_tokens:
        parser.error("the largest prefill segment bucket must equal --max-num-batched-tokens")
    if max_num_batched_tokens > args.max_model_len:
        parser.error("--max-num-batched-tokens cannot exceed --max-model-len")
    if decode_buckets and decode_buckets[-1] > args.max_model_len:
        parser.error("decode context buckets cannot exceed --max-model-len")

    neuron_config = {
        "num_batched_tokens_buckets": prefill_buckets,
        "on_device_sampling_config": None,
    }
    if args.ep_degree is not None:
        neuron_config["ep_degree"] = args.ep_degree
    # The runner always compiles max_model_len as an implicit final decode
    # bucket. Accepting it in this benchmark-facing list keeps the requested
    # geometry explicit without passing a redundant value to core validation.
    explicit_decode_buckets = [
        bucket for bucket in (decode_buckets or []) if bucket < args.max_model_len
    ]
    if explicit_decode_buckets:
        neuron_config["decode_context_length_buckets"] = explicit_decode_buckets
    if args.capture_dir is not None:
        if args.capture_attention_internals_layer is None:
            capture_modules = [
                "model.embed_tokens",
                *[
                    f"model.layers.{layer}.{module}"
                    for layer in range(3)
                    for module in (
                        "attn_hc",
                        "input_layernorm",
                        "attention",
                        "ffn_hc",
                        "post_attention_layernorm",
                        "moe",
                    )
                ],
                *(f"model.layers.{layer}" for layer in range(3)),
                "lm_head",
            ]
        else:
            layer = args.capture_attention_internals_layer
            prefix = f"model.layers.{layer}"
            capture_modules = [
                "model.embed_tokens",
                f"{prefix}.attn_hc",
                f"{prefix}.input_layernorm",
                f"{prefix}.attention",
                *(
                    f"{prefix}.attention.{module}"
                    for module in (
                        "q_a_proj",
                        "q_a_norm",
                        "q_b_proj",
                        "kv_proj",
                        "kv_norm",
                        "capture_q_roped",
                        "capture_kv_roped",
                        "capture_history",
                        "capture_key_valid",
                        "capture_attended_roped",
                        "capture_attended",
                        "o_a_proj",
                        "o_b_proj",
                    )
                ),
                prefix,
                "lm_head",
            ]
        neuron_config["tensor_capture"] = {
            "modules": capture_modules,
            "capture_dir": str(args.capture_dir),
        }

    llm = LLM(
        model=str(args.checkpoint),
        load_format=args.load_format,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=32,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        kv_cache_dtype=args.kv_cache_dtype,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        async_scheduling=False,
        additional_config={"neuron_config": neuron_config},
    )
    # Read the bound rather than hardcoding the synthetic model's 64-token vocab,
    # so a real checkpoint is validated against its own vocabulary.
    vocab_size = json.loads(
        (args.checkpoint / "config.json").read_text()
    )["vocab_size"]
    if args.workload_lengths:
        try:
            lengths = [int(item) for item in args.workload_lengths.split(",")]
        except ValueError:
            parser.error("--workload-lengths must be comma-separated integers")
        if not lengths or any(length <= 0 for length in lengths):
            parser.error("--workload-lengths entries must be positive")
    elif args.prompt_length is not None:
        lengths = [args.prompt_length]
    else:
        lengths = []
    prompts = (
        [[int(x) for x in args.prompt.split(",")]]
        if args.prompt
        else [
            [1 + (index % max(vocab_size - 1, 1)) for index in range(length)]
            for length in lengths
        ]
        if lengths
        else [list(range(1, 9))]
    )
    if any(len(prompt) + args.max_tokens > args.max_model_len for prompt in prompts):
        parser.error("every prompt length plus --max-tokens must fit --max-model-len")
    workload_results = []
    for prompt_ids in prompts:
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        )
        token_ids = list(outputs[0].outputs[0].token_ids)
        if len(token_ids) != args.max_tokens or any(
            token < 0 or token >= vocab_size for token in token_ids
        ):
            raise SystemExit(f"invalid generated tokens: {token_ids}")
        workload_results.append(
            {"prompt_length": len(prompt_ids), "token_ids": token_ids}
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"token_ids": workload_results[-1]["token_ids"]}
    if len(workload_results) > 1 or lengths:
        result["workloads"] = workload_results
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
