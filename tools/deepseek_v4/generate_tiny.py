#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the deterministic tiny checkpoint through the real vLLM engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

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
    parser.add_argument(
        "--quantization",
        choices=("bf16", "fp8"),
        help=(
            "Weight storage for the routed experts. 'fp8' additionally needs "
            "UNSAFE_FP8FNCAST=1 and the neuronx-cc e4m3fn cast flag."
        ),
    )
    parser.add_argument(
        "--sampling-backend",
        choices=("cpu", "device"),
        default="cpu",
        help=(
            "Where the token is chosen. 'device' compiles the sampler into the "
            "model graph; 'cpu' returns logits and samples on the host. Kept "
            "separate from --async-scheduling on purpose: benchmark_decode.py "
            "welded the two together, which is why an on-device sampling "
            "defect could never be isolated from the async readback."
        ),
    )
    parser.add_argument(
        "--async-scheduling",
        choices=("on", "off"),
        default="off",
        help=(
            "Return sampled tokens as device futures materialized after the "
            "step. Only meaningful with --sampling-backend device."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--num-seqs-buckets",
        help=(
            "Comma-separated compiled decode batch buckets. The last value "
            "must equal --max-num-seqs; e.g. 8 compiles only batch 8."
        ),
    )
    parser.add_argument("--block-size", type=int, default=32)
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
        "--capture-modules",
        help=(
            "Comma-separated module names to capture. Requires --capture-dir; "
            "useful for focused TP comparisons without capturing every layer."
        ),
    )
    parser.add_argument(
        "--capture-attention-internals-layer",
        type=int,
        metavar="LAYER",
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
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help=(
            "Fraction of device memory vLLM may use (default: 0.9). Raise this "
            "when engine start fails its KV-cache memory check: the admission "
            "check runs on the profiled budget, before "
            "--num-gpu-blocks-override applies, so the override alone cannot "
            "get past it."
        ),
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
    parser.add_argument(
        "--batch-prompt-lengths",
        help=(
            "Comma-separated synthetic prompt lengths to submit in one batched "
            "generate call, e.g. 8,5,1."
        ),
    )
    parser.add_argument(
        "--verify-batch-against-sequential",
        action="store_true",
        help=(
            "After a batched workload, rerun each prompt separately through "
            "the same engine and require identical greedy token ids."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--first-token-probe",
        action="store_true",
        help="Generate and time one token before the requested workload.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force generation to reach --max-tokens for throughput timing.",
    )
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
    config_path = args.checkpoint / "config.json"
    if not config_path.is_file():
        parser.error(f"checkpoint config does not exist: {config_path}")
    checkpoint_config = json.loads(config_path.read_text())
    num_hidden_layers = int(checkpoint_config["num_hidden_layers"])
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be positive")
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    if args.ep_degree is not None:
        if not args.enable_expert_parallel:
            parser.error("--ep-degree requires --enable-expert-parallel")
        if args.ep_degree < 1:
            parser.error("--ep-degree must be positive")
    if args.capture_modules and args.capture_dir is None:
        parser.error("--capture-modules requires --capture-dir")
    if args.capture_modules and args.capture_attention_internals_layer is not None:
        parser.error(
            "--capture-modules and --capture-attention-internals-layer are "
            "mutually exclusive"
        )
    if args.capture_attention_internals_layer is not None and not (
        0 <= args.capture_attention_internals_layer < num_hidden_layers
    ):
        parser.error(
            "--capture-attention-internals-layer must be in "
            f"[0, {num_hidden_layers})"
        )
    prompt_modes = sum(
        value is not None
        for value in (
            args.prompt,
            args.prompt_length,
            args.workload_lengths,
            args.batch_prompt_lengths,
        )
    )
    if prompt_modes > 1:
        parser.error(
            "--prompt, --prompt-length, --workload-lengths, and "
            "--batch-prompt-lengths are mutually exclusive"
        )
    if args.verify_batch_against_sequential and args.batch_prompt_lengths is None:
        parser.error(
            "--verify-batch-against-sequential requires --batch-prompt-lengths"
        )
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.max_model_len < 1:
        parser.error("--max-model-len must be positive")
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
    num_seqs_buckets = parse_buckets(args.num_seqs_buckets, "--num-seqs-buckets")
    if prefill_buckets[-1] != max_num_batched_tokens:
        parser.error("the largest prefill segment bucket must equal --max-num-batched-tokens")
    if max_num_batched_tokens > args.max_model_len:
        parser.error("--max-num-batched-tokens cannot exceed --max-model-len")
    if decode_buckets and decode_buckets[-1] > args.max_model_len:
        parser.error("decode context buckets cannot exceed --max-model-len")
    if num_seqs_buckets and num_seqs_buckets[-1] != args.max_num_seqs:
        parser.error("the last --num-seqs-buckets value must equal --max-num-seqs")

    if args.async_scheduling == "on" and args.sampling_backend != "device":
        parser.error("--async-scheduling on requires --sampling-backend device")

    neuron_config = {
        "num_batched_tokens_buckets": prefill_buckets,
        "on_device_sampling_config": (
            # Mirrors benchmark_decode.py's on-device configuration exactly.
            # `all_greedy=False` is not a stylistic choice: it selects the
            # generic sampling graph, and a temperature=0 request still gets
            # argmax inside it. Diverging here would mean this harness
            # compiles a different graph from the one that fails.
            {"all_greedy": False, "max_top_k": 256, "deterministic": True}
            if args.sampling_backend == "device"
            else None
        ),
    }
    if num_seqs_buckets:
        neuron_config["num_seqs_buckets"] = num_seqs_buckets
    if args.ep_degree is not None:
        neuron_config["ep_degree"] = args.ep_degree
    if args.quantization is not None:
        neuron_config["quantization"] = args.quantization
    # The runner always compiles max_model_len as an implicit final decode
    # bucket. Accepting it in this benchmark-facing list keeps the requested
    # geometry explicit without passing a redundant value to core validation.
    explicit_decode_buckets = [
        bucket for bucket in (decode_buckets or []) if bucket < args.max_model_len
    ]
    if explicit_decode_buckets:
        neuron_config["decode_context_length_buckets"] = explicit_decode_buckets
    if args.capture_dir is not None:
        if args.capture_modules:
            capture_modules = [
                module.strip() for module in args.capture_modules.split(",")
            ]
            if any(not module for module in capture_modules):
                parser.error("--capture-modules cannot contain empty names")
            if len(capture_modules) != len(set(capture_modules)):
                parser.error("--capture-modules cannot contain duplicates")
        elif args.capture_attention_internals_layer is None:
            capture_modules = [
                "model.embed_tokens",
                *[
                    f"model.layers.{layer}.{module}"
                    for layer in range(num_hidden_layers)
                    for module in (
                        "attn_hc",
                        "input_layernorm",
                        "attention",
                        "ffn_hc",
                        "post_attention_layernorm",
                        "moe",
                    )
                ],
                *(f"model.layers.{layer}" for layer in range(num_hidden_layers)),
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

    initialization_started = time.perf_counter()
    llm = LLM(
        model=str(args.checkpoint),
        load_format=args.load_format,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=args.block_size,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        async_scheduling=args.async_scheduling == "on",
        additional_config={"neuron_config": neuron_config},
    )
    initialization_seconds = time.perf_counter() - initialization_started
    # Read the bound rather than hardcoding the synthetic model's 64-token vocab,
    # so a real checkpoint is validated against its own vocabulary.
    vocab_size = checkpoint_config["vocab_size"]
    batch_workload = args.batch_prompt_lengths is not None
    length_spec = args.batch_prompt_lengths or args.workload_lengths
    if length_spec:
        try:
            lengths = [int(item) for item in length_spec.split(",")]
        except ValueError:
            option = (
                "--batch-prompt-lengths"
                if batch_workload
                else "--workload-lengths"
            )
            parser.error(f"{option} must be comma-separated integers")
        if not lengths or any(length <= 0 for length in lengths):
            option = (
                "--batch-prompt-lengths"
                if batch_workload
                else "--workload-lengths"
            )
            parser.error(f"{option} entries must be positive")
    elif args.prompt_length is not None:
        lengths = [args.prompt_length]
    else:
        lengths = []
    if batch_workload and len(lengths) > args.max_num_seqs:
        parser.error(
            "--batch-prompt-lengths contains more prompts than --max-num-seqs"
        )
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
    first_token_result = None
    if args.first_token_probe:
        probe_started = time.perf_counter()
        probe_outputs = llm.generate(
            [{"prompt_token_ids": prompts[0]}],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        )
        probe_seconds = time.perf_counter() - probe_started
        probe_tokens = list(probe_outputs[0].outputs[0].token_ids)
        if len(probe_tokens) != 1:
            raise SystemExit(f"first-token probe returned {probe_tokens}")
        first_token_result = {
            "token_id": probe_tokens[0],
            "seconds": probe_seconds,
        }

    def validate_output(prompt_ids, output, generation_seconds):
        token_ids = list(output.outputs[0].token_ids)
        if len(token_ids) != args.max_tokens or any(
            token < 0 or token >= vocab_size for token in token_ids
        ):
            raise SystemExit(f"invalid generated tokens: {token_ids}")
        return {
            "prompt_length": len(prompt_ids),
            "token_ids": token_ids,
            "generation_seconds": generation_seconds,
            "tokens_per_second": len(token_ids) / generation_seconds,
        }

    workload_results = []
    batch_generation_seconds = None
    batch_matches_sequential = None
    if batch_workload:
        generation_started = time.perf_counter()
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_ids} for prompt_ids in prompts],
            SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                ignore_eos=args.ignore_eos,
            ),
        )
        batch_generation_seconds = time.perf_counter() - generation_started
        if len(outputs) != len(prompts):
            raise SystemExit(
                f"batched generation returned {len(outputs)} outputs for "
                f"{len(prompts)} prompts"
            )
        workload_results = [
            validate_output(prompt_ids, output, batch_generation_seconds)
            for prompt_ids, output in zip(prompts, outputs, strict=True)
        ]
        if args.verify_batch_against_sequential:
            sequential_tokens = []
            for prompt_ids in prompts:
                sequential = llm.generate(
                    [{"prompt_token_ids": prompt_ids}],
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                        ignore_eos=args.ignore_eos,
                    ),
                )[0]
                sequential_tokens.append(list(sequential.outputs[0].token_ids))
            batched_tokens = [item["token_ids"] for item in workload_results]
            batch_matches_sequential = sequential_tokens == batched_tokens
            if not batch_matches_sequential:
                raise SystemExit(
                    "batched greedy tokens differ from same-engine sequential tokens: "
                    f"batch={batched_tokens}, sequential={sequential_tokens}"
                )
    else:
        for prompt_ids in prompts:
            generation_started = time.perf_counter()
            outputs = llm.generate(
                [{"prompt_token_ids": prompt_ids}],
                SamplingParams(
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    ignore_eos=args.ignore_eos,
                ),
            )
            generation_seconds = time.perf_counter() - generation_started
            workload_results.append(
                validate_output(prompt_ids, outputs[0], generation_seconds)
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "token_ids": workload_results[-1]["token_ids"],
        # Recorded because two runs of this script are only comparable when the
        # sampling path matches; a device-sampled token list and a CPU-sampled
        # one look identical in the JSON otherwise.
        "sampling_backend": args.sampling_backend,
        "async_scheduling": args.async_scheduling == "on",
        "initialization_seconds": initialization_seconds,
        "generation_seconds": workload_results[-1]["generation_seconds"],
        "tokens_per_second": workload_results[-1]["tokens_per_second"],
    }
    if first_token_result is not None:
        result["first_token_probe"] = first_token_result
    if len(workload_results) > 1 or lengths:
        result["workloads"] = workload_results
    if batch_generation_seconds is not None:
        result["batch_generation_seconds"] = batch_generation_seconds
        result["batch_tokens_per_second"] = (
            len(prompts) * args.max_tokens / batch_generation_seconds
        )
    if batch_matches_sequential is not None:
        result["batch_matches_sequential"] = batch_matches_sequential
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
