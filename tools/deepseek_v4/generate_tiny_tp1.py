#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the deterministic tiny checkpoint through the real vLLM engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
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
        "--num-gpu-blocks-override",
        type=int,
        default=256,
        help=(
            "KV cache blocks (default: 256). Lowering this shrinks compile time "
            "as well as memory: scatter_paged_latent builds a torch.where over "
            "the whole cache once per token, and capacity is num_blocks * "
            "storage_block_size. The floor is the largest per-group value in "
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
    if args.max_model_len < 12:
        parser.error("--max-model-len must fit the 8-token prompt and 4 outputs")
    if args.num_gpu_blocks_override < 32:
        parser.error("--num-gpu-blocks-override must be at least 32")

    neuron_config = {
        "num_batched_tokens_buckets": sorted({8, args.max_model_len}),
        "on_device_sampling_config": None,
    }
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
        block_size=32,
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        kv_cache_dtype=args.kv_cache_dtype,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        async_scheduling=False,
        additional_config={
            "neuron_config": neuron_config
        },
    )
    prompt_ids = (
        [int(x) for x in args.prompt.split(",")] if args.prompt else list(range(1, 9))
    )
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(temperature=0.0, max_tokens=4),
    )
    token_ids = list(outputs[0].outputs[0].token_ids)
    # Read the bound rather than hardcoding the synthetic model's 64-token vocab,
    # so a real checkpoint is validated against its own vocabulary.
    vocab_size = json.loads(
        (args.checkpoint / "config.json").read_text()
    )["vocab_size"]
    if len(token_ids) != 4 or any(token < 0 or token >= vocab_size for token in token_ids):
        raise SystemExit(f"invalid generated tokens: {token_ids}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"token_ids": token_ids}, indent=2) + "\n")


if __name__ == "__main__":
    main()
