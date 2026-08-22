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
    args = parser.parse_args()

    neuron_config = {
        "num_batched_tokens_buckets": [8, 64],
        "on_device_sampling_config": None,
    }
    if args.capture_dir is not None:
        neuron_config["tensor_capture"] = {
            "modules": [
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
            ],
            "capture_dir": str(args.capture_dir),
        }

    llm = LLM(
        model=str(args.checkpoint),
        load_format="dummy",
        max_num_seqs=1,
        max_model_len=64,
        block_size=32,
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        num_gpu_blocks_override=256,
        async_scheduling=False,
        additional_config={
            "neuron_config": neuron_config
        },
    )
    outputs = llm.generate(
        [{"prompt_token_ids": list(range(1, 9))}],
        SamplingParams(temperature=0.0, max_tokens=4),
    )
    token_ids = list(outputs[0].outputs[0].token_ids)
    if len(token_ids) != 4 or any(token < 0 or token >= 64 for token in token_ids):
        raise SystemExit(f"invalid generated tokens: {token_ids}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"token_ids": token_ids}, indent=2) + "\n")


if __name__ == "__main__":
    main()
