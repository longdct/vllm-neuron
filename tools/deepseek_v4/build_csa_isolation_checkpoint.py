#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the one-layer, official-attention-geometry CSA isolation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import DeepseekV4Config

from vllm_neuron.model.deepseek_v4.model import DeepseekV4ForCausalLM


def build(output: Path, *, max_position_embeddings: int = 256) -> None:
    output.mkdir(parents=True, exist_ok=False)
    config = DeepseekV4Config(
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        q_lora_rank=128,
        sliding_window=128,
        max_position_embeddings=max_position_embeddings,
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["moe"],
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        o_groups=8,
        o_lora_rank=64,
        architectures=["DeepseekV4ForCausalLM"],
        dtype="bfloat16",
    )
    config_dict = config.to_dict()
    # The registered on-disk config has an older MLP vocabulary.  The plugin's
    # safe default is routed MoE, which is the value requested above.
    config_dict.pop("mlp_layer_types", None)
    (output / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")

    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM.from_configs(config).eval()
    save_file(
        {name: value.contiguous() for name, value in model.named_parameters()},
        output / "model.safetensors",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-position-embeddings", type=int, default=256)
    args = parser.parse_args()
    build(
        args.output,
        max_position_embeddings=args.max_position_embeddings,
    )
