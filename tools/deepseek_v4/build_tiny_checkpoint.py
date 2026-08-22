#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the deterministic three-layer DeepSeek-V4 correctness checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import DeepseekV4Config, PreTrainedTokenizerFast

from vllm_neuron.model.deepseek_v4.model import DeepseekV4ForCausalLM


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    config = DeepseekV4Config(
        hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_local_experts=4, num_experts_per_tok=2, vocab_size=64,
        num_hidden_layers=3, num_attention_heads=1, num_key_value_heads=1,
        head_dim=16, q_lora_rank=16, sliding_window=16,
        max_position_embeddings=64,
        layer_types=["heavily_compressed_attention", "sliding_attention",
                     "compressed_sparse_attention"],
        architectures=["DeepseekV4ForCausalLM"],
    )
    config_dict = config.to_dict()
    config_dict.pop("mlp_layer_types", None)
    (output / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")

    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM.from_configs(config).eval()
    save_file(
        {name: value.contiguous() for name, value in model.named_parameters()},
        output / "model.safetensors",
    )

    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocabulary.update({f"t{i}": i for i in range(4, 64)})
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.decoder = decoders.WordPiece(prefix="")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
        unk_token="<unk>", pad_token="<pad>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} "
        "{{ message['content'] }} {% endfor %}{% if add_generation_prompt %}"
        "assistant {% endif %}"
    )
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
