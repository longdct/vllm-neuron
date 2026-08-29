#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a tiny Qwen3.5-family text checkpoint for device bring-up.

The tensors are generated from HuggingFace's own ``Qwen3_5TextModel`` rather
than hand-rolled, so every checkpoint key and shape is correct by construction.
Hand-written shapes are exactly the kind of thing that produces a checkpoint the
loader accepts and silently mis-slices.

The default geometry is chosen so that **TP=8 lands in the same sharding regime
the 27B reaches at TP=16**: 8 key heads and 24 value heads give 1 key head and 3
value heads per rank, which is what ``resolve_sharding`` calls pure head
sharding. ``--value-split`` instead drops to 2 key heads so TP=8 is forced
through the value-dimension split -- the TP=32 regime of the real model, and the
one with the cross-rank gated-norm all-reduce.

**The geometry is pinned by two constraints that nearly exclude each other.**

*Hybrid page alignment.* A hybrid model declares two cache groups with
different page sizes, and vLLM unifies them only when the larger (Mamba) page
is an exact multiple of the smaller (attention) page -- it then grows the
attention block size by the ratio (``kv_cache_utils.py:1081``). The only
alternative is an attention backend that indexes KV by block stride so the page
can be padded, which this backend does not declare; otherwise it raises
``NotImplementedError``.

*Kernel head widths.* ``nki_gdn.py`` asserts ``k_dim`` and ``v_dim`` are each
one of 16/32/64/128 -- a head must fit in one SBUF partition. So the alignment
above cannot be bought by choosing arbitrary GDN dimensions.

Together these leave very little room. ``key_head_dim=64`` / ``value_head_dim=128``
gives a 99 KiB Mamba page that divides the 1 KiB attention page exactly at
``head_dim=32`` and ``--block-size 8``. That small attention head is the price
of the alignment rule; the attention path is not what this fixture exercises.

The sharding regime is untouched -- still 8 key heads and 24 value heads, so
TP=8 gives 1 key head and 3 value heads per rank, exactly as the 27B does at
TP=16, and the partition logic is width-agnostic.

**The real 27B satisfies neither cheaply.** Its Mamba page leaves a remainder of
7680 bytes against every power-of-two attention page, at every block size, so
serving the real checkpoint needs a fix in the cache specs rather than a choice
of dimensions. Its ``head_dim`` of 256 also exceeds the segmented attention
kernel's 128-element partition bound, forcing single-shot prefill -- a separate
known blocker, and why this fixture does not use 256 either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

#: Checkpoint prefix for the text decoder. The vision tower is out of scope and
#: is simply never emitted.
TEXT_PREFIX = "model.language_model"


def build_config(args: argparse.Namespace) -> Qwen3_5TextConfig:
    key_heads = 2 if args.value_split else 8
    value_heads = key_heads * 3
    return Qwen3_5TextConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=args.head_dim,
        max_position_embeddings=args.max_position_embeddings,
        linear_num_key_heads=key_heads,
        linear_num_value_heads=value_heads,
        linear_key_head_dim=args.key_head_dim,
        linear_value_head_dim=args.value_head_dim,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "mrope_interleaved": True,
            "partial_rotary_factor": 0.25,
        },
    )


def randomize(model: torch.nn.Module, seed: int) -> None:
    """Give the norms non-degenerate values under the right convention.

    HF initializes ``Qwen3_5RMSNorm.weight`` to **zeros** because it computes
    ``x * (1 + w)``, while the gated norm inside the GDN uses the ordinary
    ``w * x`` and initializes to ones. Perturbing each around its own identity
    keeps the checkpoint faithful to both conventions -- and a checkpoint whose
    norms are all exactly identity cannot catch a loader that confuses them.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.endswith("norm.weight"):
                continue
            identity = 1.0 if name.endswith("linear_attn.norm.weight") else 0.0
            noise = torch.randn(param.shape, generator=generator) * 0.02
            param.copy_(identity + noise)


def bare_text_config(text_config: Qwen3_5TextConfig) -> dict:
    """A text-only config, which ``Qwen3_5TextConfig.from_configs`` accepts.

    This is the default because the wrapper shape does not currently survive
    the Neuron serving path -- see :func:`wrapper_config`.
    """
    config = text_config.to_dict()
    config["architectures"] = ["Qwen3_5ForCausalLM"]
    config["model_type"] = "qwen3_5_text"
    config["dtype"] = "bfloat16"
    config["torch_dtype"] = "bfloat16"
    return config


def wrapper_config(text_config: Qwen3_5TextConfig) -> dict:
    """Wrap the text config the way a released checkpoint is shaped.

    Released Qwen3.5 checkpoints declare ``Qwen3_5ForConditionalGeneration``
    with the decoder nested under ``text_config``. This shape is **not** the
    default here, because it does not currently work end to end:

    ``platform.py::_resolve_vision_auto_config`` injects a
    ``vision_neuron_config`` for any config carrying ``hf_config.vision_config``,
    which makes ``neuron_model_runner.load_model`` take its multimodal branch
    and call ``from_configs(hf_config=..., text_neuron_config=...,
    vision_neuron_config=...)``. ``Qwen3_5ForCausalLM.from_configs`` accepts
    only ``(hf_config, neuron_config)``, so every rank dies with
    ``unexpected keyword argument 'text_neuron_config'``.

    That is a real gap for released checkpoints, which always carry a vision
    tower. Kept behind a flag so the failure stays reproducible.

    The vision tower is config-only -- no ``model.visual.*`` tensors are
    emitted, and the loader never asks for them.
    """
    vision = {
        "model_type": "qwen3_5_vision",
        "depth": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_heads": 2,
        "in_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": text_config.hidden_size,
        "num_position_embeddings": 64,
    }
    text = text_config.to_dict()
    text["model_type"] = "qwen3_5_text"
    vocab = text_config.vocab_size
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        # The stock ids are ~248k and would sit outside a tiny vocabulary.
        "image_token_id": vocab - 4,
        "video_token_id": vocab - 3,
        "vision_start_token_id": vocab - 2,
        "vision_end_token_id": vocab - 1,
        "text_config": text,
        "vision_config": vision,
    }


def build(args: argparse.Namespace) -> None:
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=False)

    config = build_config(args)
    torch.manual_seed(args.seed)
    model = Qwen3_5TextModel(config).eval()
    randomize(model, args.seed)

    tensors: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        value = param.detach()
        # A_log and dt_bias feed an exp/softplus that the layer evaluates in
        # fp32; everything else is stored at the model dtype.
        keep_fp32 = name.endswith("A_log") or name.endswith("dt_bias")
        value = value.float() if keep_fp32 else value.to(torch.bfloat16)
        tensors[f"{TEXT_PREFIX}.{name}"] = value.contiguous().clone()

    generator = torch.Generator().manual_seed(args.seed + 1)
    tensors["lm_head.weight"] = (
        (torch.randn(config.vocab_size, config.hidden_size, generator=generator) * 0.02)
        .to(torch.bfloat16)
        .contiguous()
    )

    save_file(tensors, str(output / "model.safetensors"))

    shape = wrapper_config if args.wrapper_config else bare_text_config
    (output / "config.json").write_text(json.dumps(shape(config), indent=2) + "\n")

    linear = sum(1 for t in config.layer_types if t == "linear_attention")
    full = sum(1 for t in config.layer_types if t == "full_attention")
    print(f"wrote {output}")
    print(f"  layers: {linear} gated-deltanet + {full} full-attention")
    print(f"  gdn: {config.linear_num_key_heads} K heads, "
          f"{config.linear_num_value_heads} V heads, "
          f"key/value head dim {config.linear_key_head_dim}/"
          f"{config.linear_value_head_dim}")
    print(f"  tensors: {len(tensors)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=32)
    # These defaults are not arbitrary: vLLM unifies hybrid page sizes only
    # when the Mamba page is an exact multiple of the attention page
    # (kv_cache_utils.py:1081), and 168/144 is the largest near-square pair
    # that satisfies it at head_dim=128, block_size=32. See the module
    # docstring.
    parser.add_argument("--key-head-dim", type=int, default=64)
    parser.add_argument("--value-head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--max-position-embeddings", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wrapper-config",
        action="store_true",
        help="Emit the released multimodal wrapper shape instead of a bare "
             "text config. Currently fails in the worker; see wrapper_config.",
    )
    parser.add_argument(
        "--value-split",
        action="store_true",
        help="Use 2 key heads so TP=8 exercises value-dimension splitting.",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
