#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare this plugin's pre-sampling logits against the transformers reference.

The plugin and ``transformers.models.deepseek_v4`` are independent
implementations of the same architecture. Driving both from the *same* real
weights and prompt turns "is the plugin correct?" into a measurable question.

Run the plugin side first to capture its logits::

    VLLM_NEURON_ENABLE_DEEPSEEK_V4=1 VLLM_NEURON_CPU_MODE=1 \\
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
    VLLM_NEURON_TINY_VALIDATION_DIR=<capture-dir> \\
    python tools/deepseek_v4/generate_tiny_tp1.py <slice> --output <json> \\
      --enforce-eager --load-format auto --prompt <ids> --max-model-len 16

then point this at the same slice, prompt and capture directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hf_reference import load_reference_model


def _install_kv_cache_rounding() -> None:
    """Round the reference's post-RoPE K/V to BF16, as the paged cache does.

    The plugin writes K/V into a BF16 paged cache *after* RoPE, so that is the
    point to quantize -- rounding earlier (at ``kv_norm``) is a different
    computation and does not reproduce it.

    Keying on the head axis is what distinguishes the two rotary call sites:
    queries carry ``num_attention_heads`` there, K/V carry one.
    """
    import transformers.models.deepseek_v4.modeling_deepseek_v4 as reference

    original = reference.apply_rotary_pos_emb

    def rounded(x, cos, sin, unsqueeze_dim=1):
        out = original(x, cos, sin, unsqueeze_dim)
        if out.shape[1] == 1:  # single head -> this is the K/V projection
            return out.to(torch.bfloat16).to(out.dtype)
        return out

    reference.apply_rotary_pos_emb = rounded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slice_dir", type=Path)
    parser.add_argument("capture_dir", type=Path, help="plugin logits-*.pt directory")
    parser.add_argument("--prompt", required=True, help="comma-separated token ids")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.125,
        help=(
            "Per-logit absolute tolerance (default: 0.125). The plugin's lm_head "
            "emits BF16, whose ULP at the observed logit scale (~23) is 0.125, so "
            "no correct implementation can land closer than half that from the "
            "FP32 reference. A tighter bound measures the output dtype, not the "
            "implementation."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16"),
        help=(
            "Reference compute dtype. Must match the plugin's, or the comparison "
            "measures the dtype gap instead of an implementation difference: the "
            "CPU path (VLLM_NEURON_CPU_MODE=1) runs in float32, while a device "
            "run is bfloat16. Mismatching these inflates max|diff| about 3x."
        ),
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help=(
            "Round the reference's post-RoPE K/V to this dtype before attention, "
            "modelling the plugin's paged cache. The plugin stores BF16 K/V and "
            "cannot store FP32 (vLLM's CacheConfig has no float32 option and the "
            "Neuron KV-cache specs pin bfloat16), so leaving this at float32 "
            "measures that quantization rather than an implementation difference."
        ),
    )
    parser.add_argument(
        "--logits-dtype",
        default="float32",
        choices=("bfloat16", "float32"),
        help=(
            "Reference logits dtype. Keep this at float32: the plugin's lm_head "
            "emits BF16, so |bf16_plugin - fp32_reference| measures how far the "
            "plugin is from the true value, and half a BF16 ULP (0.0625 at logit "
            "scale 23) is the floor no correct implementation can beat. Rounding "
            "the reference too makes the metric *worse*, since two independently "
            "rounded values can land a full ULP apart."
        ),
    )
    args = parser.parse_args()

    if args.kv_cache_dtype == "bfloat16":
        _install_kv_cache_rounding()

    prompt = [int(x) for x in args.prompt.split(",")]
    captured = sorted(args.capture_dir.glob("logits-*.pt"))
    if not captured:
        raise SystemExit(f"no plugin logits captured in {args.capture_dir}")

    model, _ = load_reference_model(args.slice_dir, dtype=getattr(torch, args.dtype))

    # Greedy-decode alongside the plugin: at each step the reference sees the
    # prompt plus the tokens *it* produced, matching what the plugin's own greedy
    # loop fed itself. Feeding the plugin's tokens instead would hide a
    # divergence by re-syncing the two after every step.
    ids = list(prompt)
    rows = []
    for step, path in enumerate(captured):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]))
        reference = out.logits[0, -1]
        if args.logits_dtype == "bfloat16":
            reference = reference.to(torch.bfloat16)
        reference = reference.float()
        plugin = torch.load(path, weights_only=True).float().reshape(-1)
        delta = (reference - plugin).abs()
        ref_top, plug_top = int(reference.argmax()), int(plugin.argmax())
        top2 = reference.topk(2).values
        rows.append(
            (step, delta.max().item(), int((delta > args.tolerance).sum()),
             ref_top, plug_top, (top2[0] - top2[1]).item(),
             reference.abs().max().item())
        )
        ids.append(ref_top)

    print(f"{'step':<6}{'max|diff|':<14}{'>tol':<9}{'ref':<9}{'plugin':<9}"
          f"{'top1-top2':<12}{'|logit|max':<12}match")
    ok = True
    for step, worst, over, ref_top, plug_top, gap, scale in rows:
        agree = ref_top == plug_top
        ok = ok and agree and worst <= args.tolerance
        print(f"{step:<6}{worst:<14.6f}{over:<9}{ref_top:<9}{plug_top:<9}"
              f"{gap:<12.4f}{scale:<12.4f}{'yes' if agree else 'NO'}")
    print()
    print("MATCH" if ok else "DIVERGENCE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
