#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the TRN2 ``shard_on_block`` MoE kernel actually dequantize FP8 weights?

The kernel signature takes ``gate_up_proj_scale``/``down_proj_scale`` and the
docstring calls them "optional, for FP8", but **nothing in vllm_neuron has ever
passed them** -- DeepSeek-V4's own call site (``model.py`` ``_forward_nki``)
runs BF16 weights with no scales.  Every later step of the FP8 plan assumes
this path works, so prove it here, on eight experts, before touching the model.

The probe is built so a *correct* kernel matches the BF16 reference to within
accumulation noise:

  * expert weights are drawn, quantized to FP8, and dequantized back, so the
    reference is exactly representable in both FP8 and BF16 (e4m3 has 3
    mantissa bits, bfloat16 has 8) -- no quantization error is introduced;
  * the per-output-channel scale is a power of two, so ``elem * scale``
    reproduces the reference value bit-exactly rather than approximately.

A large error therefore means the kernel ignored the scales, applied them on
the wrong axis, or wants a different layout -- not that FP8 is lossy.

Both scale orderings for the fused gate/up channel axis are selectable because
the kernel documents only "[E, 1, 2*I_TP] ... reshaped to [E, 2, I_TP]", which
does not pin whether the flat index is ``shard * I + i`` or ``i * 2 + shard``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import nki.language as nl
import torch
from nkilib.core.moe.moe_cte.moe_cte import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoECTEImplementation,
)
from nkilib.core.utils.common_types import QuantizationType

_MOE_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_neuron"
    / "functional"
    / "moe"
    / "moe_cte.py"
)
_MOE_SPEC = importlib.util.spec_from_file_location("dsv4_standalone_moe_cte", _MOE_PATH)
assert _MOE_SPEC is not None and _MOE_SPEC.loader is not None
_MOE_MODULE = importlib.util.module_from_spec(_MOE_SPEC)
sys.modules[_MOE_SPEC.name] = _MOE_MODULE
_MOE_SPEC.loader.exec_module(_MOE_MODULE)
moe_cte = _MOE_MODULE.moe_cte

#: TRN2's FP8 is e4m3 *with* inf (max finite 240), not e4m3fn; torch has no
#: separate dtype for it and ``nki_dtype.torch_to_nki_dtype`` maps
#: ``float8_e4m3fn -> float8_e4m3`` on this target.  Quantize against 240 so a
#: value never lands on the unrepresentable side of that difference.
FP8_MAX = 240.0
FP8_DTYPE = torch.float8_e4m3fn


def quantize_per_channel(
    weight: torch.Tensor, channel_dims: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``weight`` into FP8 elements and a power-of-two per-channel scale.

    ``channel_dims`` names the axes that survive into the scale (the output
    channels); every other axis is reduced over.  Returns the FP8 elements, the
    float32 scale, and the exact product of the two -- the last is what the
    BF16 reference must use so both paths see identical weights.
    """
    reduce_dims = tuple(d for d in range(weight.ndim) if d not in channel_dims)
    peak = weight.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-30)
    # A power-of-two scale keeps elem*scale exact: it shifts the exponent and
    # leaves the mantissa alone. A non-power-of-two would reintroduce rounding
    # and blur the kernel's own error into the measurement.
    #
    # ceil, not floor: the scale must be at least peak/FP8_MAX so that
    # weight/scale stays inside the format. Rounding the exponent down instead
    # lets the largest channel element land above 240, which saturates to inf
    # in e4m3 and poisons the whole tensor with NaN.
    exponent = torch.ceil(torch.log2(peak / FP8_MAX))
    scale = torch.exp2(exponent)
    elements = (weight / scale).to(FP8_DTYPE)
    exact = elements.float() * scale
    return elements, scale, exact


def build_case(
    experts: int, hidden: int, intermediate: int, tokens: int, block_size: int, seed: int
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    gate_up = torch.randn(
        experts, hidden, 2, intermediate, generator=generator
    ) * 0.02
    down = torch.randn(experts, intermediate, hidden, generator=generator) * 0.02

    # gate/up output channels are (shard, intermediate); down's is hidden.
    gu_elem, gu_scale, gu_exact = quantize_per_channel(gate_up, (0, 2, 3))
    dn_elem, dn_scale, dn_exact = quantize_per_channel(down, (0, 2))

    hidden_states = (
        torch.randn(tokens, hidden, generator=generator) * 0.5
    ).to(torch.bfloat16)
    affinities = torch.zeros(tokens * experts, 1, dtype=torch.bfloat16)
    # One expert per token, round-robin, affinity 1.0 so the output is the raw
    # expert result and any scale error shows up undamped.
    for t in range(tokens):
        affinities[t * experts + (t % experts), 0] = 1.0

    blocks = (tokens + block_size - 1) // block_size + experts
    token_ids = torch.full((blocks * block_size,), -1, dtype=torch.int32)
    block_experts = torch.zeros(blocks, dtype=torch.int32)
    for t in range(tokens):
        expert = t % experts
        block_experts[expert] = expert
        slot = expert * block_size + (t // experts)
        if t // experts < block_size:
            token_ids[slot] = t

    return {
        "hidden_states": hidden_states,
        "affinities": affinities,
        "token_ids": token_ids,
        "block_experts": block_experts,
        "blocks": blocks,
        "gate_up_exact": gu_exact.to(torch.bfloat16),
        "down_exact": dn_exact.to(torch.bfloat16),
        "gate_up_fp8": gu_elem,
        "down_fp8": dn_elem,
        "gate_up_scale": gu_scale,
        "down_scale": dn_scale,
    }


class RuntimeMoE(torch.nn.Module):
    def __init__(self, block_size: int, implementation=None):
        super().__init__()
        self.block_size = block_size
        self.implementation = (
            implementation or MoECTEImplementation.shard_on_block
        )

    def forward(
        self, hidden, affinities, gate_up, down, token_ids, experts,
        gate_up_scale=None, down_scale=None,
        quantization_type=QuantizationType.NONE,
    ):
        return moe_cte(
            hidden_states=hidden,
            expert_affinities_masked=affinities,
            gate_up_proj_weight=gate_up,
            down_proj_weight=down,
            token_position_to_id=token_ids,
            block_to_expert=experts,
            block_size=self.block_size,
            implementation=self.implementation,
            activation_function=ActFnType.SiLU,
            compute_dtype=nl.bfloat16,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            gate_up_proj_scale=gate_up_scale,
            down_proj_scale=down_scale,
            quantization_type=quantization_type,
            skip_token=True,
            is_tensor_update_accumulating=True,
        )


def flatten_gate_up_scale(scale: torch.Tensor, order: str) -> torch.Tensor:
    """``[E, 1, 2, I]`` -> the kernel's flat ``[E, 1, 2*I]``."""
    experts, _, shards, intermediate = scale.shape
    if order == "shard-major":  # [2, I] row-major: index = shard * I + i
        flat = scale.reshape(experts, 1, shards * intermediate)
    else:  # interleaved: index = i * 2 + shard
        flat = scale.permute(0, 1, 3, 2).reshape(experts, 1, shards * intermediate)
    return flat.contiguous()


def compare(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    ref = reference.float()
    act = actual.float()
    denominator = ref.abs().max().clamp(min=1e-30)
    absolute = (act - ref).abs()
    # float64: over ~5e5 elements a float32 cosine drifts ~4e-5 even for
    # bit-identical tensors, which reads as a real discrepancy next to an
    # exact max_abs_error of 0.
    cosine = torch.nn.functional.cosine_similarity(
        ref.flatten().double(), act.flatten().double(), dim=0
    )
    return {
        "max_abs_error": float(absolute.max()),
        "max_rel_error": float(absolute.max() / denominator),
        "mean_abs_error": float(absolute.mean()),
        "cosine_similarity": float(cosine),
        "reference_abs_max": float(denominator),
        "bitwise_identical": bool(torch.equal(act, ref)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument(
        "--scale-order",
        choices=("shard-major", "interleaved", "both"),
        default="both",
        help="flat layout of the fused gate/up channel scale; 'both' reports each",
    )
    parser.add_argument(
        "--quantization",
        choices=("sweep", "ROW", "STATIC", "NONE"),
        default="sweep",
        help="QuantizationType for the shard_on_block FP8 path",
    )
    parser.add_argument(
        "--implementation",
        choices=("shard_on_block", "shard_on_i"),
        default="shard_on_block",
        help="TRN2 MoE kernel variant; shard_on_i requires --block-size 256+",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    case = build_case(
        args.experts, args.hidden, args.intermediate,
        args.tokens, args.block_size, args.seed,
    )
    device = torch.device("neuron:0")
    compiled = torch.compile(
        RuntimeMoE(
            args.block_size, getattr(MoECTEImplementation, args.implementation)
        ),
        backend="neuron",
        dynamic=False,
    )

    def run(gate_up, down, gu_scale=None, dn_scale=None,
            quant=QuantizationType.NONE):
        started = time.monotonic()
        out = compiled(
            case["hidden_states"].to(device),
            case["affinities"].to(device),
            gate_up.to(device),
            down.to(device),
            case["token_ids"].to(device),
            case["block_experts"].to(device),
            None if gu_scale is None else gu_scale.to(device),
            None if dn_scale is None else dn_scale.to(device),
            quant,
        )
        torch.neuron.synchronize()
        return out.to("cpu"), time.monotonic() - started

    record: dict = {
        "configuration": {
            "experts": args.experts,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "tokens": args.tokens,
            "block_size": args.block_size,
            "implementation": args.implementation,
            "blocks": case["blocks"],
            "seed": args.seed,
            "fp8_dtype": str(FP8_DTYPE),
            "fp8_max": FP8_MAX,
        },
    }

    reference, ref_seconds = run(case["gate_up_exact"], case["down_exact"])
    record["reference"] = {
        "wall_seconds": ref_seconds,
        "output_shape": list(reference.shape),
        "abs_max": float(reference.float().abs().max()),
        "finite": bool(torch.isfinite(reference.float()).all()),
    }

    orders = (
        ("shard-major", "interleaved")
        if args.scale_order == "both"
        else (args.scale_order,)
    )
    quant_names = (
        ("ROW", "STATIC") if args.quantization == "sweep" else (args.quantization,)
    )
    record["fp8"] = {}
    for quant_name in quant_names:
      for order in orders:
        key = f"{quant_name}/{order}"
        try:
            actual, seconds = run(
                case["gate_up_fp8"],
                case["down_fp8"],
                flatten_gate_up_scale(case["gate_up_scale"], order),
                case["down_scale"].reshape(args.experts, 1, args.hidden).contiguous(),
                getattr(QuantizationType, quant_name),
            )
        except Exception as error:  # noqa: BLE001 - the failure IS the result
            record["fp8"][key] = {
                "ran": False,
                "error_type": type(error).__name__,
                "error": str(error)[:600],
            }
            continue
        entry = {"ran": True, "wall_seconds": seconds}
        entry.update(compare(reference, actual))
        entry["finite"] = bool(torch.isfinite(actual.float()).all())
        record["fp8"][key] = entry

    ran = [v for v in record["fp8"].values() if v.get("ran")]
    record["verdict"] = (
        "FP8 path rejected the inputs"
        if not ran
        else "PASS: matches BF16 reference"
        if any(v["max_rel_error"] < 1e-2 for v in ran)
        else "RAN but does NOT match the reference -- scales ignored or mislaid"
    )

    text = json.dumps(record, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
