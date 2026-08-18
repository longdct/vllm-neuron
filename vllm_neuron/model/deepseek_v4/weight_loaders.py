# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-name contracts for the DeepSeek-V4 production loader."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ExpertDType = Literal["bf16", "fp8", "fp4"]


@dataclass(frozen=True)
class StackedShard:
    """One checkpoint tensor packed into a fused runtime parameter."""

    parameter_name: str
    shard_id: int


_STACKED_COMPONENTS = (
    (".w1", ".gate_up_proj", 0),
    (".w3", ".gate_up_proj", 1),
    (".compressor.wkv", ".compressor.fused_wkv_wgate", 0),
    (".compressor.wgate", ".compressor.fused_wkv_wgate", 1),
)

# Plain (not stacked/fused) attention weight renames: the official checkpoint's
# raw tensor names -> this plugin's DeepseekV4Attention parameter names.
# ``wq_a``/``wkv`` were previously mapped as two shards merged into one fused
# ``fused_wqa_wkv`` parameter, matching vLLM's own real GPU DeepSeek-V4
# backend (``vllm.models.deepseek_v4.attention.DeepseekV4Attention``, which
# fuses them into a single ``MergedColumnParallelLinear`` for kernel
# efficiency). This plugin's ``DeepseekV4Attention`` keeps them separate
# instead (matching ``transformers.models.deepseek_v4``'s reference module,
# what the device-shaped rewrite was cross-validated against -- see
# ``model.py``'s ``DeepseekV4Attention`` docstring), so each checkpoint
# tensor now renames onto its own standalone parameter rather than a shard
# of a fused one. ``kv_norm`` needs no rule -- the checkpoint's name for it
# already matches this plugin's parameter name and passes through unchanged.
_ATTENTION_RENAMES = (
    (".attn.wq_a", ".attn.q_a_proj"),
    (".attn.wkv", ".attn.kv_proj"),
    (".attn.wq_b", ".attn.q_b_proj"),
    (".attn.q_norm", ".attn.q_a_norm"),
    (".attn.wo_a", ".attn.o_a_proj"),
    (".attn.wo_b", ".attn.o_b_proj"),
    (".attn.attn_sink", ".attn.sinks"),
)


def map_checkpoint_name(name: str, expert_dtype: ExpertDType = "bf16") -> str:
    """Map an official V4 checkpoint key into the vLLM model namespace.

    The contract mirrors vLLM 0.26's DeepSeek-V4 ``WeightsMapper``. Keeping the
    pure string transformation here makes checkpoint compatibility testable
    without importing CUDA/ROCm model implementations.
    """
    if not name or name.startswith(".") or name.endswith("."):
        raise ValueError(f"invalid DeepSeek-V4 checkpoint name: {name!r}")
    if expert_dtype not in ("bf16", "fp8", "fp4"):
        raise ValueError(f"unsupported DeepSeek-V4 expert dtype: {expert_dtype!r}")

    if name.startswith("layers."):
        mapped = "model." + name
    elif name.startswith("embed."):
        mapped = "model." + name
    elif name.startswith("norm."):
        mapped = "model." + name
    elif name.startswith("hc_head"):
        mapped = "model." + name
    elif name.startswith("mtp."):
        mapped = "model." + name
    else:
        mapped = name

    if mapped.endswith("head.weight") and mapped == "head.weight":
        mapped = "lm_head.weight"
    elif mapped.endswith("embed.weight"):
        mapped = mapped[: -len("embed.weight")] + "embed_tokens.weight"
    elif mapped.endswith(".ffn.gate.bias"):
        mapped = mapped[: -len(".ffn.gate.bias")] + ".ffn.gate.e_score_correction_bias"

    mapped = mapped.replace(".shared_experts.w2", ".shared_experts.down_proj")
    for source, target in _ATTENTION_RENAMES:
        mapped = mapped.replace(source, target)
    if mapped.endswith(".scale"):
        if expert_dtype == "fp4" and re.search(r"\.experts\.\d+\.w[123]\.scale$", mapped):
            mapped = mapped[: -len(".scale")] + ".weight_scale"
        else:
            mapped = mapped[: -len(".scale")] + ".weight_scale_inv"
    return mapped


def resolve_stacked_shard(mapped_name: str) -> StackedShard | None:
    """Resolve fused attention/compressor/MLP shards after namespace mapping."""
    if ".experts." in mapped_name:
        return None
    for source, target, shard_id in _STACKED_COMPONENTS:
        if source in mapped_name:
            return StackedShard(mapped_name.replace(source, target), shard_id)
    return None


def require_weight_shape(
    name: str,
    loaded_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    """Reject shape drift before a loader copies or shards a tensor."""
    if loaded_shape != expected_shape:
        raise ValueError(
            f"DeepSeek-V4 weight {name!r} has shape {loaded_shape}, expected {expected_shape}"
        )


def load_checkpoint_weights(
    module,
    weights,
    *,
    expert_dtype: ExpertDType = "bf16",
) -> set[str]:
    """Load a checkpoint iterator through the mapped/fused parameter contract."""
    torch = __import__("torch")
    params = dict(module.named_parameters())
    loaded_params: set[str] = set()
    loaded_destinations: set[tuple[str, int | None]] = set()
    source_names: set[str] = set()

    for source_name, loaded_weight in weights:
        if source_name in source_names:
            raise ValueError(f"duplicate DeepSeek-V4 checkpoint weight {source_name!r}")
        source_names.add(source_name)
        mapped_name = map_checkpoint_name(source_name, expert_dtype)
        stacked = resolve_stacked_shard(mapped_name)
        target_name = stacked.parameter_name if stacked else mapped_name
        shard_id = stacked.shard_id if stacked else None
        destination = (target_name, shard_id)
        if destination in loaded_destinations:
            raise ValueError(
                f"multiple DeepSeek-V4 weights map to destination {destination!r}"
            )
        if target_name not in params:
            raise ValueError(
                f"DeepSeek-V4 checkpoint weight {source_name!r} maps to missing "
                f"parameter {target_name!r}"
            )

        parameter = params[target_name]
        weight_loader = getattr(parameter, "weight_loader", None)
        with torch.no_grad():
            if stacked is not None:
                if weight_loader is None:
                    raise ValueError(
                        f"fused DeepSeek-V4 parameter {target_name!r} has no shard loader"
                    )
                weight_loader(parameter, loaded_weight, shard_id)
            elif weight_loader is not None:
                weight_loader(parameter, loaded_weight)
            else:
                require_weight_shape(
                    source_name,
                    tuple(loaded_weight.shape),
                    tuple(parameter.shape),
                )
                parameter.copy_(loaded_weight)
        loaded_destinations.add(destination)
        loaded_params.add(target_name)
    return loaded_params
