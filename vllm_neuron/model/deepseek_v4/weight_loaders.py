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
    (".attn.wq_a", ".attn.fused_wqa_wkv", 0),
    (".attn.wkv", ".attn.fused_wqa_wkv", 1),
    (".compressor.wkv", ".compressor.fused_wkv_wgate", 0),
    (".compressor.wgate", ".compressor.fused_wkv_wgate", 1),
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
