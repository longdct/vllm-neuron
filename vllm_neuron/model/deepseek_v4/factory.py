# SPDX-License-Identifier: Apache-2.0
"""Per-layer component selection for the single DeepSeek-V4 model implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AttentionKind, MLPKind, NormalizedDeepseekV4Config


@dataclass(frozen=True)
class ComponentRegistry:
    sliding_attention: type[Any]
    c4_attention: type[Any]
    c128_attention: type[Any]
    hash_moe: type[Any]
    routed_moe: type[Any]


@dataclass(frozen=True)
class LayerComponents:
    attention: type[Any]
    mlp: type[Any]


def resolve_layer_components(
    config: NormalizedDeepseekV4Config,
    registry: ComponentRegistry,
) -> tuple[LayerComponents, ...]:
    """Resolve attention and MLP independently for every decoder layer."""
    resolved = []
    for layer in config.layers:
        if layer.attention is AttentionKind.SLIDING_WINDOW:
            attention = registry.sliding_attention
        elif layer.compress_ratio == 4:
            attention = registry.c4_attention
        elif layer.compress_ratio == 128:
            attention = registry.c128_attention
        else:
            raise ValueError(
                f"layer {layer.index} has no attention implementation for "
                f"compression ratio {layer.compress_ratio}"
            )
        mlp = (
            registry.hash_moe
            if layer.mlp is MLPKind.HASH_MOE
            else registry.routed_moe
        )
        resolved.append(LayerComponents(attention, mlp))
    return tuple(resolved)
