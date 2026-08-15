# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from enum import Enum

import torch


class CacheKind(str, Enum):
    FULL = "full"
    SLIDING_WINDOW = "sliding_window"
    MLA = "mla"
    COMPRESSOR_STATE = "compressor_state"
    RSWA = "rswa"


@dataclass
class LayerSpec:
    """
    Defines the KV cache specification for a single transformer layer.

    Used to specify the memory requirements and configuration for storing
    key-value pairs in the attention mechanism of a transformer layer.
    """

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None
    cache_kind: CacheKind = CacheKind.FULL
    compress_ratio: int = 1
    alignment: int | None = None
    rswa_window: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.cache_kind, str):
            self.cache_kind = CacheKind(self.cache_kind)
        # Backward compatibility for existing model declarations: before
        # CacheKind existed, setting sliding_window_size selected SWA.
        if (
            self.cache_kind is CacheKind.FULL
            and self.sliding_window_size is not None
            and self.chunk_size is None
        ):
            self.cache_kind = CacheKind.SLIDING_WINDOW
        if self.compress_ratio < 1:
            raise ValueError("compress_ratio must be positive")
        if self.cache_kind is CacheKind.MLA:
            if self.sliding_window_size is not None:
                raise ValueError("MLA and sliding_window_size are separate cache kinds")
        elif self.compress_ratio != 1:
            raise ValueError("compress_ratio is only valid for MLA caches")
        if self.cache_kind is CacheKind.SLIDING_WINDOW and self.sliding_window_size is None:
            raise ValueError("sliding-window cache requires sliding_window_size")
        if self.cache_kind is CacheKind.COMPRESSOR_STATE and self.sliding_window_size is None:
            raise ValueError("compressor carry state requires a lifecycle window")
        if self.cache_kind is CacheKind.RSWA and self.rswa_window is None:
            raise ValueError("R-SWA cache requires rswa_window")


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]
