# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from enum import Enum

import torch


class CacheKind(str, Enum):
    FULL = "full"
    SLIDING_WINDOW = "sliding_window"
    MLA = "mla"
    SLIDING_WINDOW_MLA = "sliding_window_mla"
    COMPRESSOR_STATE = "compressor_state"
    RSWA = "rswa"
    #: Per-request recurrent state for a linear-attention layer (Gated
    #: DeltaNet). Unlike every other kind here this is not paged per token: one
    #: page holds one request's whole state, because the state is an unbounded
    #: accumulator rather than a windowed function of recent tokens. Declared
    #: with ``state_shapes`` / ``state_dtypes`` instead of
    #: ``num_kv_heads`` / ``head_size``, and mapped onto vLLM's ``MambaSpec``.
    MAMBA = "mamba"


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
    block_size: int | None = None
    sliding_window_size: int | None = None
    chunk_size: int | None = None
    cache_kind: CacheKind = CacheKind.FULL
    compress_ratio: int = 1
    alignment: int | None = None
    rswa_window: int | None = None
    #: For :attr:`CacheKind.MAMBA` only. One entry per state tensor the layer
    #: carries -- Gated DeltaNet has two, a conv window and a recurrent state --
    #: with a matching dtype each. The recurrent state is an fp32 accumulator
    #: regardless of ``--kv-cache-dtype``, which is why the dtypes travel with
    #: the shapes rather than being inherited from the cache config.
    state_shapes: tuple[tuple[int, ...], ...] | None = None
    state_dtypes: tuple[torch.dtype, ...] | None = None

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
        if self.block_size is not None and self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.cache_kind is CacheKind.MLA:
            if self.sliding_window_size is not None:
                raise ValueError("MLA and sliding_window_size are separate cache kinds")
        elif self.compress_ratio != 1:
            raise ValueError("compress_ratio is only valid for MLA caches")
        if self.cache_kind in (
            CacheKind.SLIDING_WINDOW,
            CacheKind.SLIDING_WINDOW_MLA,
        ) and self.sliding_window_size is None:
            raise ValueError("sliding-window cache requires sliding_window_size")
        if self.cache_kind is CacheKind.COMPRESSOR_STATE and self.sliding_window_size is None:
            raise ValueError("compressor carry state requires a lifecycle window")
        if self.cache_kind is CacheKind.RSWA and self.rswa_window is None:
            raise ValueError("R-SWA cache requires rswa_window")
        if self.cache_kind is CacheKind.MAMBA:
            if not self.state_shapes or not self.state_dtypes:
                raise ValueError(
                    "mamba cache requires state_shapes and state_dtypes; its "
                    "geometry is per-request state tensors, not num_kv_heads x "
                    "head_size pages"
                )
            if len(self.state_shapes) != len(self.state_dtypes):
                raise ValueError(
                    f"mamba cache has {len(self.state_shapes)} state shapes but "
                    f"{len(self.state_dtypes)} dtypes; one dtype per state tensor"
                )
            if self.sliding_window_size is not None:
                raise ValueError(
                    "mamba state has no sliding window: it is an unbounded "
                    "accumulator, allocated one page per request"
                )
        elif self.state_shapes is not None or self.state_dtypes is not None:
            raise ValueError(
                "state_shapes/state_dtypes are only valid for mamba caches"
            )


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]
