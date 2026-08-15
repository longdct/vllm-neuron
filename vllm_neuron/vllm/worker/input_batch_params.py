# SPDX-License-Identifier: Apache-2.0
"""Per-KV-cache-group parameters for ``InputBatch`` construction.

Split out of ``neuron_model_runner`` deliberately. The derivation below needs
only vLLM's KV cache interfaces -- no torch device setup, no NKI kernels, no
``vllm_neuron.model`` stack -- so keeping it here makes it importable, and
therefore testable, wherever vLLM is installed. Importing the runner instead
pulls in the whole Neuron model stack (``vllm_neuron.functional`` -> ``nki`` ->
``nkilib``), which is unavailable on any interpreter without the Neuron SDK.

That matters because the heterogeneous multi-group case is what the DeepSeek-V4
cache work is built on, and it needs to be testable before Neuron hardware is.
Same convention as ``vllm_neuron/vllm/scheduler_selection.py``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)
from vllm.v1.worker.block_table import SlotMappingMode

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class InputBatchGroupParams:
    """The four per-KV-cache-group lists ``InputBatch`` needs, in group order.

    All four are positionally aligned: index *i* describes the same cache group
    in every list. ``InputBatch`` relies on that alignment without checking it,
    so they are built together here rather than at four separate call sites.
    """

    block_sizes: list[int]
    kernel_block_sizes: list[int]
    max_num_blocks_per_req: list[int]
    slot_mapping_modes: list[SlotMappingMode]

    def __post_init__(self) -> None:
        lengths = {
            "block_sizes": len(self.block_sizes),
            "kernel_block_sizes": len(self.kernel_block_sizes),
            "max_num_blocks_per_req": len(self.max_num_blocks_per_req),
            "slot_mapping_modes": len(self.slot_mapping_modes),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"InputBatch group parameter lists must be the same length, got "
                f"{lengths}. They are positionally aligned per KV cache group; a "
                f"length mismatch means some group's block size would be paired "
                f"with another group's block count."
            )


def mla_cache_shape(spec, num_blocks: int) -> tuple[int, int, int, int]:
    """Physical single-tensor MLA layout for one cache layer."""
    if num_blocks < 0:
        raise ValueError("num_blocks must be non-negative")
    if spec.block_size % spec.compress_ratio:
        raise ValueError(
            f"MLA block_size={spec.block_size} must be divisible by "
            f"compress_ratio={spec.compress_ratio}"
        )
    return (
        num_blocks,
        spec.num_kv_heads,
        spec.storage_block_size,
        spec.head_size,
    )


def build_input_batch_group_params(
    kv_cache_config: KVCacheConfig,
    vllm_config: "VllmConfig",
    max_model_len: int,
    *,
    is_encoder_decoder: bool = False,
) -> InputBatchGroupParams:
    """Derive ``InputBatch``'s per-group parameters from a KV cache config.

    Mirrors upstream ``GPUModelRunner`` (vLLM 0.26), with two deliberate Neuron
    deltas:

    * ``kernel_block_sizes`` tracks ``block_sizes``. Upstream lets a kernel
      address a sub-block; Neuron's QKV scatter and segmented attention gather
      both divide/modulo by the page ``block_size`` directly, so a distinct
      kernel block size is not representable. Tests pin this backend invariant.
    * Encoder-only groups are skipped -- matching upstream. They carry no
      per-request block table, so contributing an entry would shift every later
      group's index and silently pair specs with the wrong block counts.

    Heterogeneous groups are the reason this is a function rather than a
    comprehension: DeepSeek-V4 mixes sliding-window, latent-MLA and compressed
    caches with different block sizes *and* different per-request block counts in
    one model, so these lists genuinely differ element to element instead of
    being one value repeated.
    """
    if is_encoder_decoder:
        raise NotImplementedError(
            "Neuron does not support encoder-decoder cache block tables: their "
            "rows must be sized with max(max_model_len, max_encoder_len), but "
            "NeuronModelRunner has no max_encoder_len lifecycle yet"
        )

    block_sizes: list[int] = []
    kernel_block_sizes: list[int] = []
    max_num_blocks: list[int] = []
    slot_mapping_modes: list[SlotMappingMode] = []

    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        kind = get_kv_cache_spec_kind(spec)
        if kind == KVCacheSpecKind.ENCODER_ONLY_ATTENTION:
            continue

        block_sizes.append(spec.block_size)
        kernel_block_sizes.append(spec.block_size)
        max_num_blocks.append(spec.max_num_blocks_per_req(vllm_config, max_model_len))
        # Mamba-style groups have no token->slot mapping; everything Neuron
        # currently serves does.
        slot_mapping_modes.append(
            SlotMappingMode.NONE
            if kind == KVCacheSpecKind.MAMBA
            else SlotMappingMode.TOKEN_TO_KV_SLOT
        )

    return InputBatchGroupParams(
        block_sizes=block_sizes,
        kernel_block_sizes=kernel_block_sizes,
        max_num_blocks_per_req=max_num_blocks,
        slot_mapping_modes=slot_mapping_modes,
    )
