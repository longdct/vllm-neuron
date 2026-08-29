# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 checkpoint -> parameter mapping.

Two of the model's silent-failure traps live here, so both are implemented once
and pinned by tests:

**The RMSNorm fold.** HuggingFace's ``Qwen3_5RMSNorm`` stores a zero-initialized
weight and computes ``x_norm * (1.0 + weight)``. This backend implements the
ordinary ``weight * x_norm`` everywhere, so every such tensor gets ``+1`` at
load. Loading the raw tensor instead does not fail -- it produces near-zero
activations and a model that still runs.

Note the asymmetry: the Gated DeltaNet's ``norm`` is HF's *gated* RMSNorm, which
already uses the ordinary convention and is initialized to **ones**. Folding
that one too would be just as wrong as not folding the others.

**The per-head query gate.** ``q_proj`` emits ``num_heads * head_dim * 2`` with
each head storing ``[query | gate]`` adjacently, so the checkpoint rows are
head-major and a rank's shard is a contiguous *run of heads*, not a slice of a
flat query block followed by a slice of a flat gate block.

Text-only: the released checkpoints are multimodal wrappers whose decoder lives
under ``model.language_model.*``. The ``model.visual.*`` subtree is simply never
named in the mapping, so it is skipped without needing a filter.
"""

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

from .config import Qwen3_5TextConfig
from .parallel import Qwen3_5ShardingPolicy

#: Prefix of the text decoder inside the shipped multimodal checkpoints.
TEXT_PREFIX = "model.language_model"
#: Prefix of the vision tower, deliberately unmapped.
VISION_PREFIX = "model.visual"


# ===========================================================================
# Norm folding
# ===========================================================================


def norm_plus_one_loader() -> SafetensorsWeightLoader:
    """Load an HF ``Qwen3_5RMSNorm`` weight into an ordinary RMSNorm.

    HF applies ``(1.0 + weight)``; folding the 1 in here keeps the runtime graph
    -- and the decode kernel's gamma argument, which takes a plain scale -- in a
    single convention.
    """

    def transform(slices, rank):  # noqa: ARG001 - norms are not sharded
        return slices[0][:].float().add(1.0)

    return SafetensorsWeightLoader(transform=transform)


def plain_loader() -> SafetensorsWeightLoader:
    """Load a tensor unchanged. Used for the gated norm, which needs no fold."""

    def transform(slices, rank):  # noqa: ARG001
        return slices[0][:]

    return SafetensorsWeightLoader(transform=transform)


# ===========================================================================
# Fused, gated QKV
# ===========================================================================


def gated_qkv_weight_loader(
    config: Qwen3_5TextConfig, policy: Qwen3_5ShardingPolicy
) -> SafetensorsWeightLoader:
    """Fuse ``[q_proj (with per-head gate) | k_proj | v_proj]`` for one rank.

    Checkpoint tensors are ``[out_features, hidden]``; the parameter is
    ``[hidden, fused]``, so the result is transposed at the end.

    The query block keeps the checkpoint's head-major ``[q | gate]`` interleave,
    which is what lets :func:`~vllm_neuron.model.qwen3_5.attention.split_query_and_gate`
    recover the two by viewing as ``[..., heads, 2 * head_dim]``.

    Query heads are padded up to a multiple of the TP degree with **zero** rows.
    A zero query head produces a zero attention output, and the matching
    ``o_proj`` rows are zeroed too, so padded heads contribute nothing to the
    reduced result rather than merely being ignored.
    """
    head_dim = config.head_dim
    num_q_heads = config.num_attention_heads
    q_per_rank = policy.q_heads_per_rank
    kv_per_rank = policy.kv_heads_per_rank
    replicas = policy.num_kv_replicas

    def transform(slices, rank):
        q_slice, k_slice, v_slice = slices

        # --- query + gate, head-major, padded ---
        first_head = rank * q_per_rank
        rows = []
        for local in range(q_per_rank):
            head = first_head + local
            if head < num_q_heads:
                lo = head * 2 * head_dim
                rows.append(q_slice[lo : lo + 2 * head_dim, :][:])
            else:
                # Padded head: zeros, so it cannot perturb the output sum.
                template = q_slice[0 : 2 * head_dim, :][:]
                rows.append(torch.zeros_like(template))
        q_part = torch.cat(rows, dim=0)

        # --- key / value ---
        kv_rank = rank // replicas if replicas > 1 else rank
        kv_lo = kv_rank * kv_per_rank * head_dim
        kv_hi = kv_lo + kv_per_rank * head_dim
        k_part = k_slice[kv_lo:kv_hi, :][:]
        v_part = v_slice[kv_lo:kv_hi, :][:]

        fused = torch.cat([q_part, k_part, v_part], dim=0)
        return fused.t()

    return SafetensorsWeightLoader(transform=transform)


def gated_o_proj_weight_loader(
    config: Qwen3_5TextConfig, policy: Qwen3_5ShardingPolicy
) -> SafetensorsWeightLoader:
    """Shard ``o_proj`` by query head, zeroing the padded heads.

    Checkpoint is ``[hidden, num_heads * head_dim]``; the parameter is
    ``[num_heads_per_rank * head_dim, hidden]``.
    """
    head_dim = config.head_dim
    num_q_heads = config.num_attention_heads
    q_per_rank = policy.q_heads_per_rank

    def transform(slices, rank):
        o_slice = slices[0]
        first_head = rank * q_per_rank
        cols = []
        for local in range(q_per_rank):
            head = first_head + local
            if head < num_q_heads:
                lo = head * head_dim
                cols.append(o_slice[:, lo : lo + head_dim][:])
            else:
                template = o_slice[:, 0:head_dim][:]
                cols.append(torch.zeros_like(template))
        return torch.cat(cols, dim=1).t()

    return SafetensorsWeightLoader(transform=transform)


# ===========================================================================
# Gated DeltaNet
# ===========================================================================
#
# Unlike the attention block, whose parameters are raw tensors stored
# ``[hidden, out]``, the GDN keeps ``nn.Linear`` / ``nn.Conv1d`` modules whose
# weights are already ``[out, in]`` -- the same layout as the checkpoint. So
# none of the loaders below transposes; adding one would be silently wrong.
#
# Every partition comes from the policy's row-range helpers, so which rows a
# rank owns is stated once, in ``Qwen3_5ShardingPolicy``, and never restated.


def _rows(tensor_slice, ranges, dim: int = 0):
    """Concatenate the given half-open ranges along ``dim``."""
    parts = []
    for lo, hi in ranges:
        if dim == 0:
            parts.append(tensor_slice[lo:hi][:])
        else:
            parts.append(tensor_slice[:, lo:hi][:])
    return torch.cat(parts, dim=dim)


def gdn_qkv_weight_loader(policy: Qwen3_5ShardingPolicy) -> SafetensorsWeightLoader:
    """Shard ``in_proj_qkv`` over the fused ``[q | k | v]`` output rows.

    At tp > 16 the q and k rows are *replicated* between the ranks sharing a
    key head while only the v rows are split, which is why the result is
    ``conv_dim_per_rank`` wide and not ``conv_dim // tp``.
    """

    def transform(slices, rank):
        return _rows(slices[0], policy.conv_row_ranges(rank))

    return SafetensorsWeightLoader(transform=transform)


def gdn_conv1d_weight_loader(policy: Qwen3_5ShardingPolicy) -> SafetensorsWeightLoader:
    """Shard the depthwise ``conv1d`` weight, ``[conv_dim, 1, kernel]``.

    Depthwise means one filter per channel, so the channel partition is exactly
    the ``in_proj_qkv`` row partition -- reusing it keeps the conv and its input
    from ever drifting apart.
    """

    def transform(slices, rank):
        w = slices[0]
        parts = [w[lo:hi, :, :][:] for lo, hi in policy.conv_row_ranges(rank)]
        return torch.cat(parts, dim=0)

    return SafetensorsWeightLoader(transform=transform)


def gdn_row_weight_loader(policy: Qwen3_5ShardingPolicy) -> SafetensorsWeightLoader:
    """Shard a ``[value_dim, hidden]`` projection (``in_proj_z``) by value rows."""

    def transform(slices, rank):
        return _rows(slices[0], policy.value_row_ranges(rank))

    return SafetensorsWeightLoader(transform=transform)


def gdn_out_proj_weight_loader(
    policy: Qwen3_5ShardingPolicy,
) -> SafetensorsWeightLoader:
    """Shard ``out_proj``, ``[hidden, value_dim]``, along its **input** dim.

    Each rank then produces a partial sum over the value dimension, which the
    layer's exit collective reduces.
    """

    def transform(slices, rank):
        return _rows(slices[0], policy.value_row_ranges(rank), dim=1)

    return SafetensorsWeightLoader(transform=transform)


def gdn_head_vector_loader(policy: Qwen3_5ShardingPolicy) -> SafetensorsWeightLoader:
    """Shard a per-value-head quantity by whole heads.

    Covers both the ``[num_v_heads, hidden]`` gate projections
    (``in_proj_b`` / ``in_proj_a``) and the bare ``[num_v_heads]`` parameters
    ``dt_bias`` and ``A_log``, which differ only in rank.
    """

    def transform(slices, rank):
        w = slices[0]
        return torch.cat([w[h : h + 1][:] for h in policy.v_head_indices(rank)], dim=0)

    return SafetensorsWeightLoader(transform=transform)


def gdn_gated_norm_loader(policy: Qwen3_5ShardingPolicy) -> SafetensorsWeightLoader:
    """Slice the gated norm's ``[value_head_dim]`` weight to this rank's width.

    No ``+1`` fold: HF's ``Qwen3_5RMSNormGated`` already uses the ordinary
    convention (see the module docstring). Below tp=32 this is the identity.
    """

    def transform(slices, rank):
        lo = policy.v_dim_offset(rank)
        return slices[0][lo : lo + policy.v_dim_per_rank][:]

    return SafetensorsWeightLoader(transform=transform)


# ===========================================================================
# Checkpoint key mapping
# ===========================================================================


def text_weight_mappings(config: Qwen3_5TextConfig) -> dict:
    """Parameter name -> checkpoint key(s), text decoder only.

    ``model.visual.*`` is never referenced, so the vision tower is skipped by
    omission rather than by a filter that could drift.
    """
    mappings = {
        "model.embed_tokens.weight": f"{TEXT_PREFIX}.embed_tokens.weight",
        "model.norm.weight": f"{TEXT_PREFIX}.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for i, layer_type in enumerate(config.layer_types):
        param = f"model.layers.{i}"
        ckpt = f"{TEXT_PREFIX}.layers.{i}"

        mappings[f"{param}.input_layernorm.weight"] = f"{ckpt}.input_layernorm.weight"
        mappings[f"{param}.post_attention_layernorm.weight"] = (
            f"{ckpt}.post_attention_layernorm.weight"
        )

        mappings[f"{param}.mlp.gate_proj_weight"] = f"{ckpt}.mlp.gate_proj.weight"
        mappings[f"{param}.mlp.up_proj_weight"] = f"{ckpt}.mlp.up_proj.weight"
        mappings[f"{param}.mlp.down_proj_weight"] = f"{ckpt}.mlp.down_proj.weight"

        if layer_type == "full_attention":
            mappings[f"{param}.self_attn.qkv_proj_weight"] = [
                f"{ckpt}.self_attn.q_proj.weight",
                f"{ckpt}.self_attn.k_proj.weight",
                f"{ckpt}.self_attn.v_proj.weight",
            ]
            mappings[f"{param}.self_attn.o_proj_weight"] = (
                f"{ckpt}.self_attn.o_proj.weight"
            )
            mappings[f"{param}.self_attn.q_norm.weight"] = (
                f"{ckpt}.self_attn.q_norm.weight"
            )
            mappings[f"{param}.self_attn.k_norm.weight"] = (
                f"{ckpt}.self_attn.k_norm.weight"
            )
        else:
            gdn = f"{param}.linear_attn"
            src = f"{ckpt}.linear_attn"
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                mappings[f"{gdn}.{name}.weight"] = f"{src}.{name}.weight"
            mappings[f"{gdn}.conv1d.weight"] = f"{src}.conv1d.weight"
            mappings[f"{gdn}.dt_bias"] = f"{src}.dt_bias"
            mappings[f"{gdn}.A_log"] = f"{src}.A_log"
            # The gated norm already uses the ordinary convention -- no fold.
            mappings[f"{gdn}.norm.weight"] = f"{src}.norm.weight"

    return mappings


#: Parameter-name suffixes whose checkpoint tensors need the ``+1`` fold.
#: Everything HF declares as ``Qwen3_5RMSNorm``; deliberately excludes
#: ``linear_attn.norm``, which is the gated variant.
FOLDED_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "model.norm.weight",
)


def needs_plus_one_fold(param_name: str) -> bool:
    """True when this parameter is an HF ``Qwen3_5RMSNorm`` weight."""
    if param_name.endswith("linear_attn.norm.weight"):
        return False  # gated norm: ordinary convention, initialized to ones
    return any(param_name.endswith(suffix) for suffix in FOLDED_NORM_SUFFIXES)
