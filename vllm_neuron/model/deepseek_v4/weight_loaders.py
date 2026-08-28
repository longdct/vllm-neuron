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
    expert_id: int | None = None
    transpose: bool = False


_STACKED_COMPONENTS = (
    (".w1", ".gate_up_proj", 0),
    (".w3", ".gate_up_proj", 1),
    # The indexer's own compressor, listed before the outer one. Both rules
    # produce the same destination -- ``resolve_stacked_shard`` matches on a
    # bare substring, so ``.compressor.wkv`` already catches
    # ``.indexer.compressor.wkv`` -- but only by accident. Naming the indexer
    # explicitly, and first, means narrowing the outer rule later cannot
    # silently strand the indexer's weights.
    (".indexer.compressor.wkv", ".indexer.compressor.fused_wkv_wgate", 0),
    (".indexer.compressor.wgate", ".indexer.compressor.fused_wkv_wgate", 1),
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
    # The lightning indexer's query projection. It needs its own rule because
    # the ``.attn.wq_b`` rule above does not match ``.attn.indexer.wq_b`` --
    # deliberately, since the two are different weights on different modules.
    # ``weights_proj`` needs no rule: the checkpoint's name already matches.
    (".attn.indexer.wq_b", ".attn.indexer.q_b_proj"),
    # The compressor keeps its norm as a bare parameter rather than a submodule.
    # Matches the indexer's compressor as well as the outer one, which is what
    # is wanted -- they are the same class.
    (".compressor.norm.weight", ".compressor.norm_weight"),
)


def _stack_width(parameter_name: str) -> int:
    """How many checkpoint tensors fuse into *parameter_name*."""
    width = 0
    for _, target, shard_id in _STACKED_COMPONENTS:
        if target in parameter_name:
            width = max(width, shard_id + 1)
    if width == 0:
        raise ValueError(f"{parameter_name!r} is not a fused DeepSeek-V4 parameter")
    return width


#: Checkpoint subtrees this plugin deliberately has no parameters for.
#:
#: Empty since the lightning indexer landed. It used to hold
#: ``".attn.indexer."``: below the dense-CSA bound the indexer only *selects*
#: and never weights, so skipping it was exact, and its checkpoint tensors had
#: nowhere to go. They have somewhere to go now.
#:
#: Kept rather than deleted because "expected in a real checkpoint, expected to
#: have no destination" is a distinct state from a mapping failure, and the next
#: unmodelled subtree (MTP, say) will need to say so again.
_UNSUPPORTED_SUBTREES: tuple[str, ...] = ()

#: Container renames from the checkpoint's layout to this plugin's module tree.
#:
#: Applied *after* the leaf renames above, which still match on ``.attn.``/
#: ``.ffn.``. Note these are this plugin's names, not vLLM's: the decoder layer
#: holds ``attention``/``moe`` submodules and ``input_layernorm``/
#: ``post_attention_layernorm``, so mapping to vLLM's ``attn``/``ffn`` spelling
#: names parameters that do not exist here.
_CONTAINER_RENAMES = (
    (".attn_norm.weight", ".input_layernorm.weight"),
    (".ffn_norm.weight", ".post_attention_layernorm.weight"),
    (".attn.", ".attention."),
    (".ffn.", ".moe."),
)

#: MoE parameters that are bare ``nn.Parameter`` tensors rather than ``Linear``
#: submodules, so the checkpoint's trailing ``.weight`` has to be dropped.
_BARE_MOE_PARAMS = re.compile(
    r"\.moe\.(experts\.\d+|shared_experts)\.(gate_up_proj|down_proj|w[123])\.weight$"
)

#: mHC (manifold-constrained hyper-connection) parameter renames.
#:
#: The checkpoint flattens these into ``hc_<site>_<param>`` names, while the model
#: holds them on real submodules -- ``attn_hc``/``ffn_hc`` per layer and
#: ``hc_head`` on the model -- whose parameters are ``fn``, ``base`` and
#: ``hc_scale``. Note the deliberate asymmetry: ``fn``/``base`` drop the ``hc_``
#: prefix but ``hc_scale`` keeps it, matching the module definitions in
#: ``model.py``.
#:
#: Without these rules every mHC tensor maps to a non-existent parameter. That
#: went unnoticed because the tiny gate runs ``--load-format dummy``, so no real
#: checkpoint weights were ever mapped.
_MHC_RENAMES = (
    (".hc_attn_scale", ".attn_hc.hc_scale"),
    (".hc_attn_base", ".attn_hc.base"),
    (".hc_attn_fn", ".attn_hc.fn"),
    (".hc_ffn_scale", ".ffn_hc.hc_scale"),
    (".hc_ffn_base", ".ffn_hc.base"),
    (".hc_ffn_fn", ".ffn_hc.fn"),
    ("hc_head_scale", "hc_head.hc_scale"),
    ("hc_head_base", "hc_head.base"),
    ("hc_head_fn", "hc_head.fn"),
)


def map_checkpoint_name(name: str, expert_dtype: ExpertDType = "bf16") -> str:
    """Map an official V4 checkpoint key onto this plugin's parameter names.

    The target is ``vllm_neuron.model.deepseek_v4.model``'s module tree, which is
    what ``load_checkpoint_weights`` resolves against. That tree is not vLLM's:
    the decoder layer holds ``attention``/``moe`` submodules, the mHC parameters
    live on ``attn_hc``/``ffn_hc``/``hc_head``, and the expert projections are
    bare parameters without a ``.weight`` suffix.

    Keeping the pure string transformation here makes checkpoint compatibility
    testable without constructing a model.
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
        # The router's score correction lives directly on the MoE block here,
        # not on the gate submodule.
        mapped = mapped[: -len(".ffn.gate.bias")] + ".ffn.correction_bias"
    elif mapped.endswith(".ffn.gate.tid2eid"):
        # Hash-routing table: a registered buffer on the MoE block.
        mapped = mapped[: -len(".ffn.gate.tid2eid")] + ".ffn.tid2eid"

    # ``w2`` is the down projection for both the shared expert and each routed
    # one; ``w1``/``w3`` are stacked later by resolve_stacked_shard.
    mapped = mapped.replace(".shared_experts.w2", ".shared_experts.down_proj")
    for source, target in _ATTENTION_RENAMES:
        mapped = mapped.replace(source, target)
    for source, target in _MHC_RENAMES:
        mapped = mapped.replace(source, target)
    for source, target in _CONTAINER_RENAMES:
        mapped = mapped.replace(source, target)
    if mapped.endswith(".scale"):
        if expert_dtype == "fp4" and re.search(
            r"\.experts\.\d+\.w[123]\.scale$", mapped
        ):
            mapped = mapped[: -len(".scale")] + ".weight_scale"
        else:
            mapped = mapped[: -len(".scale")] + ".weight_scale_inv"
    elif _BARE_MOE_PARAMS.search(mapped):
        mapped = mapped[: -len(".weight")]
    return mapped


#: Model tensors that no checkpoint is expected to fill, so ``strict`` loading
#: must not treat their absence as an incomplete checkpoint.
#:
#: * ``identity_kv_weight`` -- the derived identity K/V projection, reconstructed
#:   by ``reinitialize_deterministic_buffers``.
#: * ``correction_bias`` -- only routed (``noaux_tc``) layers ship
#:   ``ffn.gate.bias``; hash-routed layers keep the zero default.
#: * ``tid2eid`` -- only hash-routed layers ship a table; routed layers keep the
#:   config-derived fallback, which their routing never reads.
#: * ``rotary_emb.*_inv_freq`` -- RoPE frequency tables, non-persistent buffers
#:   in Transformers by design, recomputed from config after ``to_empty()``.
_DETERMINISTIC_SUFFIXES = (
    ".attention.identity_kv_weight",
    ".moe.correction_bias",
    ".moe.tid2eid",
    "_inv_freq",
)


def is_deterministically_initialized(parameter_name: str) -> bool:
    """True for tensors the model derives rather than loads."""
    return parameter_name.endswith(_DETERMINISTIC_SUFFIXES)


def is_unsupported_checkpoint_name(name: str) -> bool:
    """True for checkpoint subtrees this plugin intentionally does not model.

    Distinct from a mapping failure: these tensors are expected to be present in
    a real checkpoint and expected to have nowhere to go.
    """
    return any(subtree in name for subtree in _UNSUPPORTED_SUBTREES)


def resolve_stacked_shard(mapped_name: str) -> StackedShard | None:
    """Resolve fused attention/compressor/MLP shards after namespace mapping.

    Routed experts stack exactly like the shared expert: this plugin gives each
    expert its own ``gate_up_proj`` parameter holding the ``w1``/``w3``
    concatenation, rather than a single grouped tensor addressed by expert id.
    """
    routed = re.search(
        r"^(.*\.moe)\.experts\.(\d+)\.w([123])(?:\.weight)?$", mapped_name
    )
    if routed:
        prefix, expert, component = routed.groups()
        if component == "2":
            return StackedShard(f"{prefix}.routed_down", 0, int(expert), transpose=True)
        return StackedShard(
            f"{prefix}.routed_gate_up",
            0 if component == "1" else 1,
            int(expert),
            transpose=True,
        )
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


def _copy_into(destination: torch.Tensor, source: torch.Tensor) -> None:
    """``destination.copy_(source)``, casting on the host side first.

    Neuron rejects a copy that changes device *and* dtype in one step
    ("Expected self.dtype() == dst.dtype() to be true"), which CPU performs
    silently. Checkpoints are BF16 while these parameters are FP32, so every
    real device load hits exactly that combination -- and only on device, which
    is why the CPU oracle never surfaced it.

    Casting while the source is still on the host leaves a pure transfer for the
    device, and is numerically identical to the upcast CPU was doing implicitly.
    """
    destination.copy_(source.to(destination.dtype))


def load_checkpoint_weights(
    module,
    weights,
    *,
    expert_dtype: ExpertDType = "bf16",
    strict: bool = False,
) -> set[str]:
    """Load a checkpoint iterator through the mapped/fused parameter contract."""
    torch = __import__("torch")
    # Buffers as well as parameters: the hash-routing table ``tid2eid`` is a
    # registered buffer, and a checkpoint that provides one must be able to
    # override the config-derived default rather than silently keeping it.
    params = {**dict(module.named_parameters()), **dict(module.named_buffers())}
    modules = dict(module.named_modules())
    loaded_params: set[str] = set()
    loaded_destinations: set[tuple[str, int | None]] = set()
    source_names: set[str] = set()

    for source_name, loaded_weight in weights:
        if source_name in source_names:
            raise ValueError(f"duplicate DeepSeek-V4 checkpoint weight {source_name!r}")
        source_names.add(source_name)
        if is_unsupported_checkpoint_name(source_name):
            continue
        mapped_name = map_checkpoint_name(source_name, expert_dtype)
        stacked = resolve_stacked_shard(mapped_name)
        target_name = stacked.parameter_name if stacked else mapped_name
        if stacked is not None and stacked.expert_id is not None:
            owner_name = target_name.rsplit(".", 1)[0]
            owner = modules.get(owner_name)
            local_start = int(getattr(owner, "local_start", 0))
            local_end = int(
                getattr(owner, "local_end", local_start + params[target_name].shape[0])
            )
            # Non-local experts are intentionally absent from this rank.  Skip
            # before touching the tensor payload in streaming callers.
            if not local_start <= stacked.expert_id < local_end:
                continue
        shard_id = stacked.shard_id if stacked else None
        destination = (target_name, shard_id, stacked.expert_id if stacked else None)
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
                if stacked.expert_id is not None:
                    owner = modules.get(target_name.rsplit(".", 1)[0])
                    local_start = int(getattr(owner, "local_start", 0))
                    expert_id = stacked.expert_id - local_start
                    if expert_id < 0 or expert_id >= parameter.shape[0]:
                        raise ValueError(
                            f"routed expert {expert_id} is outside {target_name!r}"
                        )
                    target = (
                        parameter[expert_id, :, stacked.shard_id, :]
                        if target_name.endswith("routed_gate_up")
                        else parameter[expert_id]
                    )
                    source = (
                        loaded_weight.transpose(0, 1)
                        if stacked.transpose
                        else loaded_weight
                    )
                    expert_tp_degree = int(getattr(owner, "expert_tp_degree", 1))
                    expert_tp_rank = int(getattr(owner, "expert_tp_rank", 0))
                    if expert_tp_degree > 1:
                        shard_dim = 1 if target_name.endswith("routed_gate_up") else 0
                        shard_size = source.shape[shard_dim] // expert_tp_degree
                        source = source.narrow(
                            shard_dim, expert_tp_rank * shard_size, shard_size
                        )
                    require_weight_shape(
                        source_name, tuple(source.shape), tuple(target.shape)
                    )
                    _copy_into(target, source)
                elif weight_loader is not None:
                    if hasattr(weight_loader, "load"):

                        class _TensorSlice:
                            def __init__(self, tensor):
                                self.tensor = tensor

                            def get_shape(self):
                                return tuple(self.tensor.shape)

                            def __getitem__(self, index):
                                return self.tensor[index]

                        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                        transformed = weight_loader.load([_TensorSlice(loaded_weight)], rank)
                        width = _stack_width(target_name)
                        chunk = parameter.shape[0] // width
                        require_weight_shape(
                            source_name, tuple(transformed.shape), (chunk, *parameter.shape[1:])
                        )
                        _copy_into(
                            parameter[shard_id * chunk : (shard_id + 1) * chunk],
                            transformed,
                        )
                    else:
                        weight_loader(parameter, loaded_weight, shard_id)
                else:
                    # No shard loader: place the shard directly. A fused
                    # parameter is its shards concatenated on dim 0 in shard-id
                    # order, so shard i owns an equal slice of the rows.
                    #
                    # This cannot rely on a ``weight_loader`` attribute set in
                    # __init__: ``load_weights`` calls ``to_empty()`` first,
                    # which rebuilds every parameter and drops attributes hung
                    # on the old objects.
                    width = _stack_width(target_name)
                    chunk, remainder = divmod(parameter.shape[0], width)
                    if remainder:
                        raise ValueError(
                            f"fused DeepSeek-V4 parameter {target_name!r} has "
                            f"{parameter.shape[0]} rows, not divisible into "
                            f"{width} shards"
                        )
                    owner = modules.get(target_name.rsplit(".", 1)[0])
                    tp_degree = int(getattr(owner, "tp_degree", 1))
                    tp_rank = int(getattr(owner, "tp_rank", 0))
                    source = loaded_weight
                    if tp_degree > 1 and target_name.endswith("shared_experts.gate_up_proj"):
                        shard_size = source.shape[0] // tp_degree
                        source = source.narrow(0, tp_rank * shard_size, shard_size)
                    require_weight_shape(
                        source_name,
                        tuple(source.shape),
                        (chunk, *tuple(parameter.shape[1:])),
                    )
                    _copy_into(
                        parameter[shard_id * chunk : (shard_id + 1) * chunk],
                        source,
                    )
            elif weight_loader is not None:
                if hasattr(weight_loader, "load"):

                    class _TensorSlice:
                        def __init__(self, tensor):
                            self.tensor = tensor

                        def get_shape(self):
                            return tuple(self.tensor.shape)

                        def __getitem__(self, index):
                            return self.tensor[index]

                    rank = 0
                    if torch.distributed.is_initialized():
                        rank = torch.distributed.get_rank()
                    transformed = weight_loader.load(
                        [_TensorSlice(loaded_weight)], rank
                    )
                    require_weight_shape(
                        source_name, tuple(transformed.shape), tuple(parameter.shape)
                    )
                    _copy_into(parameter, transformed)
                else:
                    weight_loader(parameter, loaded_weight)
            else:
                require_weight_shape(
                    source_name,
                    tuple(loaded_weight.shape),
                    tuple(parameter.shape),
                )
                _copy_into(parameter, loaded_weight)
        loaded_destinations.add(destination)
        loaded_params.add(target_name)
    if strict:
        missing = sorted(
            name
            for name in set(params) - loaded_params
            if not is_deterministically_initialized(name)
        )
        if missing:
            raise ValueError(
                "DeepSeek-V4 checkpoint did not load parameter(s): "
                + ", ".join(missing)
            )
    return loaded_params
