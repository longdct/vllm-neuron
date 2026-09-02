# SPDX-License-Identifier: Apache-2.0
"""Per-channel FP8 variants of the Qwen3.5 MLP and full-attention modules.

These subclass their BF16 siblings and override two things: the parameter
declarations (FP8 weight plus an fp32 row scale) and the kernel call. Every
other part of the module -- sequence-parallel collectives, QK-norm, partial
RoPE, the output gate, cache reads -- is precision independent and stays in
:mod:`vllm_neuron.model.qwen3_5.model`.

That is a deliberate departure from ``llama3``, which keeps a full 1300-line
copy of its attention and MLP per quantization mode. Qwen3.5's kernel calls
were factored into ``_mlp``/``_qkv_proj``/``_out_proj`` seams precisely so the
quantized path could be a handful of overrides instead, and so a change to
RoPE or the output gate cannot land in one precision and be forgotten in the
other.

Coverage
--------
This reaches the three kernel-path projections -- **69.9%** of the served
weights of the 27B (MLP 61.6%, attention 6.0%, measured from the checkpoint).
The 48 GatedDeltaNet layers (20.7%) keep BF16: their five projections are
plain ``nn.Linear`` with no kernel quantization path. ``embed_tokens`` and
``lm_head`` (9.5%) also stay BF16.

Why ROW
-------
``QuantizationType.ROW`` is Trn2 per-channel FP8 whose *activation* scales are
computed on device at runtime (``mlp_parameters.py:179``). The alternative,
``STATIC``, asserts ``*_in_scale`` tensors that neither the BF16 checkpoint nor
the official ``Qwen/Qwen3.8-27B-FP8`` release (``activation_scheme="dynamic"``)
provides. ROW therefore needs no calibration set and no offline artifact, and
measured on the real checkpoint it is no less accurate than per-tensor scaling
-- the error is set by e4m3's 3-bit mantissa, not by scale granularity.
"""

import logging

import torch
from nkilib.core.utils.common_types import QuantizationType
from torch import nn

import vllm_neuron.functional as NF
from vllm_neuron.utils.weight_loader import get_weight_loader, set_weight_loader

from .model import Qwen3_5Attention, Qwen3_5MLP
from .weight_loaders_fp8 import (
    FP8_DTYPE,
    fp8_scale_loader,
    fp8_weight_loader,
)

logger = logging.getLogger(__name__)

#: Partition dim the kernels broadcast a weight scale across.
_PMAX = 128

#: Suffix appended to a weight parameter's name to form its scale parameter.
#: The scale is a ``Parameter`` rather than a buffer on purpose: the checkpoint
#: loader iterates ``model.named_parameters()`` only
#: (``utils/checkpoints.py:348``), so a non-persistent buffer would never be
#: filled. llama3 works around that with a second manual load pass; declaring a
#: parameter lets the existing pipeline do it.
SCALE_SUFFIX = "_scale"


def _quantize_parameter(module: nn.Module, name: str) -> None:
    """Swap one ``[contraction, out]`` weight for an FP8 weight + row scale.

    Reuses the loader the BF16 constructor already attached, wrapping it rather
    than restating the sharding, transposition and QKV-fusion rules -- those
    live in ``weight_loaders.py`` and must not be duplicated per precision.
    """
    param = getattr(module, name)
    base_loader = get_weight_loader(param)
    contraction, out = param.shape

    weight = nn.Parameter(
        torch.empty(contraction, out, dtype=FP8_DTYPE, device=param.device),
        requires_grad=False,
    )
    set_weight_loader(weight, fp8_weight_loader(base_loader))
    setattr(module, name, weight)

    scale = nn.Parameter(
        torch.empty(_PMAX, out, dtype=torch.float32, device=param.device),
        requires_grad=False,
    )
    set_weight_loader(scale, fp8_scale_loader(base_loader))
    setattr(module, name + SCALE_SUFFIX, scale)


class Qwen3_5MLPFP8(Qwen3_5MLP):
    """SwiGLU MLP with FP8 gate/up/down weights. 61.6% of the 27B."""

    _FP8_WEIGHTS = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")

    def __init__(self, config, policy):
        super().__init__(config, policy)
        for name in self._FP8_WEIGHTS:
            _quantize_parameter(self, name)

    def _mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            quantization_type=QuantizationType.ROW,
            gate_w_scale=self.gate_proj_weight_scale,
            up_w_scale=self.up_proj_weight_scale,
            down_w_scale=self.down_proj_weight_scale,
        )


class Qwen3_5AttentionFP8(Qwen3_5Attention):
    """Full attention with FP8 QKV and output projections. 6.0% of the 27B.

    Only the two projections are quantized. The attention itself still runs in
    BF16: ``flash_attention`` reads the dense KV cache, which this does not
    touch (KV-cache quantization is a separate, orthogonal feature driven by
    ``kv_cache_dtype``).
    """

    _FP8_WEIGHTS = ("qkv_proj_weight", "o_proj_weight")

    def __init__(self, config, policy, layer_idx):
        super().__init__(config, policy, layer_idx)
        for name in self._FP8_WEIGHTS:
            _quantize_parameter(self, name)

    def _qkv_proj(self, hidden_states):
        return NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            quantization_type=QuantizationType.ROW,
            qkv_w_scale=self.qkv_proj_weight_scale,
        ).squeeze(0)

    def _out_proj(self, attn_output):
        # ``input_scales`` stays None: ROW derives the activation scale on
        # device. Passing one here would be silently ignored at best.
        return NF.o_proj(
            attn_output.unsqueeze(0),
            self.o_proj_weight,
            None,
            quantization_type=QuantizationType.ROW,
            weight_scales=self.o_proj_weight_scale,
        ).squeeze(0)


def resolve_module_classes(
    quantization: str | None, modules_to_not_convert: list[str] | None
) -> tuple[type, type]:
    """Return ``(AttentionCls, MLPCls)`` for the requested precision.

    ``modules_to_not_convert`` follows the convention already used by
    ``neuron_config`` (and by NxDI before it): a module stays BF16 if any
    entry matches its name. It is what makes "MLP only" reachable without a
    code change -- ``["self_attn"]`` keeps attention in BF16.
    """
    if quantization not in ("fp8",):
        return Qwen3_5Attention, Qwen3_5MLP

    skip = tuple(modules_to_not_convert or ())

    def keep_bf16(module_name: str) -> bool:
        return any(entry and entry in module_name for entry in skip)

    attention_cls = Qwen3_5Attention if keep_bf16("self_attn") else Qwen3_5AttentionFP8
    mlp_cls = Qwen3_5MLP if keep_bf16("mlp") else Qwen3_5MLPFP8

    if attention_cls is Qwen3_5Attention and mlp_cls is Qwen3_5MLP:
        logger.warning(
            "Qwen3.5: quantization='fp8' but modules_to_not_convert=%s excludes "
            "both mlp and self_attn, so every weight stays BF16.",
            list(skip),
        )
    return attention_cls, mlp_cls
