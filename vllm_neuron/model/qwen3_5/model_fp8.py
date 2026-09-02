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
import os

import torch
from nkilib.core.utils.common_types import QuantizationType
from torch import nn

import vllm_neuron.functional as NF
from vllm_neuron.utils.weight_loader import get_weight_loader, set_weight_loader

from .model import Qwen3_5Attention, Qwen3_5MLP
from .weight_loaders_fp8 import (
    FP8_CLAMP_MAX,
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


def _enable_legacy_fp8_cast() -> None:
    """Tell NKI to read ``torch.float8_e4m3fn`` as Trn2's legacy ``e4m3``.

    Without this, tracing dies at the first quantized kernel with::

        RuntimeError: float8_e4m3fn is not supported in nki.
        Set UNSAFE_FP8FNCAST to enable e4m3fn -> e4m3 casting.

    Torch has no legacy-e4m3 dtype, so an FP8 weight can only be *stored* as
    ``float8_e4m3fn``; Trn2's tensor engine only speaks legacy ``e4m3``. This
    flag is the documented bridge between the two
    (``nki/compiler/target.py:148-160``): on TRN2 it is the only setting under
    which ``float8_e4m3fn`` compiles at all.

    "Unsafe" refers to values in (240, 448] -- representable in OCP e4m3fn,
    but inf or NaN in legacy e4m3. That cannot happen here:
    :func:`~vllm_neuron.model.qwen3_5.weight_loaders_fp8.quantize_row_fp8`
    clamps every value to ``FP8_CLAMP_MAX``, which *is* 240 on Trn2, and the
    two encodings are bit-identical below it. The reinterpretation is exact.

    Set here rather than left to the operator because the two halves must
    agree: the flag without the clamp silently turns large weights into NaN,
    and the clamp without the flag simply fails to compile.
    """
    current = os.environ.get("UNSAFE_FP8FNCAST", "")
    if current.lower() not in ("1", "true", "yes", "on"):
        os.environ["UNSAFE_FP8FNCAST"] = "1"
        logger.info(
            "Qwen3.5 FP8: set UNSAFE_FP8FNCAST=1 so NKI reads float8_e4m3fn "
            "as Trn2's legacy e4m3. Exact here -- weights are clamped to %s.",
            FP8_CLAMP_MAX,
        )


def unsupported_platform_reason() -> str | None:
    """Why FP8 cannot run here, or ``None`` if it can.

    On Trn2 the prefill (CTE) MLP kernel requires the **activation** to
    already be FP8 whenever the weights are. ``mlp_cte_constants.py:145-157``
    sets the PE-transpose destination dtype to the quantized weight dtype as
    soon as the weights are quantized::

        if use_pe_xpose_flag and nisa.get_nc_version() >= nisa.nc_version.gen3:
            if mlpp_has_quantized_weights(mlp_params):
                return src_proj_quant_data_type

    while the tensor being transposed is the bf16 hidden state. gen3's
    transpose-mode ``nc_matmul`` requires the two to match, so tracing dies
    with::

        nc_matmul (transpose mode) dst dtype must match input dtype on gen3+,
        got dst=float8_e4m3 but input=bfloat16

    Measured on a trn2.48xlarge, one NF.mlp call at Qwen3.5-27B's TP=8 dims:

        mode              activation   decode (TKG)   prefill (CTE)
        bf16              bf16         OK             OK
        ROW fp8           bf16         OK             FAIL
        STATIC fp8        bf16         OK             FAIL
        ROW fp8           fp8          --             OK
        STATIC fp8        fp8          --             OK

    So this is not specific to ``ROW``: per-tensor ``STATIC`` fails the same
    way with a bf16 activation. What makes llama3's Trn2 static-FP8 path work
    is that its MLP owns the post-attention RMSNorm and calls
    ``NF.rmsnorm_quant`` to hand the kernel an **fp8** activation.

    Qwen3.5 cannot do that under ``ROW``: pre-quantizing the activation needs
    the scale used to be passed alongside it, and ROW has no input-scale
    argument -- it is defined to compute activation scales itself. Reaching
    FP8 prefill on Trn2 therefore means moving to ``STATIC`` with calibrated
    ``input_scale`` tensors, which no Qwen3.5 checkpoint ships (the official
    FP8 release declares ``activation_scheme="dynamic"``), plus a fused
    RMSNorm-quant on the prefill path. That is a separate piece of work.

    Raising here rather than letting it compile costs the operator ten
    minutes of graph extraction and a NKI assertion instead of a sentence.
    """
    from torch_neuronx.utils import get_platform_target

    try:
        target = get_platform_target()
    except RuntimeError:
        # No NRT and no override: a bare CPU run, where nothing compiles
        # anyway. Not this function's business to fail.
        return None

    if target.startswith("trn2"):
        return (
            "FP8 weights cannot serve prefill on trn2. With quantized weights "
            "the CTE MLP kernel transposes into an fp8 destination "
            "(mlp_cte_constants.py:145-157) while the hidden state is still "
            "bf16, and gen3 requires transpose src/dst dtypes to match: "
            "'nc_matmul (transpose mode) dst dtype must match input dtype on "
            "gen3+'. Decode compiles; prefill does not, in per-channel (ROW) "
            "and per-tensor (STATIC) modes alike. Serving FP8 here needs "
            "static activation scales from a calibrated checkpoint plus a "
            "fused rmsnorm-quant on the prefill path -- neither exists for "
            "Qwen3.5 today. Use quantization='bf16', or run on trn3."
        )
    return None


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

    _enable_legacy_fp8_cast()
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
