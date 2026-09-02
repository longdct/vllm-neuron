# SPDX-License-Identifier: Apache-2.0
"""Reading a Qwen3.5-family checkpoint's ``quantization_config``.

This module only *describes* what a checkpoint contains. Deciding whether the
running build can serve that description is the factory's job
(:meth:`Qwen3_5ForCausalLM._validate_config`), which keeps "what the
checkpoint is" separate from "what we have implemented today".

Why this exists at all: without it, a quantized checkpoint does not fail, it
*loads wrong*. ``text_weight_mappings`` maps only ``.weight`` keys, so every
``.weight_scale_inv`` lands in ``unexpected_keys`` and is discarded; then
``checkpoints.py`` sees ``float8_e4m3fn != bfloat16`` and simply casts. The
result is raw FP8 bytes reinterpreted as BF16 magnitudes with the scales
thrown away -- coherent-looking garbage, announced only by a warning flood.

The formats named below are the ones a Qwen3.5 checkpoint actually ships;
they are not hypothetical:

``Qwen/Qwen3.8-27B-FP8`` (the official quantized 27B) declares::

    {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
     "weight_block_size": [128, 128], "modules_to_not_convert": [...]}

``activation_scheme="dynamic"`` is the load-bearing detail: the checkpoint
carries **no** activation scales, so the static-per-tensor FP8 path llama3
uses (``QuantizationType.STATIC``, which requires ``input_scale`` tensors) can
never serve it. That is why the Neuron side of this work targets
``QuantizationType.ROW``, whose activation scales are computed at runtime.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: The block shape this module understands for ``quant_method="fp8"``.
#: Both the official Qwen FP8 release and DeepSeek-V3-style checkpoints use
#: 128x128 tiles with one scale each.
_FP8_BLOCK = 128


class WeightFormat(Enum):
    """How the *checkpoint* stores its linear weights."""

    #: Unquantized. Weights are read at their declared dtype (bf16).
    BF16 = "bf16"

    #: ``float8_e4m3fn`` values plus one fp32/bf16 ``weight_scale_inv`` per
    #: 128x128 tile. Dequantized with
    #: ``tools.deepseek_v4.quant_formats.dequantize_fp8_blockwise``.
    FP8_BLOCK128 = "fp8_block128"


@dataclass(frozen=True)
class Qwen3_5QuantizationSpec:
    """What a checkpoint's ``quantization_config`` says about its weights."""

    weight_format: WeightFormat

    #: Module-name substrings the checkpoint left unquantized. The official
    #: 27B FP8 release lists ~512 of these (the whole vision tower, every
    #: norm, and the MTP head). A name matching any entry must be read as
    #: bf16 even when the rest of the checkpoint is FP8.
    modules_to_not_convert: tuple[str, ...] = ()

    def is_quantized(self) -> bool:
        return self.weight_format is not WeightFormat.BF16

    def keeps_full_precision(self, checkpoint_key: str) -> bool:
        """Whether ``checkpoint_key`` was left unquantized by the producer.

        Matching is by substring rather than equality because the entries are
        module paths (``model.visual.blocks.0.attn.qkv``) while the keys we
        test are parameter paths (``...attn.qkv.weight``).
        """
        return any(m and m in checkpoint_key for m in self.modules_to_not_convert)


def read_quantization_spec(hf_config: Any) -> Qwen3_5QuantizationSpec | None:
    """Pull ``quantization_config`` off a HuggingFace config and parse it.

    Accepts a ``PretrainedConfig`` or a plain dict, since the Qwen3.5 factory
    is reached with both (real checkpoints go through transformers; tiny
    fixtures pass dicts). The ``to_dict()`` fallback matters for ModelOpt- and
    compressor-injected configs, where the key exists in ``__dict__`` but was
    never declared on the pretrained class, so ``getattr`` misses it.
    """
    if hf_config is None:
        return None

    quantization_config = None
    if isinstance(hf_config, dict):
        quantization_config = hf_config.get("quantization_config")
    else:
        quantization_config = getattr(hf_config, "quantization_config", None)
        if quantization_config is None and hasattr(hf_config, "to_dict"):
            quantization_config = hf_config.to_dict().get("quantization_config")

    return parse_quantization_config(quantization_config)


def parse_quantization_config(
    quantization_config: dict[str, Any] | None,
) -> Qwen3_5QuantizationSpec | None:
    """Describe a checkpoint's quantization, or raise if it is unreadable.

    Returns ``None`` for an unquantized checkpoint, so callers can treat
    "absent" and "bf16" identically.

    Raises:
        ValueError: the checkpoint is quantized in a way this model cannot
            read. Raising is the point -- the alternative is silent
            corruption, not a graceful bf16 fallback.
    """
    if not quantization_config:
        return None
    if not isinstance(quantization_config, dict):
        raise ValueError(
            "Expected quantization_config to be a dict, got "
            f"{type(quantization_config).__name__}."
        )

    quant_method = str(quantization_config.get("quant_method", "")).lower()
    exclude = _read_modules_to_not_convert(quantization_config)

    if quant_method == "fp8":
        return _parse_blockwise_fp8(quantization_config, exclude)

    if quant_method == "compressed-tensors":
        return _parse_compressed_tensors(quantization_config, exclude)

    if quant_method == "modelopt":
        # Parsable, but useless here: ModelOpt FP8 is per-tensor with static
        # activation scales, which only the STATIC kernel path consumes.
        # Qwen3.5 has no STATIC path (see the module docstring), so accepting
        # this would mean loading the weights and ignoring `input_scale`.
        raise ValueError(
            "Qwen3.5 cannot serve ModelOpt (per-tensor static FP8) "
            "checkpoints: its FP8 path uses per-channel weight scales with "
            "runtime activation scales, and has nowhere to apply the "
            "checkpoint's static input_scale tensors. Use the BF16 "
            "checkpoint, or the official blockwise-FP8 release."
        )

    raise ValueError(
        f"Unsupported quantization_config.quant_method={quant_method!r} for "
        "the Qwen3.5 family. Supported: 'fp8' (128x128 blockwise, e.g. "
        "Qwen/Qwen3.8-27B-FP8), 'compressed-tensors' (KV-cache scales only), "
        "or an unquantized BF16 checkpoint."
    )


def _read_modules_to_not_convert(
    quantization_config: dict[str, Any],
) -> tuple[str, ...]:
    raw = quantization_config.get("modules_to_not_convert") or ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "quantization_config.modules_to_not_convert must be a list, got "
            f"{type(raw).__name__}."
        )
    return tuple(str(m) for m in raw)


def _parse_blockwise_fp8(
    quantization_config: dict[str, Any], exclude: tuple[str, ...]
) -> Qwen3_5QuantizationSpec:
    """Parse ``quant_method="fp8"`` -- the official Qwen3.8-27B-FP8 shape."""
    fmt = str(quantization_config.get("fmt", "e4m3")).lower()
    if fmt not in ("e4m3", "float8_e4m3fn"):
        # e5m2 has 2 mantissa bits against e4m3's 3; nothing in this stack
        # reads it, and quietly treating it as e4m3 would misread every value.
        raise ValueError(
            f"Qwen3.5 FP8 checkpoints must be e4m3, got fmt={fmt!r}."
        )

    block = quantization_config.get("weight_block_size")
    if block is None:
        raise ValueError(
            "quantization_config declares quant_method='fp8' but no "
            "weight_block_size. Per-tensor FP8 is not supported here; see "
            "the ModelOpt note in vllm_neuron/model/qwen3_5/quantization.py."
        )
    if list(block) != [_FP8_BLOCK, _FP8_BLOCK]:
        raise ValueError(
            f"Qwen3.5 supports {_FP8_BLOCK}x{_FP8_BLOCK} FP8 weight blocks, "
            f"got weight_block_size={list(block)!r}."
        )

    activation_scheme = str(
        quantization_config.get("activation_scheme", "dynamic")
    ).lower()
    if activation_scheme != "dynamic":
        # A "static" checkpoint carries activation scales we would ignore,
        # which changes the arithmetic the producer calibrated for.
        raise ValueError(
            "Qwen3.5 supports activation_scheme='dynamic' only, got "
            f"{activation_scheme!r}: the Neuron ROW path computes activation "
            "scales at runtime and cannot honour calibrated static scales."
        )

    return Qwen3_5QuantizationSpec(
        weight_format=WeightFormat.FP8_BLOCK128,
        modules_to_not_convert=exclude,
    )


def _parse_compressed_tensors(
    quantization_config: dict[str, Any], exclude: tuple[str, ...]
) -> Qwen3_5QuantizationSpec | None:
    """Accept KV-cache-only compressed-tensors; reject quantized weights.

    Mirrors the platform-level rule in
    ``vllm_neuron/vllm/platform.py::_validate_quantization_config``: a
    compressed-tensors config that only carries ``q_scale``/``k_scale``/
    ``v_scale`` is a KV-cache scheme, handled elsewhere, and leaves the
    weights in bf16. Anything that quantizes weights is not readable here.
    """
    groups = quantization_config.get("config_groups") or {}
    if not isinstance(groups, dict):
        raise ValueError(
            "quantization_config.config_groups must be a dict, got "
            f"{type(groups).__name__}."
        )

    for name, group in groups.items():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        if weights:
            num_bits = weights.get("num_bits") if isinstance(weights, dict) else None
            hint = ""
            if num_bits == 4:
                hint = (
                    " 4-bit weights (NVFP4/MXFP4) have no Neuron path on this "
                    "platform: Trainium2 offers no FP4 tensor-engine format, "
                    "and nkilib cannot quantize to FP4."
                )
            raise ValueError(
                "Qwen3.5 cannot serve compressed-tensors checkpoints with "
                f"quantized weights (config_groups[{name!r}].weights is set)."
                f"{hint} Only KV-cache scales are supported from this format."
            )

    del exclude  # weights are bf16 on this path, so the exclude list is moot
    return None
