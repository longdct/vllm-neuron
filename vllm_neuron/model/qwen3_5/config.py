# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 / Qwen3.6 / Qwen3.8 Text Config
=======================================

One architecture, three product names. Qwen3.6-27B and Qwen3.8-27B ship
byte-identical ``text_config`` blocks and both declare
``architectures: ["Qwen3_5ForConditionalGeneration"]``, so this module is named
for the architecture rather than the release.

Text decoder only -- the vision tower is deliberately out of scope, and the
checkpoint's ``model.visual.*`` subtree is skipped at load time.

Shape of the model (27B):
  64 layers on a repeating 3:1 schedule -- 48 ``linear_attention`` (Gated
  DeltaNet) and 16 ``full_attention``. Dense SwiGLU on every layer.

Two things here differ from every other Qwen config in this repo and are easy to
get wrong; both are validated rather than assumed:
  - ``head_dim`` is 256, twice the usual, and exceeds the 128-element SBUF
    partition bound of the segmented-attention kernel.
  - RoPE is *partial* (``partial_rotary_factor`` 0.25 -> only the leading 64 of
    256 channels rotate) and *interleaved mRoPE*, so ``sum(mrope_section)`` must
    equal ``rotary_dim // 2``.
"""

import json
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

if TYPE_CHECKING:
    from .quantization import Qwen3_5QuantizationSpec

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

#: Head dimensions above this cannot be served by the segmented-attention
#: kernel: ``attention_segmented_cte.py`` raises because a head must fit in one
#: SBUF partition. ``attention_cte.py`` merely falls back to torch. Qwen3.5's
#: head_dim of 256 is over the line, so chunked prefill is unavailable and
#: ``max_model_len`` is capped at the single-shot limit until a tiled kernel
#: exists. See docs/model-dev/ and the plan's section 1.5.
MAX_KERNEL_HEAD_DIM = 128

#: Storage dtypes the paged GDN state may use. Both states share one value --
#: see ``mamba_state_dtype``. fp32 keeps the recurrent accumulator exact; bf16
#: halves the page and is what vLLM defaults to for this architecture.
SUPPORTED_STATE_DTYPES = frozenset({"float32", "bfloat16"})


def _from_hf_sub_config(cls, hf_sub_config, neuron_config=None):
    """Build a dataclass from an HF config sub-object.

    Mirrors ``qwen3_vl/config.py::_from_hf_sub_config``: filter to known fields,
    coerce the dtype string, attach the neuron_config. Kept as a local copy
    rather than an import so this module stays importable without the VL stack.
    """
    if isinstance(hf_sub_config, PretrainedConfig):
        config_dict = hf_sub_config.to_dict()
        if (
            hasattr(hf_sub_config, "torch_dtype")
            and hf_sub_config.torch_dtype is not None
        ):
            config_dict["torch_dtype"] = hf_sub_config.torch_dtype
    elif isinstance(hf_sub_config, dict):
        config_dict = hf_sub_config
    else:
        raise TypeError(f"Unsupported config type: {type(hf_sub_config)}")

    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in config_dict.items() if k in field_names}

    # HF config.json spells it "dtype"; the dataclass uses "torch_dtype".
    if (
        "torch_dtype" not in filtered
        and "dtype" in config_dict
        and "torch_dtype" in field_names
    ):
        filtered["torch_dtype"] = config_dict["dtype"]

    if "torch_dtype" in filtered and isinstance(filtered["torch_dtype"], str):
        filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

    if neuron_config is not None:
        filtered["neuron_config"] = neuron_config

    return cls(**filtered)


@dataclass
class Qwen3_5TextConfig:
    """Text decoder config, read from ``hf_config.text_config``.

    Defaults are the Qwen3.8-27B / Qwen3.6-27B values.
    """

    # -- Backbone
    vocab_size: int = 248320
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # -- Full attention layers
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    attention_bias: bool = False
    #: Qwen3.5 emits ``num_heads * head_dim * 2`` from q_proj: each head is
    #: [query | gate] side by side, and the gate multiplies the attention
    #: output before o_proj. The interleaving is *per head*, so a flat column
    #: split shards it wrongly -- see weight_loaders.py.
    attn_output_gate: bool = True
    partial_rotary_factor: float = 0.25
    #: ``mrope_section`` is deliberately absent from this default. The real
    #: checkpoints always supply it, and when they do it is validated against
    #: ``rotary_dim`` below. When it is absent -- tiny fixtures, hand-built
    #: configs -- it is derived as a balanced three-way split of the half-rotary
    #: band, which reproduces the shipped ``[11, 11, 10]`` for head_dim 256
    #: exactly. Hard-coding the 27B value here instead would make every config
    #: with a smaller head_dim fail validation.
    rope_parameters: dict = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "mrope_interleaved": True,
            "partial_rotary_factor": 0.25,
        }
    )

    # -- Gated DeltaNet (linear attention) layers
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    #: Storage dtype for *both* paged GDN states -- the conv window and the
    #: recurrent state alike. They are deliberately one knob and not two: the
    #: runner carves both from a single raw page as strided views over one
    #: storage (neuron_model_runner.py:8546) and the model mutates both in
    #: place, and aot_autograd refuses input mutations on views of one storage
    #: with differing dtypes. Internal arithmetic is fp32 whatever this says --
    #: every consumer upcasts the state on entry -- so this chooses only the
    #: precision the state is *stored* at between steps. "bfloat16" is what
    #: vLLM itself defaults to for this architecture (mamba_ssm_cache_dtype
    #: "auto" makes the recurrent state follow the model dtype) and halves
    #: state memory; "float32" keeps the accumulator exact across long
    #: sequences and is the conservative default here.
    mamba_state_dtype: str = "float32"
    output_gate_type: str = "swish"

    # -- Layer schedule. Either given explicitly, or derived from the interval.
    layer_types: list[str] | None = None
    full_attention_interval: int = 4

    # -- Not implemented. The released checkpoint carries no MTP weights even
    #    though the config advertises a layer, so this must stay 0/None in
    #    practice; a non-zero value with real weights is rejected in the factory.
    mtp_num_hidden_layers: int = 0

    neuron_config: NeuronConfig | None = None

    #: What the *checkpoint* says about its own weights, parsed from the
    #: top-level ``quantization_config``. ``None`` for an unquantized
    #: checkpoint. Set by the factory rather than by ``from_hf_config``,
    #: because ``quantization_config`` sits on the multimodal wrapper config
    #: and never reaches ``text_config``.
    quant_spec: "Qwen3_5QuantizationSpec | None" = None

    # ------------------------------------------------------------------
    # Normalization and validation
    # ------------------------------------------------------------------

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self._normalize_layer_types()
        self._validate_rope()
        self._validate_gdn()

    def _normalize_layer_types(self) -> None:
        """Resolve ``layer_types`` and refuse to guess.

        A config that quietly defaults is a config that quietly narrows the
        test gate: the DeepSeek-V4 bring-up shipped a synthetic config whose
        missing ``mlp_layer_types`` silently selected routed-MoE, so the device
        gate never exercised the hash router at all
        (docs/model-dev/deepseek-v4-real-weight-validation.md). So derive from
        the interval when absent, but validate hard either way.
        """
        if self.layer_types is None:
            if not self.full_attention_interval or self.full_attention_interval < 1:
                raise ValueError(
                    "layer_types is absent and full_attention_interval="
                    f"{self.full_attention_interval!r} cannot derive it. Qwen3.5 "
                    "places a full_attention layer at every Nth position."
                )
            # HF: layer i is full attention iff (i + 1) % interval == 0.
            self.layer_types = [
                FULL_ATTENTION
                if (i + 1) % self.full_attention_interval == 0
                else LINEAR_ATTENTION
                for i in range(self.num_hidden_layers)
            ]

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers={self.num_hidden_layers}."
            )

        unknown = sorted(set(self.layer_types) - {LINEAR_ATTENTION, FULL_ATTENTION})
        if unknown:
            raise ValueError(
                f"Unsupported layer_types entries {unknown}. This implementation "
                f"handles only {LINEAR_ATTENTION!r} and {FULL_ATTENTION!r}."
            )

        if not self.full_layer_indices:
            raise ValueError(
                "layer_types contains no full_attention layer; a purely linear "
                "stack is not a Qwen3.5-family model."
            )
        if not self.linear_layer_indices:
            raise ValueError(
                "layer_types contains no linear_attention layer. Use the plain "
                "Qwen3 implementation (vllm_neuron/model/qwen3) for a dense model."
            )

    def _validate_rope(self) -> None:
        rotary_dim = self.rotary_dim
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError(
                f"partial_rotary_factor={self.partial_rotary_factor} gives "
                f"rotary_dim={rotary_dim}, which must be positive and even."
            )
        if rotary_dim > self.head_dim:
            raise ValueError(
                f"rotary_dim={rotary_dim} exceeds head_dim={self.head_dim}."
            )

        # Interleaved mRoPE splits the half-rotary frequency band three ways; a
        # mismatch silently mis-assigns frequencies to the T/H/W axes. Validate
        # only what the checkpoint actually stated -- a derived section is
        # correct by construction.
        section = self.rope_parameters.get("mrope_section")
        if section is not None:
            if len(section) != 3:
                raise ValueError(
                    f"mrope_section={section} must have exactly 3 entries "
                    "(temporal, height, width)."
                )
            if sum(section) != rotary_dim // 2:
                raise ValueError(
                    f"mrope_section={section} sums to {sum(section)} but "
                    f"rotary_dim // 2 = {rotary_dim // 2}."
                )

    def _validate_gdn(self) -> None:
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError(
                f"linear_num_value_heads={self.linear_num_value_heads} must be a "
                f"multiple of linear_num_key_heads={self.linear_num_key_heads}; "
                "each key head is repeat_interleaved across its value heads."
            )
        if self.linear_conv_kernel_dim < 1:
            raise ValueError("linear_conv_kernel_dim must be >= 1.")
        if self.mamba_state_dtype not in SUPPORTED_STATE_DTYPES:
            raise ValueError(
                f"mamba_state_dtype={self.mamba_state_dtype!r} is not supported; "
                f"choose one of {sorted(SUPPORTED_STATE_DTYPES)}. Both GDN states "
                "are stored at this dtype, so it must name a real torch dtype "
                "the depthwise conv and the scan both accept."
            )

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    @property
    def linear_layer_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == LINEAR_ATTENTION]

    @property
    def full_layer_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == FULL_ATTENTION]

    @property
    def rotary_dim(self) -> int:
        """Number of leading head channels that rotate. 64 of 256 by default."""
        factor = self.rope_parameters.get(
            "partial_rotary_factor", self.partial_rotary_factor
        )
        return int(self.head_dim * factor)

    @property
    def rope_theta(self) -> float:
        return float(self.rope_parameters.get("rope_theta", 10000000.0))

    @property
    def mrope_section(self) -> list[int]:
        """Per-axis widths of the interleaved mRoPE frequency band.

        Uses the checkpoint's value when present. Otherwise splits
        ``rotary_dim // 2`` as evenly as possible with the remainder on the
        leading axes -- which yields ``[11, 11, 10]`` for the shipped head_dim
        of 256, matching HuggingFace.
        """
        section = self.rope_parameters.get("mrope_section")
        if section is not None:
            return list(section)

        half = self.rotary_dim // 2
        base, remainder = divmod(half, 3)
        return [base + (1 if i < remainder else 0) for i in range(3)]

    @property
    def mrope_interleaved(self) -> bool:
        return bool(self.rope_parameters.get("mrope_interleaved", True))

    @property
    def uses_mrope(self) -> bool:
        """Mirror of vLLM's own mRoPE determination for this checkpoint.

        ``vllm/transformers_utils/config.py::_uses_mrope`` is exactly
        ``"mrope_section" in rope_parameters`` on the *raw* HF config, so this
        deliberately checks the raw dict rather than the derived
        :attr:`mrope_section` -- otherwise the model would claim mRoPE on a
        fixture whose config.json omits it, while the model runner (following
        vLLM) never passes ``rotary_position_ids``, and the two would disagree.

        True for every shipped Qwen3.5-family checkpoint, including text-only
        runs, which makes ``SupportsMRoPE.get_mrope_input_positions`` mandatory.
        """
        return "mrope_section" in self.rope_parameters

    @property
    def num_v_per_k(self) -> int:
        """Value heads sharing one key head (3 for the 27B)."""
        return self.linear_num_value_heads // self.linear_num_key_heads

    @property
    def key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        """Width of the depthwise causal conv: [q | k | v] stacked."""
        return 2 * self.key_dim + self.value_dim

    @property
    def state_dtype(self) -> torch.dtype:
        """Storage dtype shared by both paged GDN states."""
        return getattr(torch, self.mamba_state_dtype)

    @property
    def needs_single_shot_prefill(self) -> bool:
        """True when head_dim rules out the segmented-attention kernel."""
        return self.head_dim > MAX_KERNEL_HEAD_DIM

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_hf_config(cls, hf_text_config, neuron_config: NeuronConfig = None):
        return _from_hf_sub_config(cls, hf_text_config, neuron_config)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig | dict | str, neuron_config: NeuronConfig = None
    ) -> "Qwen3_5TextConfig":
        """Accept the top-level (multimodal) config and pull out ``text_config``.

        The released checkpoints are ``Qwen3_5ForConditionalGeneration`` with the
        decoder nested under ``text_config``; a bare text config is also accepted
        so tiny fixtures can skip the wrapper.
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                hf_config = json.load(f)

        text_config = None
        if isinstance(hf_config, PretrainedConfig):
            text_config = getattr(hf_config, "text_config", None)
        elif isinstance(hf_config, dict):
            text_config = hf_config.get("text_config")

        return cls.from_hf_config(
            text_config if text_config is not None else hf_config, neuron_config
        )
