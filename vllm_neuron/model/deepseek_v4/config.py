# SPDX-License-Identifier: Apache-2.0
"""Normalization and validation of DeepSeek-V4 configuration.

DeepSeek-V4 describes its per-layer structure in two different shapes depending
on where the config came from:

* **Raw checkpoint form** -- ``config.json`` carries ``compress_ratios``, one
  entry per decoder layer, valued 0 (sliding window), 4, or 128.
* **Normalized Transformers form** -- depending on the pinned Transformers
  release, the loaded config may instead expose ``layer_types`` alongside
  ``compress_rates``.

Everything downstream (per-layer component selection, heterogeneous cache
registration, the dense-CSA bound) needs *one* representation, so both forms are
folded into :class:`LayerSpec` here and nowhere else. A third, unrecognized form
must fail loudly rather than be guessed at: silently mis-reading which layers are
c4 and which are c128 would produce a model that loads, runs, and is wrong.

This module is deliberately free of ``torch`` and ``vllm`` imports. The mapping
is pure data, and keeping it importable on a bare interpreter is what allows it
to be tested before a Neuron or Linux environment is available -- the same
convention already used by ``vllm_neuron/vllm/patches/guards.py`` and
``vllm_neuron/vllm/scheduler_selection.py``.

**Verified against Transformers 5.15.0** (``DeepseekV4Config``), which settled
three things this module previously guessed at, all of them wrongly:

* The ``layer_types`` vocabulary is ``sliding_attention`` /
  ``compressed_sparse_attention`` / ``heavily_compressed_attention`` -- not the
  ``compressed_attention`` spelling originally assumed. The strict-raise design
  worked as intended: the wrong guess would have failed loudly at startup rather
  than mis-typing layers.
* ``compress_rates`` is a **dict keyed by layer type**, not a per-layer list.
  The two have different lengths and different meanings, and reading one as the
  other produces silently wrong ratios rather than an error.
* Per-layer MLP structure comes from ``mlp_layer_types``. ``num_hash_layers``
  exists only as a legacy kwarg that upstream consumes during ``__post_init__``,
  so it never survives on a loaded config object.

The same is true of ``compress_ratios``: upstream pops it, folds it into
``layer_types``, and does not retain it. Both legacy forms therefore appear only
on raw checkpoint JSON, which is exactly why both paths are still supported here.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "SUPPORTED_COMPRESS_RATIOS",
    "AttentionKind",
    "DeepseekV4ConfigError",
    "LayerSpec",
    "MLPKind",
    "NormalizedDeepseekV4Config",
    "normalize_config",
    "normalize_layer_specs",
]


class DeepseekV4ConfigError(ValueError):
    """Raised when a DeepSeek-V4 config is unrecognized or unsupported.

    Deliberately distinct from a bare ``ValueError`` so callers can tell "this
    checkpoint is not something we support" apart from an ordinary bad argument.
    """


class AttentionKind(str, Enum):
    """Per-layer attention structure."""

    #: ``compress_ratio == 0``: local sliding-window attention, no compression.
    SLIDING_WINDOW = "sliding_window"
    #: ``compress_ratio in (4, 128)``: hierarchical compressed attention.
    COMPRESSED = "compressed"


class MLPKind(str, Enum):
    """Per-layer feed-forward structure."""

    #: Layers ``< num_hash_layers``: expert *selection* comes from ``tid2eid``,
    #: while the learned gate still supplies the weights.
    HASH_MOE = "hash_moe"
    #: Ordinary routed MoE: ``sqrtsoftplus`` scoring with ``noaux_tc`` selection.
    ROUTED_MOE = "routed_moe"


#: Compression ratios the implementation knows how to lay out in the KV cache.
#: 0 means "no compression, sliding window". Anything else is a checkpoint
#: variant this plugin has not been built or tested for.
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)

#: ``layer_types`` spellings mapped to the compression ratio each denotes.
#:
#: Verified against ``transformers.models.deepseek_v4.configuration_deepseek_v4``
#: at Transformers 5.15.0, which defines the vocabulary as
#: ``DEEPSEEK_V4_LAYER_TYPES`` and the inverse mapping as
#: ``_COMPRESS_RATIO_TO_LAYER_TYPE = {0: "sliding_attention",
#: 4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}``.
#:
#: The ratio for a *compressed* type is only a default: upstream lets
#: ``compress_rates`` override the value per type, so this table supplies the
#: fallback and :func:`_ratios_from_layer_types` prefers the config's own dict.
#: ``sliding_attention`` is the exception -- it means "no compression", and no
#: ``compress_rates`` entry exists for it.
_LAYER_TYPE_TO_RATIO: Mapping[str, int] = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}

#: The one ``layer_types`` value that means "uncompressed sliding window".
_SLIDING_LAYER_TYPE = "sliding_attention"

#: ``mlp_layer_types`` spellings, from ``DEEPSEEK_V4_MLP_LAYER_TYPES``.
_MLP_TYPE_KINDS: Mapping[str, "MLPKind"] = {
    "hash_moe": MLPKind.HASH_MOE,
    "moe": MLPKind.ROUTED_MOE,
}

#: The only scoring/selection combination the MoE implementation targets.
_SUPPORTED_SCORING_FUNC = "sqrtsoftplus"
_SUPPORTED_TOPK_METHOD = "noaux_tc"

_MISSING = object()


@dataclass(frozen=True)
class LayerSpec:
    """The structural identity of one decoder layer.

    This is what per-layer component selection dispatches on (plan P3, mirroring
    llama3's ``resolve_attention_mlp_classes``), and what the heterogeneous cache
    registration groups by.
    """

    index: int
    attention: AttentionKind
    #: 0 for sliding-window layers, otherwise the compression ratio (4 or 128).
    compress_ratio: int
    mlp: MLPKind

    @property
    def is_compressed(self) -> bool:
        return self.attention is AttentionKind.COMPRESSED

    @property
    def cache_group_key(self) -> tuple[str, int]:
        """Layers sharing this key share a KV-cache layout.

        The key is ``(attention kind, compress_ratio)`` rather than the ratio
        alone so that the sliding-window group stays distinct by construction
        instead of by the convention that its ratio happens to be 0.
        """
        return (self.attention.value, self.compress_ratio)


@dataclass(frozen=True)
class NormalizedDeepseekV4Config:
    """Validated, form-independent view of a DeepSeek-V4 config."""

    layers: tuple[LayerSpec, ...]
    num_hidden_layers: int
    num_hash_layers: int
    sliding_window: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    num_nextn_predict_layers: int

    def distinct_cache_groups(self) -> tuple[tuple[str, int], ...]:
        """Cache-layout groups present, in first-appearance order.

        Feeds the heterogeneous cache registration (plan P1.1): one entry per
        distinct layout that must be allocated.
        """
        seen: dict[tuple[str, int], None] = {}
        for layer in self.layers:
            seen.setdefault(layer.cache_group_key, None)
        return tuple(seen)

    def layer_indices_for(self, key: tuple[str, int]) -> tuple[int, ...]:
        """Indices of the layers bound to one cache-layout group."""
        return tuple(l.index for l in self.layers if l.cache_group_key == key)


def _get(config: Any, name: str, default: Any = _MISSING) -> Any:
    """Read *name* from a Transformers config object or a raw dict.

    Both shapes reach this module -- an object from ``AutoConfig``, or the parsed
    ``config.json`` -- and neither is worth converting to the other just to read
    a dozen fields.
    """
    if isinstance(config, Mapping):
        value = config.get(name, _MISSING)
    else:
        value = getattr(config, name, _MISSING)
    if value is _MISSING or value is None:
        if default is _MISSING:
            raise DeepseekV4ConfigError(
                f"DeepSeek-V4 config is missing required field {name!r}"
            )
        return default
    return value


def _require_positive_int(config: Any, name: str, default: Any = _MISSING) -> int:
    value = _get(config, name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DeepseekV4ConfigError(
            f"DeepSeek-V4 config field {name!r} must be a positive integer, got {value!r}"
        )
    return value


def _as_int_list(value: Any, name: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise DeepseekV4ConfigError(
            f"DeepSeek-V4 config field {name!r} must be a list of integers, got {value!r}"
        )
    out: list[int] = []
    for i, entry in enumerate(value):
        if not isinstance(entry, int) or isinstance(entry, bool):
            raise DeepseekV4ConfigError(
                f"DeepSeek-V4 config field {name!r}[{i}] must be an integer, got {entry!r}"
            )
        out.append(entry)
    return out


def _validate_ratios(ratios: list[int], num_hidden_layers: int, source: str) -> None:
    if len(ratios) != num_hidden_layers:
        raise DeepseekV4ConfigError(
            f"{source} has {len(ratios)} entries but num_hidden_layers is "
            f"{num_hidden_layers}; every decoder layer needs exactly one entry"
        )
    unsupported = sorted({r for r in ratios if r not in SUPPORTED_COMPRESS_RATIOS})
    if unsupported:
        raise DeepseekV4ConfigError(
            f"{source} contains unsupported compression ratio(s) {unsupported}; "
            f"this implementation supports {list(SUPPORTED_COMPRESS_RATIOS)}"
        )


def _ratios_from_layer_types(
    layer_types: list[Any], compress_rates: Any, num_hidden_layers: int
) -> list[int]:
    """Fold the normalized form into plain per-layer ratios.

    Note the shapes, which are **not** symmetric and were the easiest thing to
    get wrong here: ``layer_types`` is one entry *per layer*, while
    ``compress_rates`` is a dict keyed *by layer type* -- upstream's default is
    ``{"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}``.
    So the per-layer ratio is a lookup through the type name, not a positional
    zip. Treating ``compress_rates`` as a per-layer list yields ratios that are
    silently wrong for every layer past the second.

    ``sliding_attention`` carries no ``compress_rates`` entry -- it means "no
    compression" -- and is resolved to 0 without consulting the dict.
    """
    if len(layer_types) != num_hidden_layers:
        raise DeepseekV4ConfigError(
            f"'layer_types' has {len(layer_types)} entries but num_hidden_layers "
            f"is {num_hidden_layers}"
        )

    if compress_rates is None:
        rates: Mapping[str, Any] = {}
    elif isinstance(compress_rates, Mapping):
        rates = compress_rates
    else:
        raise DeepseekV4ConfigError(
            f"'compress_rates' must be a mapping from layer type to compression "
            f"ratio (e.g. {{'compressed_sparse_attention': 4}}), got "
            f"{type(compress_rates).__name__}. A per-layer list is the legacy "
            f"'compress_ratios' field, which is a different thing"
        )

    ratios: list[int] = []
    for index, raw_type in enumerate(layer_types):
        if not isinstance(raw_type, str):
            raise DeepseekV4ConfigError(
                f"'layer_types'[{index}] must be a string, got {raw_type!r}"
            )
        if raw_type not in _LAYER_TYPE_TO_RATIO:
            raise DeepseekV4ConfigError(
                f"'layer_types'[{index}] is {raw_type!r}, which this implementation "
                f"does not recognize; known values are "
                f"{sorted(_LAYER_TYPE_TO_RATIO)}. Refusing to guess the layer's "
                f"structure -- see vllm_neuron/model/deepseek_v4/config.py"
            )

        if raw_type == _SLIDING_LAYER_TYPE:
            # No compression, and upstream carries no rate entry for it. A config
            # that supplies one is describing something this code does not model.
            if raw_type in rates and rates[raw_type] != 0:
                raise DeepseekV4ConfigError(
                    f"'compress_rates' gives {raw_type!r} a ratio of "
                    f"{rates[raw_type]!r}, but that layer type means "
                    f"'uncompressed sliding window'"
                )
            ratios.append(0)
            continue

        ratio = rates.get(raw_type, _LAYER_TYPE_TO_RATIO[raw_type])
        if not isinstance(ratio, int) or isinstance(ratio, bool):
            raise DeepseekV4ConfigError(
                f"'compress_rates'[{raw_type!r}] must be an integer, got {ratio!r}"
            )
        ratios.append(ratio)

    _validate_ratios(ratios, num_hidden_layers, "'layer_types' + 'compress_rates'")
    return ratios


def _extract_compress_ratios(config: Any, num_hidden_layers: int) -> list[int]:
    """Resolve whichever per-layer form the config carries.

    Both forms present is fine *if* they agree -- that is a config that has been
    through normalization while retaining its original field. Both present and
    disagreeing is not recoverable and is not a case to pick a winner in.
    """
    raw_ratios = _get(config, "compress_ratios", None)
    layer_types = _get(config, "layer_types", None)
    compress_rates = _get(config, "compress_rates", None)

    from_raw: list[int] | None = None
    if raw_ratios is not None:
        from_raw = _as_int_list(raw_ratios, "compress_ratios")
        _validate_ratios(from_raw, num_hidden_layers, "'compress_ratios'")

    from_normalized: list[int] | None = None
    if layer_types is not None:
        # ``compress_rates`` may legitimately be absent: upstream defaults it, and
        # the per-type fallback in ``_LAYER_TYPE_TO_RATIO`` matches that default.
        # ``layer_types`` is the field that cannot be defaulted, because it is the
        # only per-layer signal.
        from_normalized = _ratios_from_layer_types(
            list(layer_types), compress_rates, num_hidden_layers
        )
    elif compress_rates is not None and raw_ratios is None:
        raise DeepseekV4ConfigError(
            "DeepSeek-V4 config provides 'compress_rates' without 'layer_types'; "
            "'compress_rates' is keyed by layer type and says nothing about which "
            "layer is which, so per-layer structure cannot be determined"
        )

    if from_raw is None and from_normalized is None:
        raise DeepseekV4ConfigError(
            "DeepSeek-V4 config carries neither 'compress_ratios' nor "
            "'layer_types' + 'compress_rates'; per-layer structure cannot be "
            "determined"
        )
    if from_raw is not None and from_normalized is not None:
        if from_raw != from_normalized:
            mismatches = [
                i for i, (a, b) in enumerate(zip(from_raw, from_normalized)) if a != b
            ]
            raise DeepseekV4ConfigError(
                f"DeepSeek-V4 config carries both 'compress_ratios' and "
                f"'layer_types'/'compress_rates', and they disagree at layer(s) "
                f"{mismatches[:8]}; refusing to choose between them"
            )
        return from_raw
    return from_raw if from_raw is not None else from_normalized  # type: ignore[return-value]


def normalize_layer_specs(config: Any) -> tuple[LayerSpec, ...]:
    """Per-layer structure, independent of which form *config* used."""
    num_hidden_layers = _require_positive_int(config, "num_hidden_layers")
    ratios = _extract_compress_ratios(config, num_hidden_layers)

    mlp_kinds = _extract_mlp_kinds(config, num_hidden_layers)

    return tuple(
        LayerSpec(
            index=index,
            attention=(
                AttentionKind.SLIDING_WINDOW
                if ratio == 0
                else AttentionKind.COMPRESSED
            ),
            compress_ratio=ratio,
            mlp=mlp_kinds[index],
        )
        for index, ratio in enumerate(ratios)
    )


def _extract_mlp_kinds(config: Any, num_hidden_layers: int) -> list[MLPKind]:
    """Per-layer feed-forward structure.

    Two forms, mirroring the attention side:

    * ``mlp_layer_types`` -- one ``"hash_moe"`` / ``"moe"`` entry per layer. This
      is what a config loaded through ``DeepseekV4Config`` exposes, and it is
      authoritative.
    * ``num_hash_layers`` -- the legacy scalar, meaning "the first *n* layers are
      hash-MoE". Upstream consumes this kwarg during ``__post_init__`` and folds
      it into ``mlp_layer_types``, so it survives only on raw checkpoint JSON.

    Attention and MLP structure are **independent**: a layer's compression ratio
    does not determine whether it is hash- or routed-MoE. Both lists are read
    separately rather than one being derived from the other.
    """
    mlp_layer_types = _get(config, "mlp_layer_types", None)
    num_hash_layers = _get(config, "num_hash_layers", None)

    from_types: list[MLPKind] | None = None
    if mlp_layer_types is not None:
        entries = list(mlp_layer_types)
        if len(entries) != num_hidden_layers:
            raise DeepseekV4ConfigError(
                f"'mlp_layer_types' has {len(entries)} entries but "
                f"num_hidden_layers is {num_hidden_layers}"
            )
        from_types = []
        for index, raw in enumerate(entries):
            kind = _MLP_TYPE_KINDS.get(raw) if isinstance(raw, str) else None
            if kind is None:
                raise DeepseekV4ConfigError(
                    f"'mlp_layer_types'[{index}] is {raw!r}, which this "
                    f"implementation does not recognize; known values are "
                    f"{sorted(_MLP_TYPE_KINDS)}"
                )
            from_types.append(kind)

    from_count: list[MLPKind] | None = None
    if num_hash_layers is not None:
        if (
            not isinstance(num_hash_layers, int)
            or isinstance(num_hash_layers, bool)
            or num_hash_layers < 0
        ):
            raise DeepseekV4ConfigError(
                f"'num_hash_layers' must be a non-negative integer, got "
                f"{num_hash_layers!r}"
            )
        if num_hash_layers > num_hidden_layers:
            raise DeepseekV4ConfigError(
                f"'num_hash_layers' ({num_hash_layers}) exceeds 'num_hidden_layers' "
                f"({num_hidden_layers})"
            )
        from_count = [
            MLPKind.HASH_MOE if i < num_hash_layers else MLPKind.ROUTED_MOE
            for i in range(num_hidden_layers)
        ]

    if from_types is not None and from_count is not None and from_types != from_count:
        mismatches = [
            i for i, (a, b) in enumerate(zip(from_types, from_count)) if a != b
        ]
        raise DeepseekV4ConfigError(
            f"DeepSeek-V4 config carries both 'mlp_layer_types' and "
            f"'num_hash_layers', and they disagree at layer(s) {mismatches[:8]}; "
            f"refusing to choose between them"
        )

    if from_types is not None:
        return from_types
    if from_count is not None:
        return from_count
    # Neither form present: no hash-MoE layers. Unlike the attention side this is
    # a safe default rather than an error -- "no layer uses hash routing" is a
    # coherent model, whereas "no layer has a compression ratio" is not.
    return [MLPKind.ROUTED_MOE] * num_hidden_layers


def _validate_unsupported_variants(config: Any) -> None:
    """Reject config variants this implementation does not handle.

    Each of these would otherwise run and quietly produce wrong output, which is
    the failure mode the whole plan is organized to avoid.
    """
    scoring_func = _get(config, "scoring_func", _SUPPORTED_SCORING_FUNC)
    if scoring_func != _SUPPORTED_SCORING_FUNC:
        raise DeepseekV4ConfigError(
            f"unsupported 'scoring_func' {scoring_func!r}; this implementation "
            f"targets {_SUPPORTED_SCORING_FUNC!r}"
        )

    topk_method = _get(config, "topk_method", _SUPPORTED_TOPK_METHOD)
    if topk_method != _SUPPORTED_TOPK_METHOD:
        raise DeepseekV4ConfigError(
            f"unsupported 'topk_method' {topk_method!r}; this implementation "
            f"targets {_SUPPORTED_TOPK_METHOD!r}"
        )

    # V4 dropped V3's grouped / node-limited routing. A config that still carries
    # them is a different architecture, not a V4 with extra fields.
    for field in ("n_group", "topk_group"):
        value = _get(config, field, None)
        if value is not None:
            raise DeepseekV4ConfigError(
                f"config sets {field!r}={value!r}; grouped (node-limited) expert "
                f"routing is a DeepSeek-V3 feature and is not implemented for V4"
            )


def normalize_config(config: Any) -> NormalizedDeepseekV4Config:
    """Validate *config* and return the form-independent view.

    Raises :class:`DeepseekV4ConfigError` on any config that is unrecognized,
    internally inconsistent, or uses a variant this implementation does not
    support. It never falls back to a default reading.
    """
    layers = normalize_layer_specs(config)
    _validate_unsupported_variants(config)

    num_key_value_heads = _require_positive_int(config, "num_key_value_heads")
    if num_key_value_heads != 1:
        # MLA keeps a single latent KV per token; anything else means the
        # attention block is not the one implemented here.
        raise DeepseekV4ConfigError(
            f"'num_key_value_heads' is {num_key_value_heads}, expected 1 for "
            f"DeepSeek-V4 latent (MLA) attention"
        )

    n_routed_experts = _require_positive_int(config, "n_routed_experts")
    num_experts_per_tok = _require_positive_int(config, "num_experts_per_tok")
    if num_experts_per_tok > n_routed_experts:
        raise DeepseekV4ConfigError(
            f"'num_experts_per_tok' ({num_experts_per_tok}) exceeds "
            f"'n_routed_experts' ({n_routed_experts})"
        )

    sliding_window = _require_positive_int(config, "sliding_window")

    n_shared_experts = _get(config, "n_shared_experts", 0)
    if (
        not isinstance(n_shared_experts, int)
        or isinstance(n_shared_experts, bool)
        or n_shared_experts < 0
    ):
        raise DeepseekV4ConfigError(
            f"'n_shared_experts' must be a non-negative integer, got {n_shared_experts!r}"
        )

    num_nextn_predict_layers = _get(config, "num_nextn_predict_layers", 0)
    if (
        not isinstance(num_nextn_predict_layers, int)
        or isinstance(num_nextn_predict_layers, bool)
        or num_nextn_predict_layers < 0
    ):
        raise DeepseekV4ConfigError(
            f"'num_nextn_predict_layers' must be a non-negative integer, got "
            f"{num_nextn_predict_layers!r}"
        )

    return NormalizedDeepseekV4Config(
        layers=layers,
        num_hidden_layers=len(layers),
        num_hash_layers=sum(1 for l in layers if l.mlp is MLPKind.HASH_MOE),
        sliding_window=sliding_window,
        num_attention_heads=_require_positive_int(config, "num_attention_heads"),
        num_key_value_heads=num_key_value_heads,
        head_dim=_require_positive_int(config, "head_dim"),
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        n_shared_experts=n_shared_experts,
        index_topk=_require_positive_int(config, "index_topk"),
        hc_mult=_require_positive_int(config, "hc_mult"),
        hc_sinkhorn_iters=_require_positive_int(config, "hc_sinkhorn_iters"),
        num_nextn_predict_layers=num_nextn_predict_layers,
    )
