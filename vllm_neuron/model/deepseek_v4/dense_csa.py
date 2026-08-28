# SPDX-License-Identifier: Apache-2.0
"""The dense-CSA equivalence bound and its admission guard.

**The lightning indexer now exists** (``indexer.py``), so the guard this module
computes is no longer installed by default -- see
``platform.py::_configure_deepseek_v4_dense_csa_bound`` and the
``VLLM_NEURON_DEEPSEEK_V4_DENSE_CSA_BOUND=1`` opt-in. What follows described why
the bound was needed while CSA attended densely; it is kept because the
equivalence it derives is still the thing that makes the indexer *testable*.
Below the bound, an indexer that selects everything and a correct one are
indistinguishable, which is why the indexer's own tests deliberately run past
it (``test_deepseek_v4_component_oracles.py``,
``test_deepseek_v4_model_assembly.py``).

Plan P5. DeepSeek-V4's lightning indexer **only selects; it never weights**. So
wherever the eligible compressed set is no larger than ``index_topk``, selecting
the top-k *is* selecting everything, and running dense attention over the whole
compressed set is not an approximation -- it is the same computation. That is the
lever the staged bring-up rides on: the indexer can be omitted entirely below the
bound, and every correctness result gathered there is a real result.

Above the bound it stops being true silently. Nothing throws, nothing looks
wrong, and the logits are simply incorrect. Hence this module, and hence two
deliberate design choices:

**No fabricated constants.** :class:`CompressorGeometry` has no default kernel
width or initial offset. They must be read from the pinned compressor
implementation and passed in. A plausible-looking default would produce a
plausible-looking bound that had never been derived from anything, which is the
exact failure this guard exists to prevent.

**Conservative by default.** :attr:`CountingMode.STARTED` counts every window the
sequence has begun, complete or not, which upper-bounds the true eligible count
and therefore lower-bounds the safe length. Switching to
:attr:`CountingMode.COMPLETE` is only correct once the pinned implementation is
known to exclude in-flight partial windows from indexer candidacy. **It does**:
implementing the indexer settled this. Candidates are exactly the emitted
entries, ``(position + 1) // ratio`` of them
(:func:`~vllm_neuron.model.deepseek_v4.attention.visible_compressed_entries`),
and a window still filling has emitted nothing to select. ``COMPLETE`` is the
correct mode, not merely the optimistic one.

The admission rule is the other half. Checking the *current* sequence length is
not enough: a request admitted inside the bound can generate its way across it
mid-decode, at which point the dense path is quietly wrong for the remainder of
the response. Admission therefore tests ``prompt + maximum requested output``.

Free of ``torch`` and ``vllm`` imports: this is integer arithmetic, and keeping
it testable on a bare interpreter is what lets the boundary cases be pinned
before hardware exists.
"""

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "CompressorGeometry",
    "CountingMode",
    "DenseCsaBound",
    "DenseCsaUnsupportedError",
    "check_admission",
    "eligible_entries",
    "geometry_from_config",
    "max_dense_csa_tokens",
    "model_bound",
]


class DenseCsaUnsupportedError(ValueError):
    """Raised when a request could leave the range where dense CSA is exact."""


class CountingMode(str, Enum):
    """Which compressed windows count as eligible for indexer selection."""

    #: Every window the sequence has begun, including one still filling.
    #: Upper-bounds the true count, so the resulting bound is safe under
    #: uncertainty about how partial carry state is treated.
    STARTED = "started"
    #: Only fully-populated windows -- the correct mode. A window still filling
    #: has emitted no entry, so there is nothing for the indexer to select;
    #: confirmed by the indexer's own candidate set, ``(position + 1) // ratio``.
    COMPLETE = "complete"


@dataclass(frozen=True)
class CompressorGeometry:
    """One layer's compression geometry, as read from the pinned implementation.

    Every field is required. See the module docstring: defaults here would be
    guesses wearing the costume of a derivation.

    Window ``i`` covers native token positions
    ``[initial_offset + i*stride, initial_offset + i*stride + kernel_width)``.
    """

    #: Layer's ``compress_ratio``. 0 marks a sliding-window layer, which has no
    #: compressed entries and therefore no bound.
    compress_ratio: int
    #: Native token positions between consecutive compressed entries.
    stride: int
    #: Native token positions folded into one compressed entry.
    kernel_width: int
    #: Native token positions before the first compressed window begins.
    initial_offset: int
    #: Entries always eligible regardless of length (sinks, reserved slots).
    #: They consume top-k budget, so they shrink the bound.
    reserved_entries: int

    def __post_init__(self) -> None:
        if self.compress_ratio < 0:
            raise ValueError("compress_ratio must be non-negative")
        if self.initial_offset < 0:
            raise ValueError("initial_offset must be non-negative")
        if self.reserved_entries < 0:
            raise ValueError("reserved_entries must be non-negative")
        if self.is_sliding_window:
            return
        if self.stride < 1:
            raise ValueError("stride must be >= 1 for a compressed layer")
        if self.kernel_width < 1:
            raise ValueError("kernel_width must be >= 1 for a compressed layer")

    @property
    def is_sliding_window(self) -> bool:
        return self.compress_ratio == 0


#: Compression rate each compressed layer type denotes, used when a config
#: carries ``compress_ratios`` but not the ``compress_rates`` dict.
_RATE_BY_LAYER_TYPE = {
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def geometry_from_config(config, layer_type: str) -> CompressorGeometry:
    """Derive entry-emission geometry from a pinned DeepSeek-V4 config.

    Transformers 5.15 emits one entry after each complete ``compress_rate``
    group and carries the incomplete suffix. CSA combines two adjacent groups
    when computing an entry, but that overlap does not delay emission: its
    eligible-entry cadence remains one per c4 group.
    """
    if layer_type == "sliding_attention":
        return CompressorGeometry(0, 0, 0, 0, 0)
    rates = getattr(config, "compress_rates", None)
    if isinstance(rates, dict) and layer_type in rates:
        rate = rates[layer_type]
    else:
        # Official checkpoints carry only the per-layer ``compress_ratios`` list
        # and no ``compress_rates`` dict, so recover the rate from the layer type
        # rather than rejecting the config.
        rate = _RATE_BY_LAYER_TYPE.get(layer_type)
        if rate is None:
            raise DenseCsaUnsupportedError(
                f"no pinned compressor rate for layer type {layer_type!r}"
            )
    if not isinstance(rate, int) or isinstance(rate, bool) or rate < 1:
        raise DenseCsaUnsupportedError(
            f"invalid compressor rate {rate!r} for layer type {layer_type!r}"
        )
    return CompressorGeometry(
        compress_ratio=rate,
        stride=rate,
        # Entry-emission completion width. CSA's value computation has a
        # 2*rate receptive field but still emits every complete rate-token group.
        kernel_width=rate,
        initial_offset=0,
        reserved_entries=0,
    )


def eligible_entries(
    total_tokens: int,
    geometry: CompressorGeometry,
    *,
    mode: CountingMode = CountingMode.STARTED,
) -> int:
    """Compressed entries the indexer would choose among at *total_tokens*.

    Monotonically non-decreasing in *total_tokens*, which is what makes the
    inverse in :func:`max_dense_csa_tokens` a single threshold rather than a
    region.
    """
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if geometry.is_sliding_window:
        return 0

    span = total_tokens - geometry.initial_offset
    if span <= 0:
        windows = 0
    elif mode is CountingMode.STARTED:
        # Window i is started once total_tokens > initial_offset + i*stride.
        windows = math.ceil(span / geometry.stride)
    else:
        # Window i is complete once total_tokens >= offset + i*stride + width.
        if total_tokens < geometry.initial_offset + geometry.kernel_width:
            windows = 0
        else:
            windows = (
                total_tokens - geometry.initial_offset - geometry.kernel_width
            ) // geometry.stride + 1
    return windows + geometry.reserved_entries


def max_dense_csa_tokens(
    geometry: CompressorGeometry,
    index_topk: int,
    *,
    mode: CountingMode = CountingMode.STARTED,
) -> int | None:
    """Greatest total length at which dense attention still equals top-k CSA.

    ``None`` means unbounded: a sliding-window layer never accumulates
    compressed entries, so the equivalence cannot lapse.

    Raises :class:`DenseCsaUnsupportedError` when the layer's reserved entries
    alone exceed ``index_topk``. That is not a short bound -- it is a
    configuration where the eligible set is over budget at *every* length,
    including an empty sequence, so there is no safe range to return.
    """
    if index_topk < 1:
        raise ValueError("index_topk must be >= 1")
    if geometry.is_sliding_window:
        return None

    budget = index_topk - geometry.reserved_entries
    if budget < 0:
        raise DenseCsaUnsupportedError(
            f"layer reserves {geometry.reserved_entries} always-eligible entries, "
            f"which alone exceed index_topk={index_topk}; dense CSA is never "
            f"equivalent to top-k selection for this configuration, at any "
            f"sequence length"
        )
    if budget == 0:
        # The reserved entries exactly consume the budget, so no compressed
        # window may ever become eligible: the bound is the last length before
        # the first window counts.
        limit = geometry.initial_offset
        if mode is CountingMode.COMPLETE:
            limit += geometry.kernel_width - 1
        return max(0, limit)

    if mode is CountingMode.STARTED:
        # ceil(span / stride) <= budget  <=>  span <= stride * budget
        return geometry.initial_offset + geometry.stride * budget
    # (span - width) // stride + 1 <= budget  <=>  span <= width + stride*budget - 1
    return geometry.initial_offset + geometry.kernel_width + geometry.stride * budget - 1


@dataclass(frozen=True)
class DenseCsaBound:
    """The model-wide safe length, and which layer set it."""

    #: ``None`` when no layer imposes a bound.
    max_total_tokens: int | None
    #: Index of the binding layer, or ``None`` when unbounded.
    binding_layer: int | None
    mode: CountingMode

    def permits(self, total_tokens: int) -> bool:
        return self.max_total_tokens is None or total_tokens <= self.max_total_tokens


def model_bound(
    geometries: dict[int, CompressorGeometry],
    index_topk: int,
    *,
    mode: CountingMode = CountingMode.STARTED,
) -> DenseCsaBound:
    """The tightest per-layer bound across the model.

    The model is only safe where *every* layer is safe, so the minimum binds.
    Ties resolve to the lowest layer index, purely for a stable error message.
    """
    if not geometries:
        raise ValueError("no layer geometries supplied")

    tightest: int | None = None
    binding: int | None = None
    for layer_index in sorted(geometries):
        limit = max_dense_csa_tokens(geometries[layer_index], index_topk, mode=mode)
        if limit is None:
            continue
        if tightest is None or limit < tightest:
            tightest, binding = limit, layer_index
    return DenseCsaBound(max_total_tokens=tightest, binding_layer=binding, mode=mode)


def check_admission(
    prompt_tokens: int,
    max_output_tokens: int | None,
    bound: DenseCsaBound,
) -> int:
    """Validate a request against *bound*, returning the total it was checked at.

    ``max_output_tokens=None`` means the response length is not capped. That is
    rejected outright whenever a bound exists: an uncapped generation cannot be
    shown to stay inside any finite range, and admitting it would defer the
    failure to the middle of a response.
    """
    if prompt_tokens < 0:
        raise ValueError("prompt_tokens must be non-negative")
    if max_output_tokens is not None and max_output_tokens < 0:
        raise ValueError("max_output_tokens must be non-negative")

    if bound.max_total_tokens is None:
        return prompt_tokens + (max_output_tokens or 0)

    if max_output_tokens is None:
        raise DenseCsaUnsupportedError(
            "request has no output-length cap, so it cannot be shown to stay "
            f"within the dense-CSA safe length of {bound.max_total_tokens} tokens "
            f"(set by layer {bound.binding_layer}). Set an explicit maximum "
            "output length, or enable the CSA indexer"
        )

    total = prompt_tokens + max_output_tokens
    if total > bound.max_total_tokens:
        raise DenseCsaUnsupportedError(
            f"request may reach {total} tokens "
            f"({prompt_tokens} prompt + {max_output_tokens} generated), above the "
            f"dense-CSA safe length of {bound.max_total_tokens} tokens set by "
            f"layer {bound.binding_layer} (counting mode: {bound.mode.value}). "
            "Below that length, skipping the indexer is exact; above it the "
            "result would be silently wrong. Shorten the request or enable the "
            "CSA indexer"
        )
    return total
