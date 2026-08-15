# SPDX-License-Identifier: Apache-2.0
"""Per-rank memory accounting from checkpoint metadata, without loading weights.

Plan P7a. The question this answers is "does this checkpoint fit, per rank, on
the target instance" -- asked *before* spending hardware time discovering the
answer, and before writing a loader whose shape is decided by the answer.

Two properties make it worth having:

* **Parameter accounting is exact, even on a laptop.** A safetensors file begins
  with an 8-byte little-endian header length followed by that many bytes of
  JSON, giving every tensor's dtype and shape. Reading a few kilobytes yields
  byte-exact parameter totals for a 284B-parameter checkpoint. No download of
  the payload, no torch, no device.
* **Everything else is a modelled estimate**, and is labelled as such. Compiler
  arena, activation peak, and allocator fragmentation cannot be derived from
  metadata. They are carried as ranges with an explicit basis, and the total is
  reported as a range -- never as a single number that would read as measured.

That distinction is the whole point. A budget that silently blends exact and
guessed quantities is the kind of artifact that gets quoted as evidence, so
:class:`Component` records which kind each line is and :class:`MemoryBudget`
refuses to collapse to a scalar.

This module is free of ``torch``, ``vllm``, and ``safetensors`` imports so it can
run wherever the checkpoint metadata can be fetched.
"""

import json
import math
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = [
    "Component",
    "ComponentKind",
    "Estimate",
    "MemoryBudget",
    "ParallelLayout",
    "ShardRule",
    "TensorMeta",
    "UnknownDtypeError",
    "build_weight_budget",
    "conversion_peak_bytes",
    "default_shard_rule",
    "parse_safetensors_header",
    "read_safetensors_header",
]

GIB = 1024**3


class UnknownDtypeError(ValueError):
    """Raised for a safetensors dtype with no known width.

    Deliberately fatal rather than defaulted: guessing a width silently scales
    the entire budget by a wrong factor, which is worse than no budget.
    """


#: Bytes per element for safetensors dtype strings. Sub-byte types are stored
#: packed and expressed as a fraction, so ``numel * width`` stays correct.
_DTYPE_BYTES: dict[str, float] = {
    "F64": 8.0,
    "F32": 4.0,
    "F16": 2.0,
    "BF16": 2.0,
    "F8_E4M3": 1.0,
    "F8_E5M2": 1.0,
    "F4": 0.5,
    "I64": 8.0,
    "U64": 8.0,
    "I32": 4.0,
    "U32": 4.0,
    "I16": 2.0,
    "U16": 2.0,
    "I8": 1.0,
    "U8": 1.0,
    "BOOL": 1.0,
}


def dtype_width(dtype: str) -> float:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError:
        raise UnknownDtypeError(
            f"unknown safetensors dtype {dtype!r}; known dtypes are "
            f"{sorted(_DTYPE_BYTES)}. Refusing to assume a width"
        ) from None


@dataclass(frozen=True)
class TensorMeta:
    """One tensor's metadata, as recorded in a safetensors header."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        """Stored size. Sub-byte dtypes round up: a tensor occupies whole bytes."""
        return math.ceil(self.numel * dtype_width(self.dtype))


def parse_safetensors_header(data: bytes) -> tuple[TensorMeta, ...]:
    """Parse tensor metadata from the leading bytes of a safetensors file.

    Accepts any buffer long enough to contain the 8-byte length prefix and the
    JSON header; the tensor payload need not be present.
    """
    if len(data) < 8:
        raise ValueError(
            f"safetensors header needs at least 8 bytes, got {len(data)}"
        )
    (header_len,) = struct.unpack("<Q", data[:8])
    if len(data) < 8 + header_len:
        raise ValueError(
            f"safetensors header declares {header_len} bytes but only "
            f"{len(data) - 8} are available; read more of the file"
        )
    header = json.loads(data[8 : 8 + header_len])
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")

    tensors = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict) or "dtype" not in entry:
            raise ValueError(f"safetensors header entry {name!r} is malformed")
        tensors.append(
            TensorMeta(
                name=name,
                dtype=entry["dtype"],
                shape=tuple(entry.get("shape", ())),
            )
        )
    return tuple(tensors)


def read_safetensors_header(path: str | Path, *, max_header: int = 64 << 20):
    """Read just the header of a safetensors file from disk.

    Reads the 8-byte prefix, then exactly the declared header length -- so this
    touches kilobytes of a file that may be hundreds of gigabytes.
    """
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise ValueError(f"{path}: too short to be a safetensors file")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > max_header:
            raise ValueError(
                f"{path}: header claims {header_len} bytes, above the "
                f"{max_header}-byte sanity limit"
            )
        return parse_safetensors_header(prefix + handle.read(header_len))


class ShardRule(str, Enum):
    """How one tensor is divided across ranks."""

    #: Present in full on every rank (norms, embeddings when not sharded).
    REPLICATED = "replicated"
    #: Split across the tensor-parallel group.
    TENSOR_PARALLEL = "tensor_parallel"
    #: Split across the expert-parallel group.
    EXPERT_PARALLEL = "expert_parallel"


@dataclass(frozen=True)
class ParallelLayout:
    """The parallelism the budget is computed for."""

    tp_size: int = 1
    ep_size: int = 1
    #: Shard sizes are rounded up to this many elements, modelling the padding
    #: real sharding introduces. 1 disables the adjustment.
    alignment_elements: int = 1

    def __post_init__(self) -> None:
        for name in ("tp_size", "ep_size", "alignment_elements"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")

    def divisor(self, rule: ShardRule) -> int:
        return {
            ShardRule.REPLICATED: 1,
            ShardRule.TENSOR_PARALLEL: self.tp_size,
            ShardRule.EXPERT_PARALLEL: self.ep_size,
        }[rule]


def default_shard_rule(name: str) -> ShardRule:
    """A serviceable default classifier keyed on tensor name.

    Expert weights shard expert-parallel; per-layer norms and scalars replicate;
    everything else shards tensor-parallel. Callers with a real sharding plan
    should pass their own -- this exists so a budget can be produced before the
    loader is written, not to be authoritative about it.
    """
    lowered = name.lower()
    if "expert" in lowered:
        return ShardRule.EXPERT_PARALLEL
    if any(token in lowered for token in ("norm", "bias", "scale", "sink")):
        return ShardRule.REPLICATED
    return ShardRule.TENSOR_PARALLEL


def sharded_bytes(
    tensor: TensorMeta,
    layout: ParallelLayout,
    rule: ShardRule,
    *,
    width: float | None = None,
) -> int:
    """Bytes this tensor occupies on one rank.

    Rounds the per-rank element count up to the layout's alignment and up to a
    whole byte, so the imbalance from indivisible shapes is counted rather than
    averaged away.
    """
    divisor = layout.divisor(rule)
    per_rank_elements = math.ceil(tensor.numel / divisor)
    if layout.alignment_elements > 1:
        per_rank_elements = (
            math.ceil(per_rank_elements / layout.alignment_elements)
            * layout.alignment_elements
        )
    return math.ceil(per_rank_elements * (width if width is not None else dtype_width(tensor.dtype)))


class ComponentKind(str, Enum):
    """Whether a budget line is derived or modelled."""

    #: Computed from checkpoint metadata. Byte-exact given the sharding plan.
    EXACT = "exact"
    #: Modelled. Carries a range and a stated basis.
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class Estimate:
    """A modelled quantity with an explicit range and rationale."""

    low: int
    high: int
    basis: str

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"estimate low ({self.low}) exceeds high ({self.high})")
        if not self.basis.strip():
            raise ValueError("an estimate must state its basis")


@dataclass(frozen=True)
class Component:
    """One line of the per-rank budget."""

    name: str
    kind: ComponentKind
    low: int
    high: int
    note: str = ""

    @classmethod
    def exact(cls, name: str, nbytes: int, note: str = "") -> "Component":
        return cls(name, ComponentKind.EXACT, nbytes, nbytes, note)

    @classmethod
    def estimated(cls, name: str, estimate: Estimate) -> "Component":
        return cls(name, ComponentKind.ESTIMATED, estimate.low, estimate.high, estimate.basis)


@dataclass(frozen=True)
class MemoryBudget:
    """A per-rank budget, reported as a range.

    There is deliberately no ``total`` scalar. Any single number would have to
    silently pick a point inside the estimated range, and this budget's whole
    purpose is to keep the modelled part visibly separate from the derived part.
    """

    components: tuple[Component, ...]
    layout: ParallelLayout

    @property
    def exact_bytes(self) -> int:
        return sum(c.low for c in self.components if c.kind is ComponentKind.EXACT)

    @property
    def total_low(self) -> int:
        return sum(c.low for c in self.components)

    @property
    def total_high(self) -> int:
        return sum(c.high for c in self.components)

    def fits_in(self, capacity_bytes: int) -> bool | None:
        """``True``/``False`` when unambiguous, ``None`` when the range straddles.

        A straddling range is not a "probably yes". It means the estimated
        components decide the answer, which is exactly when a measurement is
        required instead (plan P7b).
        """
        if self.total_high <= capacity_bytes:
            return True
        if self.total_low > capacity_bytes:
            return False
        return None

    def render(self) -> str:
        """A human-readable table, kinds marked."""
        width = max(len(c.name) for c in self.components) if self.components else 0
        lines = []
        for c in self.components:
            mark = "=" if c.kind is ComponentKind.EXACT else "~"
            if c.low == c.high:
                size = f"{c.low / GIB:8.2f} GiB"
            else:
                size = f"{c.low / GIB:8.2f}-{c.high / GIB:.2f} GiB"
            lines.append(f"  {mark} {c.name:<{width}}  {size}   {c.note}".rstrip())
        lines.append(
            f"  TOTAL per rank: {self.total_low / GIB:.2f}-{self.total_high / GIB:.2f} GiB "
            f"(exact portion {self.exact_bytes / GIB:.2f} GiB)"
        )
        return "\n".join(lines)


def conversion_peak_bytes(
    tensors: Sequence[TensorMeta],
    layout: ParallelLayout,
    *,
    classify: Callable[[str], ShardRule] = default_shard_rule,
    destination_width: float | None = None,
    streaming: bool = True,
) -> int:
    """Peak per-rank bytes while converting weights to ``destination_width``.

    This is the quantity that decides whether BF16 loading is viable, and the
    reason the plan calls for streaming shard-by-shard conversion:

    * **Streaming** holds the finished destination plus *one* source shard, so
      the peak is ``sum(destination) + max(source shard)``.
    * **Non-streaming** materializes every source tensor before converting, so
      the peak is ``sum(source) + sum(destination)`` -- on a 284B checkpoint the
      difference is hundreds of gigabytes per rank.

    ``destination_width=None`` means no conversion: the peak is just the
    resident source.
    """
    resident_source = 0
    resident_destination = 0
    largest_source_shard = 0
    for tensor in tensors:
        rule = classify(tensor.name)
        source = sharded_bytes(tensor, layout, rule)
        resident_source += source
        largest_source_shard = max(largest_source_shard, source)
        if destination_width is not None:
            resident_destination += sharded_bytes(
                tensor, layout, rule, width=destination_width
            )

    if destination_width is None:
        return resident_source
    if streaming:
        return resident_destination + largest_source_shard
    return resident_source + resident_destination


def build_weight_budget(
    tensors: Iterable[TensorMeta],
    layout: ParallelLayout,
    *,
    classify: Callable[[str], ShardRule] = default_shard_rule,
    destination_width: float | None = None,
    streaming: bool = True,
    extra: Sequence[Component] = (),
) -> MemoryBudget:
    """Assemble a per-rank budget from checkpoint metadata.

    Only the weight-derived lines are produced here, because only those can be
    computed exactly. Runtime lines -- KV cache, compressor state, activations,
    compiler arena, collective buffers -- are passed in via *extra*, so the
    caller states them explicitly (and, if modelled, with a basis) rather than
    having this function invent them.
    """
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError("no tensor metadata supplied")

    resident_width = destination_width
    resident = sum(
        sharded_bytes(t, layout, classify(t.name), width=resident_width)
        for t in tensors
    )
    peak = conversion_peak_bytes(
        tensors,
        layout,
        classify=classify,
        destination_width=destination_width,
        streaming=streaming,
    )

    components = [
        Component.exact(
            "weights (resident)",
            resident,
            "converted" if destination_width is not None else "as stored",
        )
    ]
    transient = peak - resident
    if transient > 0:
        components.append(
            Component.exact(
                "weights (load transient)",
                transient,
                "largest source shard, streaming"
                if streaming
                else "full source held during conversion",
            )
        )
    components.extend(extra)
    return MemoryBudget(components=tuple(components), layout=layout)
