# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 tensor-parallel sharding policy
=======================================

Qwen3.5's head counts do not divide the tensor-parallel degree the way a
conventional model's do, and the two layer types disagree about which degrees
are convenient:

  full attention   24 Q heads,  4 KV heads
  gated deltanet   16 K heads, 48 V heads (3 value heads per key head)

24 is not a multiple of 16 or 32, so a naive "heads // tp" rejects the very
degrees a 27B model wants. This module computes an explicit, validated policy
instead, so the failure mode is a clear error at config time rather than a
silently wrong shard -- which is how ``deepseek_v4/parallel.py`` handles the
same class of problem.

Two mechanisms cover the awkward degrees:

*Query-head padding.* Q heads are padded up to a multiple of ``tp`` with zeroed
rows in both ``q_proj`` and ``o_proj``, so padded heads compute zeros and
contribute nothing to the output sum. 24 -> 32 at tp 16 and 32.

*Value-dimension splitting.* The gated delta rule is separable along the value
dimension: the state update ``S[k, v] += k_k * delta_v`` couples over ``k``
only, so different ``v`` columns are independent. Once ``tp`` exceeds the 16 key
heads we stop splitting heads and start splitting each head's 128-wide value
dimension. The one thing this costs is the gated RMSNorm, which normalizes over
``head_v_dim``: with the value dimension split, its sum-of-squares needs an
all-reduce across the ranks sharing a head.
"""

from dataclasses import dataclass
from math import ceil

from .config import Qwen3_5TextConfig


@dataclass(frozen=True)
class TPContext:
    """The tensor-parallel group, degrading to a single CPU rank.

    Mirrors ``deepseek_v4/parallel.py::resolve_parallel_topology``: outside a
    vLLM engine there is no initialized process group, and a model that cannot
    be constructed without one cannot be diffed against HuggingFace on CPU --
    which is where every accuracy bug should be found first.
    """

    group: object | None
    world_size: int
    rank: int

    @property
    def device_group(self):
        return self.group.device_group if self.group is not None else None


def resolve_tp_context() -> TPContext:
    try:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        return TPContext(group, group.world_size, group.rank_in_group)
    except (AssertionError, RuntimeError, AttributeError, ValueError):
        return TPContext(None, 1, 0)


@dataclass(frozen=True)
class Qwen3_5ShardingPolicy:
    """Resolved per-rank geometry for one tensor-parallel degree."""

    tp_degree: int

    # -- Full attention
    num_q_heads: int
    num_kv_heads: int
    padded_q_heads: int
    q_heads_per_rank: int
    kv_heads_per_rank: int
    num_kv_replicas: int

    # -- Gated DeltaNet
    num_k_heads: int
    num_v_heads: int
    num_v_per_k: int
    key_head_dim: int
    value_head_dim: int
    k_heads_per_rank: int
    v_heads_per_rank: int
    #: How many ranks share one key-head group by splitting the value dim.
    #: 1 means pure head sharding.
    v_dim_shards: int
    v_dim_per_rank: int

    # -- Dense MLP
    intermediate_per_rank: int

    @property
    def q_head_padding(self) -> int:
        """Zero heads appended to reach a multiple of tp_degree."""
        return self.padded_q_heads - self.num_q_heads

    @property
    def gated_norm_needs_allreduce(self) -> bool:
        """True when the value dim is split, so the gated RMSNorm is partial."""
        return self.v_dim_shards > 1

    # -- Gated DeltaNet geometry ------------------------------------------
    #
    # Every consumer -- the layer's module sizes, the state-cache spec and all
    # five weight loaders -- derives its widths from here, so the partition is
    # defined exactly once.

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        """Width of the fused ``[q | k | v]`` conv input."""
        return 2 * self.key_dim + self.value_dim

    @property
    def key_dim_per_rank(self) -> int:
        return self.k_heads_per_rank * self.key_head_dim

    @property
    def value_dim_per_rank(self) -> int:
        return self.v_heads_per_rank * self.v_dim_per_rank

    @property
    def conv_dim_per_rank(self) -> int:
        """This rank's conv width. **Not** ``conv_dim // tp_degree``.

        Under value-dimension splitting the query and key blocks are
        *replicated* across the ranks sharing a key head -- only ``v`` is
        split. At tp=32 a rank therefore holds ``2 * 128 + 3 * 64 = 448``
        channels, not ``10240 // 32 = 320``. The naive form is correct at every
        degree up to 16 and wrong only at 32, which is the worst way to be
        wrong, so nothing may recompute it locally.
        """
        return 2 * self.key_dim_per_rank + self.value_dim_per_rank

    # -- Full-attention head partition -------------------------------------
    #
    # These exist for the same reason as the Gated DeltaNet helpers below: so
    # the partition is stated in one place instead of being re-derived inside
    # each weight loader. Their absence is precisely why the KV mis-pairing
    # below went unnoticed -- the attention loaders computed
    # ``rank // num_kv_replicas`` inline, which is right only when the query
    # heads are unpadded, and no test asserted which global heads a rank owns.

    @property
    def gqa_group_size(self) -> int:
        """Query heads sharing one KV head in the *unsharded* checkpoint."""
        return self.num_q_heads // self.num_kv_heads

    def q_head_indices(self, rank: int) -> list[int]:
        """Global query heads owned by ``rank``.

        Indices at or beyond ``num_q_heads`` are padding: they are loaded as
        zero rows in ``q_proj`` and their ``o_proj`` columns are zeroed too.
        """
        first = rank * self.q_heads_per_rank
        return list(range(first, first + self.q_heads_per_rank))

    def kv_head_index(self, rank: int) -> int:
        """Global KV head serving ``rank``'s query heads.

        Derived from the rank's first *real* query head, not from the rank
        index. ``rank // num_kv_replicas`` -- what the loader used to do --
        assumes rank order maps linearly onto KV heads, which holds only when
        the query heads are unpadded. It is therefore correct for the 0.8B
        (8 Q / 2 KV, no padding at any degree) and wrong for the 27B
        (24 Q / 4 KV) as soon as padding appears: at tp=16 the 24 heads pad to
        32, and rank 3 owns global query heads 6-7, which belong to KV head 1,
        while ``3 // 4`` hands it KV head 0. Ranks 3, 6, 7, 9, 10 and 11 are all
        mis-paired at tp=16, and tp=32 is wrong too.

        Ranks holding only padded query heads are clamped to the last KV head;
        their attention output is annihilated by the zeroed ``o_proj`` columns,
        so the choice is arbitrary but the index must stay in range.
        """
        first_head = rank * self.q_heads_per_rank
        return min(first_head // self.gqa_group_size, self.num_kv_heads - 1)

    # -- Which global heads / rows belong to a rank ------------------------

    def k_head_indices(self, rank: int) -> list[int]:
        """Global key heads held by ``rank``."""
        if self.v_dim_shards > 1:
            # Consecutive ranks share a key head and split its value dim.
            return [rank // self.v_dim_shards]
        start = rank * self.k_heads_per_rank
        return list(range(start, start + self.k_heads_per_rank))

    def v_head_indices(self, rank: int) -> list[int]:
        """Global value heads held by ``rank``. Value head ``j`` belongs to key
        head ``j // num_v_per_k``, so these follow the key heads."""
        return [
            k * self.num_v_per_k + j
            for k in self.k_head_indices(rank)
            for j in range(self.num_v_per_k)
        ]

    def v_dim_offset(self, rank: int) -> int:
        """Offset into each value head's dimension, 0 unless the dim is split."""
        if self.v_dim_shards == 1:
            return 0
        return (rank % self.v_dim_shards) * self.v_dim_per_rank

    def key_row_ranges(self, rank: int) -> list[tuple[int, int]]:
        """Half-open row ranges into one ``key_dim``-wide block."""
        d = self.key_head_dim
        return [(h * d, (h + 1) * d) for h in self.k_head_indices(rank)]

    def value_row_ranges(self, rank: int) -> list[tuple[int, int]]:
        """Half-open row ranges into one ``value_dim``-wide block."""
        d = self.value_head_dim
        lo = self.v_dim_offset(rank)
        width = self.v_dim_per_rank
        return [(h * d + lo, h * d + lo + width) for h in self.v_head_indices(rank)]

    def conv_row_ranges(self, rank: int) -> list[tuple[int, int]]:
        """Row ranges into the fused ``[q | k | v]`` conv block, in order."""
        kd = self.key_dim
        qk = self.key_row_ranges(rank)
        return (
            qk
            + [(lo + kd, hi + kd) for lo, hi in qk]
            + [(lo + 2 * kd, hi + 2 * kd) for lo, hi in self.value_row_ranges(rank)]
        )


def resolve_sharding(
    config: Qwen3_5TextConfig, tp_degree: int
) -> Qwen3_5ShardingPolicy:
    """Compute and validate the sharding policy, or raise with a clear reason."""
    if tp_degree < 1:
        raise ValueError(f"tp_degree must be >= 1, got {tp_degree}")

    # ---- Dense MLP -------------------------------------------------------
    if config.intermediate_size % tp_degree:
        raise ValueError(
            f"intermediate_size={config.intermediate_size} is not divisible by "
            f"tp_degree={tp_degree}."
        )
    if config.hidden_size % tp_degree:
        # Not because the hidden dim is sharded -- it is not. Sequence
        # parallelism scatters *tokens* (dim 0); every rank holds the full
        # hidden width. The real constraint is that the token count handed to
        # the embedding must divide the world size (nn/embedding.py:200
        # asserts it), and the bucket sizes that satisfy that are powers of two
        # like hidden_size. Keeping the check is cheap insurance against an
        # exotic degree; the original rationale was simply wrong.
        raise ValueError(
            f"hidden_size={config.hidden_size} is not divisible by "
            f"tp_degree={tp_degree}."
        )

    # ---- Full attention --------------------------------------------------
    # Q heads pad up; padded heads are zeroed in q_proj and o_proj so they
    # contribute nothing to the reduced output.
    padded_q_heads = ceil(config.num_attention_heads / tp_degree) * tp_degree
    q_heads_per_rank = padded_q_heads // tp_degree

    kv_heads = config.num_key_value_heads
    if tp_degree >= kv_heads:
        if tp_degree % kv_heads:
            raise ValueError(
                f"tp_degree={tp_degree} must be a multiple of "
                f"num_key_value_heads={kv_heads} when it exceeds it, so each KV "
                "head can be replicated across a whole number of ranks."
            )
        kv_heads_per_rank = 1
        num_kv_replicas = tp_degree // kv_heads
    else:
        if kv_heads % tp_degree:
            raise ValueError(
                f"num_key_value_heads={kv_heads} is not divisible by "
                f"tp_degree={tp_degree}."
            )
        kv_heads_per_rank = kv_heads // tp_degree
        num_kv_replicas = 1

    # A rank holding one KV head must have all of its real query heads inside
    # that head's GQA group, or one of them attends against the wrong K/V.
    #
    # This can only break when the query heads are padded, because padding is
    # appended at the end rather than distributed per group, so it shifts the
    # rank boundaries off the group boundaries. Both shipped checkpoints are
    # clean at every supported degree -- 8Q/2KV (group 4) and 24Q/4KV (group 6)
    # -- but the combination is expressible, and a straddling rank is exactly
    # the kind of silently-wrong shard this module exists to turn into an error.
    if kv_heads_per_rank == 1 and config.num_attention_heads % kv_heads == 0:
        group = config.num_attention_heads // kv_heads
        for rank in range(tp_degree):
            first = rank * q_heads_per_rank
            if first >= config.num_attention_heads:
                continue  # padding only; its output is zeroed by o_proj
            last = min(first + q_heads_per_rank - 1, config.num_attention_heads - 1)
            if first // group != last // group:
                raise ValueError(
                    f"tp_degree={tp_degree} splits query heads "
                    f"{q_heads_per_rank} per rank, but rank {rank} would own "
                    f"heads {first}-{last}, which span two GQA groups of "
                    f"{group} (num_attention_heads="
                    f"{config.num_attention_heads}, num_key_value_heads="
                    f"{kv_heads}). One KV head cannot serve both, so this "
                    "degree would shard attention silently wrong."
                )

    # ---- Gated DeltaNet --------------------------------------------------
    k_heads = config.linear_num_key_heads
    v_heads = config.linear_num_value_heads
    v_head_dim = config.linear_value_head_dim

    if k_heads % tp_degree == 0:
        # Pure head sharding. v_heads is a multiple of k_heads by config
        # validation, so this divides too.
        k_heads_per_rank = k_heads // tp_degree
        v_heads_per_rank = v_heads // tp_degree
        v_dim_shards = 1
    elif tp_degree % k_heads == 0:
        # More ranks than key heads: split each head's value dimension.
        v_dim_shards = tp_degree // k_heads
        if v_head_dim % v_dim_shards:
            raise ValueError(
                f"tp_degree={tp_degree} needs the value dim split "
                f"{v_dim_shards} ways, but linear_value_head_dim={v_head_dim} "
                "is not divisible by that."
            )
        k_heads_per_rank = 1
        v_heads_per_rank = config.num_v_per_k
    else:
        raise ValueError(
            f"tp_degree={tp_degree} is not supported for Gated DeltaNet: it "
            f"neither divides linear_num_key_heads={k_heads} nor is a multiple "
            f"of it. Supported degrees divide {k_heads} (pure head sharding) or "
            f"are a multiple of it (value-dimension splitting)."
        )

    return Qwen3_5ShardingPolicy(
        tp_degree=tp_degree,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        padded_q_heads=padded_q_heads,
        q_heads_per_rank=q_heads_per_rank,
        kv_heads_per_rank=kv_heads_per_rank,
        num_kv_replicas=num_kv_replicas,
        num_k_heads=k_heads,
        num_v_heads=v_heads,
        num_v_per_k=config.num_v_per_k,
        key_head_dim=config.linear_key_head_dim,
        value_head_dim=v_head_dim,
        k_heads_per_rank=k_heads_per_rank,
        v_heads_per_rank=v_heads_per_rank,
        v_dim_shards=v_dim_shards,
        v_dim_per_rank=v_head_dim // v_dim_shards,
        intermediate_per_rank=config.intermediate_size // tp_degree,
    )
