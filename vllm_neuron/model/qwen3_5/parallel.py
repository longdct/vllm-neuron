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
class Qwen3_5ShardingPolicy:
    """Resolved per-rank geometry for one tensor-parallel degree."""

    tp_degree: int

    # -- Full attention
    num_q_heads: int
    padded_q_heads: int
    q_heads_per_rank: int
    kv_heads_per_rank: int
    num_kv_replicas: int

    # -- Gated DeltaNet
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
        # Sequence parallelism scatters the hidden dim across ranks.
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
        padded_q_heads=padded_q_heads,
        q_heads_per_rank=q_heads_per_rank,
        kv_heads_per_rank=kv_heads_per_rank,
        num_kv_replicas=num_kv_replicas,
        k_heads_per_rank=k_heads_per_rank,
        v_heads_per_rank=v_heads_per_rank,
        v_dim_shards=v_dim_shards,
        v_dim_per_rank=v_head_dim // v_dim_shards,
        intermediate_per_rank=config.intermediate_size // tp_degree,
    )
