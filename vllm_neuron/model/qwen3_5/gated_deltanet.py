# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 Gated DeltaNet (linear attention)
=========================================

48 of the 64 layers. Carries a recurrent state instead of a KV cache:

    S_t = (a_t * I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T
    o_t = S_t^T q_t

**This module is the CPU oracle, not the device path.** Its chunked form is a
faithful port of ``transformers``' ``torch_chunk_gated_delta_rule`` and exists to
be diffed per-element against it; the device path is the NKI kernel in
``nki_gdn.py``. The distinction matters because the reference contains two
Python loops that are fine in eager torch and catastrophic under XLA tracing:

* a ``for i in range(1, chunk_size)`` UT transform -- 63 sequential in-place
  slice writes;
* a loop over ``seq_len / chunk_size`` chunks.

Traced with static bucket shapes, both unroll completely. That is precisely how
DeepSeek-V4 turned a 2,637-node decode graph into a 55,179-node prefill graph
and then spent 2h52m in the compiler without emitting a NEFF
(docs/model-dev/deepseek-v4-trn2-compilation.md,
docs/model-dev/deepseek-v4-mla-callsite-explosion.md).

The UT transform's fix lives here rather than in the kernel because it is pure
algebra and is worth proving on CPU first: see
:func:`unit_triangular_inverse`. The chunk loop's fix is structural -- it must
become an ``nl.fori_loop`` inside the kernel -- and belongs to Phase 5.

Two Neuron lowering hazards are avoided in the code below rather than noted:
``Tensor.split`` with a list of sizes is divergence #1 (silently wrong data on
any dim but 0), so the q/k/v separation is explicit slicing; and the conv state
is read as a fixed-size window plus a validity mask rather than a
Python-int-length slice, which is what nine consecutive DeepSeek-V4 Dynamo
blockers were all made of.
"""

import torch
import torch.nn.functional as F
from torch import nn

from vllm_neuron.utils.weight_loader import set_weight_loader

from .attention import Qwen3_5RMSNormGated
from .config import Qwen3_5TextConfig
from .parallel import Qwen3_5ShardingPolicy, resolve_sharding, resolve_tp_context
from .weight_loaders import (
    gdn_conv1d_weight_loader,
    gdn_gated_norm_loader,
    gdn_head_vector_loader,
    gdn_out_proj_weight_loader,
    gdn_qkv_weight_loader,
    gdn_row_weight_loader,
)

# The convolution lives in nki_gdn.py alongside its NKI dispatcher, so the
# kernel and its torch reference travel together -- the arrangement nki_mla.py
# uses for its own oracle. Re-exported because this module is the layer's
# public face.
from .nki_gdn import (
    can_use_chunk_scan_kernel,
    causal_conv1d,
    causal_conv1d_with_state,
    chunk_gated_delta_rule_nki,
)

__all__ = [
    "Qwen3_5GatedDeltaNet",
    "causal_conv1d",
    "causal_conv1d_with_state",
    "chunk_gated_delta_rule",
    "l2norm",
    "recurrent_gated_delta_rule",
    "unit_triangular_inverse",
]


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Match the FLA library's l2norm, which the reference calls in-kernel."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


# ===========================================================================
# The UT transform, without the sequential loop
# ===========================================================================


def _block_substitution_masks(
    n: int, dtype: torch.dtype, device: torch.device
) -> list[torch.Tensor]:
    """One constant mask per level of the block substitution.

    Level ``b`` merges adjacent ``b``-sized diagonal blocks into ``2b`` ones, so
    its mask selects exactly the lower off-diagonal quadrant of each ``2b``
    block: same ``2b`` block, different ``b`` half, below the diagonal.

    ``log2(n)`` masks, all compile-time constants.
    """
    idx = torch.arange(n, device=device)
    row, col = idx[:, None], idx[None, :]
    masks = []
    b = 1
    while b < n:
        same_pair = row.div(2 * b, rounding_mode="floor") == col.div(
            2 * b, rounding_mode="floor"
        )
        other_half = row.div(b, rounding_mode="floor") != col.div(
            b, rounding_mode="floor"
        )
        masks.append((same_pair & other_half & (row > col)).to(dtype))
        b *= 2
    return masks


def unit_triangular_inverse(a: torch.Tensor) -> torch.Tensor:
    """Invert ``I - A`` for strictly lower-triangular ``A``, without a scan.

    The reference does this by forward substitution:

        for i in range(1, chunk_size):
            row = attn[..., i, :i]
            attn[..., i, :i] = row + (row.unsqueeze(-1) * attn[..., :i, :i]).sum(-2)

    which is ``chunk_size - 1`` sequential, data-dependent, in-place slice
    writes -- 63 of them at the default chunk size. Under XLA that unrolls into
    63 op groups *per chunk per head per layer*, so it cannot be used as-is.

    **Do not replace this with the Neumann/binary-powering identity.** ``A`` is
    nilpotent, so ``(I - A)^-1 == I + A + ... + A^(n-1) ==
    (I + A)(I + A^2)(I + A^4)...(I + A^(n/2))``, which is algebraically exact and
    needs only ``2*log2(n)`` matmuls. It is also catastrophically unstable for
    the matrices this layer actually produces, and it was the cause of a
    whole-model NaN.

    Here ``A = -(k_beta @ key^T) * decay_mask`` with l2-normalized keys, so when
    successive keys are near-identical -- which bucket padding guarantees, since
    every padded row carries the same token -- ``k . k == 1`` and ``A`` is close
    to ``-beta`` times the strictly-lower-triangular ones matrix. Its powers
    then *grow*: at ``n = 64`` and ``beta = 0.99``, ``max|A^32| = 3.4e17`` while
    the true inverse is bounded by 1. The sum is pure cancellation, and fp32
    keeps none of it -- measured error 6.9e10 against a bounded-by-1 answer.
    fp64 does not rescue it either (error 64). Random test matrices hide this
    completely: entries of 0.1 give ``max|A^32| = 5e-24``.

    So instead: recursive 2x2 block substitution. For ``M = I - A`` split into
    two ``b``-sized halves,

        M = [[M11, 0  ]        M^-1 = [[X11,         0  ]
             [M21, M22]]               [X22 A21 X11, X22]]

    If ``x`` already holds the inverses of the ``b``-sized diagonal blocks (so it
    is block diagonal), then ``x @ a @ x`` restricted to a ``2b`` block's lower
    quadrant is exactly ``X22 @ A21 @ X11``. One masked update per level and
    ``log2(n)`` levels -- 12 matmuls at ``n = 64`` against the binary form's 10,
    with the same static shapes and compile-time-constant bound.

    No power of ``A`` is ever formed. Every factor is either a true inverse or a
    sub-block of ``A``, so there is nothing to cancel: measured error 3e-8 on the
    same matrices, and exact at ``beta = 1``.

    Args:
        a: ``[..., n, n]``, strictly lower triangular.

    Returns:
        ``(I - a)^-1``, unit lower triangular.
    """
    n = a.shape[-1]
    eye = torch.eye(n, dtype=a.dtype, device=a.device)

    # Seed: every 1x1 diagonal block of a unit triangular matrix inverts to 1.
    result = eye.expand_as(a).clone()
    for mask in _block_substitution_masks(n, a.dtype, a.device):
        result = result + mask * (result @ a @ result)
    return result


# ===========================================================================
# Delta rule
# ===========================================================================


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked gated delta rule.

    Inputs are ``[B, T, H, D]`` (``g``/``beta`` are ``[B, T, H]``), matching the
    reference's calling convention. Returns ``(output, final_state)`` with
    output ``[B, T, H, Dv]`` and state ``[B, H, Dk, Dv]``.

    All internal arithmetic is fp32; the recurrent state is an accumulator and
    bf16 drift compounds across chunks.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # [B, T, H, D] -> [B, H, T, D]
    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    )

    batch, heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))
    total = seq_len + pad

    query = query * (k_dim**-0.5)
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    def to_chunks(x):
        return x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])

    query, key, value, k_beta, v_beta = (
        to_chunks(x) for x in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(batch, heads, -1, chunk_size)

    # Intra-chunk log-decay, cumulative along the chunk.
    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float().tril()

    strict_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    a = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(strict_upper, 0)
    attn = unit_triangular_inverse(a)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = (
        torch.zeros(batch, heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    causal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    core_out = torch.zeros_like(value)

    # NOTE: this loop is the reason this module is an oracle and not the device
    # path. On device it becomes an nl.fori_loop inside the NKI kernel.
    for i in range(total // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(
            causal_mask, 0
        )
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        core_out[:, :, i] = attn_inter + attn_i @ v_new
        g_last = g[:, :, i, -1, None]
        state = state * g_last[..., None].exp() + (
            k_i * (g_last - g[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new

    core_out = core_out.reshape(batch, heads, -1, v_dim)[:, :, :seq_len]
    return core_out.transpose(1, 2).contiguous().to(initial_dtype), state


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step recurrence, used for decode. Inputs ``[B, T, H, D]``.

    For ``T == 1`` this is one step and the Python loop below has one iteration,
    so it costs nothing under tracing. It is written generally only so the
    oracle can validate multi-token decode against the chunked path.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    )

    batch, heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    query = query * (k_dim**-0.5)

    state = (
        torch.zeros(batch, heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_out = torch.zeros(
        batch, heads, seq_len, v_dim, dtype=value.dtype, device=value.device
    )

    for t in range(seq_len):
        q_t, k_t, v_t = query[:, :, t], key[:, :, t], value[:, :, t]
        g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, t].unsqueeze(-1)

        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    return core_out.transpose(1, 2).contiguous().to(initial_dtype), state


# ===========================================================================
# Module
# ===========================================================================


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer (CPU oracle form).

    Qwen3.5 keeps the four input projections **separate** (``in_proj_qkv``,
    ``in_proj_z``, ``in_proj_b``, ``in_proj_a``), unlike Qwen3-Next which fuses
    them into qkvz/ba pairs. That makes weight loading straightforward and is
    why upstream constructs this layer with ``gqa_interleaved_layout=False``.
    """

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_idx: int,
        policy: Qwen3_5ShardingPolicy | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.dtype = config.torch_dtype

        tp = resolve_tp_context()
        self.tp_group = tp.group
        self.world_size = tp.world_size
        self.rank = tp.rank
        # Tests construct this layer directly with no process group, where the
        # context degrades to a single rank and the default policy is the
        # unsharded one. The engine always passes the model's resolved policy.
        self.policy = (
            policy if policy is not None else resolve_sharding(config, tp.world_size)
        )

        self.hidden_size = config.hidden_size
        self.num_k_heads = self.policy.k_heads_per_rank
        self.num_v_heads = self.policy.v_heads_per_rank
        self.head_k_dim = config.linear_key_head_dim
        #: This rank's slice of each value head. Equals the full
        #: ``linear_value_head_dim`` unless tp exceeds the 16 key heads, at
        #: which point the value dimension itself is split.
        self.head_v_dim = self.policy.v_dim_per_rank
        self.num_v_per_k = config.num_v_per_k
        self.conv_kernel_size = config.linear_conv_kernel_dim

        #: Chunk width for the prefill delta rule. 64 is the reference default
        #: and the widest the NKI scan accepts while leaving room for a
        #: [chunk, chunk] tile alongside the state on one SBUF partition set.
        #: Must stay in the kernel's accepted set (16/32/64/128).
        self.chunk_size = 64

        # Per-rank widths. conv_dim is *not* the global conv_dim // tp: under
        # value-dimension splitting the q and k blocks are replicated across
        # the ranks sharing a key head. See Qwen3_5ShardingPolicy.
        self.key_dim = self.policy.key_dim_per_rank
        self.value_dim = self.policy.value_dim_per_rank
        self.conv_dim = self.policy.conv_dim_per_rank

        # Every projection is built at config.torch_dtype, like Qwen3_5MLP and
        # Qwen3_5Attention. Omitting it leaves them at torch's fp32 default,
        # which on device makes the depthwise conv fail to compile with
        # "nc_matmul: if one input is tfloat32/float32, both must be. Got
        # stationary=bfloat16, moving=float32" -- the loaded bf16 filter
        # against an fp32 activation. dt_bias and A_log stay fp32 on purpose:
        # they feed an exp/softplus the layer evaluates in fp32.
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.conv_dim, bias=False, dtype=self.dtype
        )
        self.in_proj_z = nn.Linear(
            self.hidden_size, self.value_dim, bias=False, dtype=self.dtype
        )
        self.in_proj_b = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False, dtype=self.dtype
        )
        self.in_proj_a = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False, dtype=self.dtype
        )
        self.out_proj = nn.Linear(
            self.value_dim, self.hidden_size, bias=False, dtype=self.dtype
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=self.conv_kernel_size - 1,
            dtype=self.dtype,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))

        self.norm = Qwen3_5RMSNormGated(
            self.head_v_dim, config.rms_norm_eps, config.torch_dtype
        )

        # Global value-head columns this rank owns, used to place its partial
        # sums of squares in the cross-rank gated-norm reduction.
        self.global_v_heads = self.policy.v_head_indices(self.rank)
        self.needs_norm_allreduce = (
            self.policy.gated_norm_needs_allreduce and self.world_size > 1
        )

        self._install_weight_loaders()

        # Bound by Qwen3_5TextForCausalLM.bind_kv_cache. Both are indexed by
        # request, not by token: [num_requests, ...].
        self.conv_state_cache = None
        self.recurrent_state_cache = None

    def _install_weight_loaders(self):
        """Attach the per-rank shard transforms.

        All six loaders are rank-generic -- they take the rank as an
        argument like
        ``gated_qkv_weight_loader`` does -- so a single policy drives every
        rank's slice and the partition is never restated here.
        """
        policy = self.policy
        set_weight_loader(self.in_proj_qkv.weight, gdn_qkv_weight_loader(policy))
        set_weight_loader(self.conv1d.weight, gdn_conv1d_weight_loader(policy))
        set_weight_loader(self.in_proj_z.weight, gdn_row_weight_loader(policy))
        set_weight_loader(self.out_proj.weight, gdn_out_proj_weight_loader(policy))
        set_weight_loader(self.in_proj_b.weight, gdn_head_vector_loader(policy))
        set_weight_loader(self.in_proj_a.weight, gdn_head_vector_loader(policy))
        set_weight_loader(self.dt_bias, gdn_head_vector_loader(policy))
        set_weight_loader(self.A_log, gdn_head_vector_loader(policy))
        set_weight_loader(self.norm.weight, gdn_gated_norm_loader(policy))

    # -- pieces, exposed so tests can diff them individually ---------------

    def split_mixed_qkv(self, mixed: torch.Tensor):
        """Separate ``[q | k | v]`` from the conv output.

        Explicit slicing, never ``Tensor.split([key_dim, key_dim, value_dim],
        dim=-1)``: a list-of-sizes split on any dim but 0 is divergence #1 on
        Neuron and returns silently wrong data
        (docs/model-dev/neuron-cpu-op-divergences.md, reproducer at
        tools/repro_neuron_split_lowering.py).
        """
        k_end = self.key_dim
        v_start = 2 * self.key_dim
        return mixed[..., :k_end], mixed[..., k_end:v_start], mixed[..., v_start:]

    def head_sum_squares(self, core_out_flat: torch.Tensor):
        """``(sum_squares, dim_size)`` for the gated norm, or ``(None, None)``.

        The gated RMSNorm normalizes over the *full* ``linear_value_head_dim``.
        When tp exceeds the 16 key heads each rank holds only part of that
        width, so its local sum of squares is a fraction of the true one and
        normalizing with it would scale every shard differently -- silently,
        and differently per rank.
        """
        if not self.needs_norm_allreduce:
            return None, None
        local = core_out_flat.float().pow(2).sum(-1).reshape(-1, self.num_v_heads)
        totals = self._all_reduce_head_sums(local).reshape(-1, 1)
        return totals, self.policy.value_head_dim

    def _all_reduce_head_sums(self, local_sums: torch.Tensor) -> torch.Tensor:
        """Sum each value head's partial squares across the ranks sharing it.

        The reduction runs over the **whole** TP group on a
        ``[tokens, num_v_heads]`` buffer in which each rank writes only its own
        head columns. Ranks holding different key heads write disjoint columns
        and cannot interfere; ranks sharing a key head write the same columns
        and their partials add to the true full-width sum. Using the full group
        avoids creating ``v_dim_shards``-sized subgroups, which would have to
        be built collectively and exactly once across all 48 GDN layers, and
        costs one [tokens, 48] fp32 all-reduce per layer.

        Overridable so a single-process test can accumulate across simulated
        shards without a production-only injection seam.
        """
        cols = torch.tensor(
            self.global_v_heads, dtype=torch.long, device=local_sums.device
        )
        buffer = torch.zeros(
            local_sums.shape[0],
            self.policy.num_v_heads,
            dtype=local_sums.dtype,
            device=local_sums.device,
        )
        buffer = buffer.index_copy(1, cols, local_sums)
        reduced = self.tp_group.all_reduce(buffer)
        if reduced is not None:
            buffer = reduced
        return buffer.index_select(1, cols)

    def gates(self, hidden_states: torch.Tensor):
        """``beta`` and the log-decay ``g``.

        ``A_log`` is exponentiated in fp32: in fp16 the reference notes ``A``
        can otherwise become ``-inf``.
        """
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        return beta, g

    def mask_padding(
        self,
        beta: torch.Tensor,
        g: torch.Tensor,
        valid_len: torch.Tensor | None,
    ):
        """Neutralise padded rows in the delta rule.

        Full attention tolerates bucket padding because a padded key is simply
        a column the causal mask discards. A recurrence has no such mask: every
        row it walks over mutates the state that decode later resumes from, so
        the ~2000 padding rows of a short prompt in a 2048 bucket are *carried*
        rather than ignored. Worse, padding rows repeat one token, so their
        l2-normalised keys are near-identical and the delta rule's update is at
        its most aggressive exactly where the data is meaningless.

        Two quantities have to be neutralised, and both, not either:

        * ``beta = 0`` zeroes ``k_beta`` and ``v_beta``, so the row adds nothing
          to the state;
        * ``g = 0`` makes ``exp(g) == 1``, so the row does not *decay* the state
          it passes through either. Masking only ``beta`` leaves 2000 steps of
          decay applied to a state that should have stopped evolving, which is
          the subtler half and produces plausible-looking output.

        Multiplying by a mask rather than indexing keeps the shape static.
        """
        if valid_len is None:
            return beta, g
        keep = (
            torch.arange(beta.shape[1], device=beta.device).reshape(1, -1)
            < valid_len.to(torch.long).reshape(-1, 1)
        ).unsqueeze(-1)
        return beta * keep.to(beta.dtype), g * keep.to(g.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
        use_recurrent: bool = False,
        valid_len: torch.Tensor | None = None,
    ):
        """Args:
            hidden_states: ``[B, T, hidden]``.
            conv_state: ``[B, conv_dim, kernel - 1]`` or None for a fresh sequence.
            recurrent_state: ``[B, H, Dk, Dv]`` or None.
            use_recurrent: take the single-step path (decode) rather than chunked.
            valid_len: ``[B]`` count of leading real tokens per row, or None
                when the caller guarantees every row is real. See
                ``mask_padding`` for why a recurrent layer cannot ignore this.

        Returns:
            ``(output, new_conv_state, new_recurrent_state)``.
        """
        batch, seq_len, _ = hidden_states.shape

        mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [B, conv_dim, T]

        if conv_state is None:
            conv_state = torch.zeros(
                batch,
                self.conv_dim,
                self.conv_kernel_size - 1,
                dtype=mixed.dtype,
                device=mixed.device,
            )
        mixed, new_conv_state = causal_conv1d_with_state(
            mixed,
            conv_state,
            self.conv1d.weight.squeeze(1),
            None,
            "silu",
            valid_len=valid_len,
        )

        mixed = mixed.transpose(1, 2)  # [B, T, conv_dim]
        query, key, value = self.split_mixed_qkv(mixed)

        query = query.reshape(batch, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch, seq_len, -1, self.head_v_dim)

        beta, g = self.gates(hidden_states)
        beta, g = self.mask_padding(beta, g, valid_len)

        if self.num_v_per_k > 1:
            query = query.repeat_interleave(self.num_v_per_k, dim=2)
            key = key.repeat_interleave(self.num_v_per_k, dim=2)

        # Decode is one token, so its recurrence has a single iteration and
        # costs nothing to trace. Prefill takes the NKI scan when it is
        # available, and the torch chunk rule otherwise -- the latter stays the
        # oracle the kernel is diffed against, never a silent second
        # implementation.
        if use_recurrent:
            core_out, new_recurrent_state = recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state
            )
        elif can_use_chunk_scan_kernel(query, self.chunk_size):
            core_out, new_recurrent_state = chunk_gated_delta_rule_nki(
                query,
                key,
                value,
                g=g,
                beta=beta,
                chunk_size=self.chunk_size,
                initial_state=recurrent_state,
            )
        else:
            core_out, new_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                chunk_size=self.chunk_size,
                initial_state=recurrent_state,
            )

        z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
        core_out = core_out.reshape(-1, self.head_v_dim)
        sum_squares, dim_size = self.head_sum_squares(core_out)
        core_out = self.norm(core_out, z, sum_squares=sum_squares, dim_size=dim_size)
        core_out = core_out.reshape(batch, seq_len, -1)

        return self.out_proj(core_out), new_conv_state, new_recurrent_state

    # ------------------------------------------------------------------
    # Paged entry point
    # ------------------------------------------------------------------

    def state_index(self, attn_metadata) -> torch.Tensor:
        """Per-request state slot.

        The Mamba group's block table is a single column -- one page per
        request, because ``mamba_block_size`` defaults to ``max_model_len`` --
        so column 0 *is* the state index. ``slot_mapping`` does not apply to
        this group; it addresses tokens, and this state is not per token.
        Matches upstream's ``mamba_get_block_table_tensor``.
        """
        meta = attn_metadata[f"layers.{self.layer_idx}.linear_attn"]
        return meta["block_table_tensor"][:, 0].to(torch.long)

    def valid_len(self, meta, rows_per_request: int) -> torch.Tensor:
        """``[B]`` real-token count per request row, defaulting to all real.

        The runner pads each request's token block up to the bucket, so this is
        the only signal that separates prompt from filler. It is absent from
        metadata built by callers that never pad -- the oracle tests and the
        older paged-state tests -- and those genuinely have no padding, so the
        default is a full row rather than an error.
        """
        counts = meta.get("num_valid_tokens")
        if counts is None:
            return None
        return counts.reshape(-1).to(torch.long).clamp(max=rows_per_request)

    def forward_paged(self, hidden_states, attn_metadata, is_decode: bool):
        """Read state for the batch, run the layer, write the state back.

        The "is this a fresh sequence?" decision is a **mask**, never a Python
        branch: a zero-valued state is exactly the correct representation of no
        history for both the conv window (the reference left-pads with zeros)
        and the recurrent accumulator. Branching on ``cached_seq_len`` instead
        is what produced nine consecutive DeepSeek-V4 Dynamo blockers.
        """
        if self.conv_state_cache is None or self.recurrent_state_cache is None:
            raise RuntimeError(
                f"layer {self.layer_idx}: GDN state cache was never bound; "
                "bind_kv_cache must run before the first forward"
            )

        # Sequence parallelism: on prefill the residual stream arrives
        # scattered along tokens, and this layer *cannot* work on a slice --
        # the delta rule is a token-ordered recurrence, so a rank holding a
        # subset of the sequence computes a different result, not a partial
        # one. Gather first, exactly as Qwen3_5MLP and Qwen3_5Attention do.
        if not is_decode and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        meta = attn_metadata[f"layers.{self.layer_idx}.linear_attn"]
        index = self.state_index(attn_metadata)

        batch = index.shape[0]
        tokens = hidden_states.shape[0]
        if tokens % batch:
            raise ValueError(
                f"layer {self.layer_idx}: {tokens} tokens do not divide evenly "
                f"across {batch} requests"
            )
        x = hidden_states.view(batch, tokens // batch, -1)
        valid_len = self.valid_len(meta, x.shape[1])

        conv_state = self.conv_state_cache[index]
        recurrent_state = self.recurrent_state_cache[index]

        cached = meta.get("cached_seq_len")
        if cached is not None:
            fresh = cached.reshape(-1).to(torch.int64) > 0
            # Cast the mask to *each* state's own dtype rather than building
            # one mask and reusing it. get_kv_spec stores both states at the
            # same dtype today, but a mask built at one state's dtype and
            # multiplied into the other silently promotes it whenever they
            # ever differ, and that promotion survives CPU (torch just
            # promotes) only to fail on device, where the promoted conv state
            # reaches the depthwise conv through torch.cat and the kernel
            # rejects the pair: "nc_matmul: if one input is tfloat32/float32,
            # both must be. Got stationary=bfloat16, moving=float32". Deriving
            # each mask from its own state keeps that impossible by
            # construction.
            conv_keep = fresh.to(conv_state.dtype)
            recurrent_keep = fresh.to(recurrent_state.dtype)
            # Broadcast the mask over each state's trailing dims.
            conv_state = conv_state * conv_keep.reshape(
                -1, *([1] * (conv_state.dim() - 1))
            )
            recurrent_state = recurrent_state * recurrent_keep.reshape(
                -1, *([1] * (recurrent_state.dim() - 1))
            )

        output, new_conv_state, new_recurrent_state = self.forward(
            x,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            use_recurrent=is_decode,
            valid_len=valid_len,
        )

        self.conv_state_cache.index_copy_(
            0, index, new_conv_state.to(self.conv_state_cache.dtype)
        )
        self.recurrent_state_cache.index_copy_(
            0, index, new_recurrent_state.to(self.recurrent_state_cache.dtype)
        )

        output = output.reshape(tokens, -1)

        # out_proj is column-sharded, so each rank holds a partial sum over the
        # value dimension: reduce on the way out, scattering back to this
        # rank's token slice on prefill.
        if self.world_size > 1:
            if is_decode:
                # Assign -- see the note in Qwen3_5MLP.forward. This module
                # already assigns the gated-norm all_reduce a few hundred lines
                # up; the two are now consistent.
                output = self.tp_group.all_reduce(output)
            else:
                output = self.tp_group.reduce_scatter(output, dim=0)
        return output
