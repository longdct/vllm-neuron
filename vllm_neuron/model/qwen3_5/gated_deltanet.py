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

from .attention import Qwen3_5RMSNormGated
from .config import Qwen3_5TextConfig

# The convolution lives in nki_gdn.py alongside its NKI dispatcher, so the
# kernel and its torch reference travel together -- the arrangement nki_mla.py
# uses for its own oracle. Re-exported because this module is the layer's
# public face.
from .nki_gdn import causal_conv1d, causal_conv1d_with_state

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


def unit_triangular_inverse(a: torch.Tensor) -> torch.Tensor:
    """Invert ``I - A`` for strictly lower-triangular ``A``, without a scan.

    The reference does this by forward substitution:

        for i in range(1, chunk_size):
            row = attn[..., i, :i]
            attn[..., i, :i] = row + (row.unsqueeze(-1) * attn[..., :i, :i]).sum(-2)

    which is ``chunk_size - 1`` sequential, data-dependent, in-place slice
    writes -- 63 of them at the default chunk size. Under XLA that unrolls into
    63 op groups *per chunk per head per layer*.

    ``A`` is strictly lower triangular, hence nilpotent with ``A^n == 0``, so

        (I - A)^-1  =  I + A + A^2 + ... + A^(n-1)
                    =  (I + A)(I + A^2)(I + A^4) ... (I + A^(n/2))

    by the binary expansion of the exponents. That is ``2 * log2(n)`` matmuls --
    10 rather than 63 at n = 64 -- with a loop bound that is a compile-time
    constant, so it unrolls to a fixed, small, shape-independent graph.

    Args:
        a: ``[..., n, n]``, strictly lower triangular.

    Returns:
        ``(I - a)^-1``, unit lower triangular.
    """
    n = a.shape[-1]
    eye = torch.eye(n, dtype=a.dtype, device=a.device)

    result = eye + a
    power = a
    span = 1
    while span * 2 < n:
        power = power @ power
        result = result @ (eye + power)
        span *= 2
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

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.dtype = config.torch_dtype

        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_v_per_k = config.num_v_per_k
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.conv_dim = config.conv_dim

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))

        self.norm = Qwen3_5RMSNormGated(
            self.head_v_dim, config.rms_norm_eps, config.torch_dtype
        )

        # Bound by Qwen3_5TextForCausalLM.bind_kv_cache. Both are indexed by
        # request, not by token: [num_requests, ...].
        self.conv_state_cache = None
        self.recurrent_state_cache = None

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

    def gates(self, hidden_states: torch.Tensor):
        """``beta`` and the log-decay ``g``.

        ``A_log`` is exponentiated in fp32: in fp16 the reference notes ``A``
        can otherwise become ``-inf``.
        """
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        return beta, g

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
        use_recurrent: bool = False,
    ):
        """Args:
            hidden_states: ``[B, T, hidden]``.
            conv_state: ``[B, conv_dim, kernel - 1]`` or None for a fresh sequence.
            recurrent_state: ``[B, H, Dk, Dv]`` or None.
            use_recurrent: take the single-step path (decode) rather than chunked.

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
            mixed, conv_state, self.conv1d.weight.squeeze(1), None, "silu"
        )

        mixed = mixed.transpose(1, 2)  # [B, T, conv_dim]
        query, key, value = self.split_mixed_qkv(mixed)

        query = query.reshape(batch, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch, seq_len, -1, self.head_v_dim)

        beta, g = self.gates(hidden_states)

        if self.num_v_per_k > 1:
            query = query.repeat_interleave(self.num_v_per_k, dim=2)
            key = key.repeat_interleave(self.num_v_per_k, dim=2)

        rule = recurrent_gated_delta_rule if use_recurrent else chunk_gated_delta_rule
        core_out, new_recurrent_state = rule(
            query, key, value, g=g, beta=beta, initial_state=recurrent_state
        )

        z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
        core_out = self.norm(core_out.reshape(-1, self.head_v_dim), z)
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

        conv_state = self.conv_state_cache[index]
        recurrent_state = self.recurrent_state_cache[index]

        cached = meta.get("cached_seq_len")
        if cached is not None:
            keep = (cached.reshape(-1).to(torch.int64) > 0).to(recurrent_state.dtype)
            # Broadcast the mask over each state's trailing dims.
            conv_state = conv_state * keep.reshape(-1, *([1] * (conv_state.dim() - 1)))
            recurrent_state = recurrent_state * keep.reshape(
                -1, *([1] * (recurrent_state.dim() - 1))
            )

        output, new_conv_state, new_recurrent_state = self.forward(
            x,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            use_recurrent=is_decode,
        )

        self.conv_state_cache.index_copy_(
            0, index, new_conv_state.to(self.conv_state_cache.dtype)
        )
        self.recurrent_state_cache.index_copy_(
            0, index, new_recurrent_state.to(self.recurrent_state_cache.dtype)
        )

        return output.reshape(tokens, -1)
