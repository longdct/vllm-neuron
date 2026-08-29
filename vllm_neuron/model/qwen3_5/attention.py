# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 full-attention numerics
===============================

The 16 ``full_attention`` layers of the 3:1 hybrid stack. Everything here is
device-agnostic tensor math with no cache plumbing, so it can be diffed
per-element against ``transformers.models.qwen3_5.modeling_qwen3_5`` on CPU
before any of it reaches a NeuronCore.

Four things differ from the dense Qwen3 in ``vllm_neuron/model/qwen3`` and each
one fails *silently* if ported by eye:

1. **Two RMSNorm conventions in one model.** HF's ``Qwen3_5RMSNorm`` stores
   ``weight`` initialized to **zeros** and computes ``x_norm * (1.0 + weight)``,
   while ``Qwen3_5RMSNormGated`` inside the Gated DeltaNet uses the ordinary
   ``weight * x_norm``. This module implements the *ordinary* form for both and
   folds the ``+1`` into the checkpoint tensor at load time (see
   ``weight_loaders.py``), so the runtime graph and the decode kernel's gamma
   argument stay in one convention.

2. **Partial RoPE rotates the leading channels.** ``partial_rotary_factor`` 0.25
   rotates the first 64 of 256 head channels, pairing ``i`` with ``i + 32``
   inside that slice; the remaining 192 pass through untouched. Note this is the
   *opposite end* of the head from ``deepseek_v4/attention.py::apply_partial_rotary``,
   which rotates the trailing channels with interleaved pairing -- that helper
   is not reusable here.

3. **The query projection carries a gate.** ``q_proj`` emits
   ``num_heads * head_dim * 2`` and each head is ``[query | gate]`` side by
   side, so the gate is interleaved *per head*, not appended in a block. The
   gate multiplies the attention output before ``o_proj``.

4. **head_dim is 256**, which exceeds the 128-element SBUF partition bound of
   the segmented-attention kernel. ``NF.segmented_attention`` raises above 128,
   so it must never be called from here; ``NF.flash_attention`` merely falls
   back to torch and is safe.
"""

import torch
from torch import nn

from .config import Qwen3_5TextConfig


# ===========================================================================
# Normalization
# ===========================================================================


class Qwen3_5RMSNorm(nn.Module):
    """RMS normalization in the ordinary ``weight * x_norm`` form.

    HF computes ``x_norm * (1.0 + weight)`` with a zero-initialized weight. The
    ``+1`` is folded into the tensor when the checkpoint is loaded, so a weight
    of 1.0 here is HF's 0.0. ``default_weight`` reflects that: a freshly
    constructed module is the identity, matching a zero-filled HF checkpoint.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class Qwen3_5RMSNormGated(nn.Module):
    """Gated RMS norm used by the Gated DeltaNet output.

    Unlike :class:`Qwen3_5RMSNorm` this one matches HF exactly with no fold:
    HF's ``Qwen3_5RMSNormGated`` initializes ``weight`` to **ones** and applies
    ``weight * x_norm``. Normalization happens *before* the gate.

    ``sum_squares`` may be supplied when the value dimension is sharded across
    ranks (tp > number of key heads), in which case the caller is responsible
    for having all-reduced it -- normalizing over a partial vector would
    otherwise silently scale each shard differently.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
        sum_squares: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if sum_squares is None:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
        else:
            # Value dimension is split across ranks; the caller all-reduced the
            # sum of squares over the full head width.
            variance = sum_squares / dim_size

        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * torch.nn.functional.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ===========================================================================
# Rotary embedding: partial rotation + interleaved mRoPE
# ===========================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split-half rotation, applied *within* the rotary slice."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """Rotate the **leading** ``rotary_dim`` channels; pass the rest through.

    ``cos``/``sin`` are ``[..., rotary_dim]`` (already doubled from the
    ``rotary_dim // 2`` frequency band by the caller).

    The recombination uses ``torch.index_copy`` rather than
    ``torch.cat([rotated, passthrough])`` deliberately. Neuron lowered a small
    rotary segment in a rank-4 ``cat`` as a dead/zero operand -- divergence #2 in
    docs/model-dev/neuron-cpu-op-divergences.md, fixed the same way in
    ``deepseek_v4/attention.py`` (commit 15e548c). That divergence did *not*
    reproduce at rank 3, so the shape it is used at here matters.
    """
    if rotary_dim == x.shape[-1]:
        return x * cos + rotate_half(x) * sin

    x_rot = x[..., :rotary_dim]
    rotated = x_rot * cos + rotate_half(x_rot) * sin

    indices = torch.arange(rotary_dim, device=x.device)
    return torch.index_copy(x, -1, indices, rotated.to(x.dtype))


def compute_interleaved_mrope(
    freqs: torch.Tensor, mrope_section: list[int]
) -> torch.Tensor:
    """Reorder chunked [TTT..HHH..WWW] frequencies into interleaved [THWTHW..].

    Adapted from ``qwen3_vl/model_bf16.py::compute_interleaved_mrope``, which
    uses ``torch.where`` rather than in-place slice assignment for XLA
    compatibility. The only difference here is that the band is
    ``rotary_dim // 2`` wide rather than ``head_dim // 2``, because Qwen3.5
    rotates only part of the head.

    Args:
        freqs: ``[3, ..., rotary_dim // 2]`` per-axis frequencies.
        mrope_section: per-axis widths, summing to ``rotary_dim // 2``.
    """
    last_dim = freqs.shape[-1]
    indices = torch.arange(last_dim, device=freqs.device, dtype=torch.int64)

    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        mask = (indices % 3 == offset) & (indices < length)
        freqs_t = torch.where(mask, freqs[dim], freqs_t)

    return freqs_t


class Qwen3_5RotaryEmbedding(nn.Module):
    """Partial rotary embedding with interleaved mRoPE.

    Emits ``cos``/``sin`` of width ``rotary_dim // 2``; the apply step doubles
    them via ``cat((x, x))`` in the usual way.
    """

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.rotary_dim = config.rotary_dim
        self.mrope_section = config.mrope_section

        # Build on CPU: Neuron rejects a fused ``.to(device=..., dtype=...)``,
        # and stock RoPE init hits exactly that (divergence #8).
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device="cpu")
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
        position_ids: ``[T]`` text-only, or ``[3, T]`` / ``[3, bs, T]`` mRoPE.

        Returns:
            ``(cos, sin)``, each ``[T, rotary_dim // 2]``.
        """
        if position_ids.ndim == 1:
            position_ids = position_ids[None, None, :].expand(3, 1, -1)
        elif position_ids.ndim == 2 and position_ids.shape[0] == 3:
            position_ids = position_ids.unsqueeze(1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = compute_interleaved_mrope(freqs, self.mrope_section)

        cos = freqs.cos()
        sin = freqs.sin()

        if cos.shape[0] == 1:
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)

        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def double_freqs(cos: torch.Tensor, sin: torch.Tensor):
    """Widen a ``rotary_dim // 2`` band to the full ``rotary_dim``."""
    return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)


# ===========================================================================
# Attention core
# ===========================================================================


def split_query_and_gate(
    qg: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate the per-head ``[query | gate]`` pairs.

    ``q_proj`` emits ``num_heads * head_dim * 2`` where **each head** stores its
    query and its gate adjacently. Viewing as ``[..., num_heads, 2 * head_dim]``
    and slicing the last dim is what makes the per-head interleaving explicit;
    a flat split of the projection into two halves would mix heads.

    Explicit slicing, not ``Tensor.split([a, b], dim=-1)``: a list-of-sizes
    split on any dim but 0 is divergence #1 on Neuron -- silently wrong data.
    """
    qg = qg.view(*qg.shape[:-1], num_heads, 2 * head_dim)
    query = qg[..., :head_dim]
    gate = qg[..., head_dim:]
    return query, gate


def apply_output_gate(
    attn_output: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """``attn_output * sigmoid(gate)``, applied before ``o_proj``.

    This is why the fused decode megakernel cannot be used as-is: it folds
    ``o_proj`` into the kernel, leaving nowhere to insert the gate. Passing
    ``W_out=None`` makes it return the pre-projection tensor instead.
    """
    return attn_output * torch.sigmoid(gate.to(attn_output.dtype))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads for GQA. ``[N_kv, T, D] -> [N_q, T, D]``."""
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=0)
