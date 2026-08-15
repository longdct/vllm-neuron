# SPDX-License-Identifier: Apache-2.0
"""Portable DeepSeek-V4 MLA reference operations used by T0 bring-up."""

from __future__ import annotations

import math

import torch


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply interleaved rotary embedding to the last dimension."""
    if x.shape[-1] % 2:
        raise ValueError("rotary dimension must be even")
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def mla_attention_reference(
    query: torch.Tensor,
    latent: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    attention_sinks: torch.Tensor | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Materialized fp32 MLA oracle supporting prefill and one-token decode.

    ``query`` is ``[B,T,H,Dq]``, ``latent`` is ``[B,S,L]``, projection weights
    are ``[H,L,Dq]`` and ``[H,L,Dv]``. Optional RoPE features are concatenated
    to Q/K after the latent projection. The implementation is deliberately
    straightforward and never used as a production kernel.
    """
    if latent.shape[-1] != key_weight.shape[1] or key_weight.shape[:2] != value_weight.shape[:2]:
        raise ValueError("latent/projection dimensions do not agree")
    q = query.float()
    k = torch.einsum("bsl,hld->bshd", latent.float(), key_weight.float())
    v = torch.einsum("bsl,hlv->bshv", latent.float(), value_weight.float())
    if (query_rope is None) != (key_rope is None):
        raise ValueError("query_rope and key_rope must be provided together")
    if query_rope is not None:
        q = torch.cat((q, query_rope.float()), dim=-1)
        k = torch.cat((k, key_rope.float()), dim=-1)
    scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    t, s = query.shape[1], latent.shape[1]
    if causal:
        # Decode (T=1) sees the whole supplied cache; prefill aligns its last
        # query with the last key when S >= T.
        qpos = torch.arange(s - t, s, device=scores.device)[:, None]
        kpos = torch.arange(s, device=scores.device)[None, :]
        allowed = kpos <= qpos
        if sliding_window is not None:
            allowed &= kpos > (qpos - sliding_window)
        scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    if attention_sinks is not None:
        if attention_sinks.shape != (query.shape[2],):
            raise ValueError("attention_sinks must have shape [num_heads]")
        sink = attention_sinks.float()[None, :, None, None].expand(
            query.shape[0], -1, t, 1
        )
        weights = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :s]
    else:
        weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bshv->bthv", weights, v).to(query.dtype)

