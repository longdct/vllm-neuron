# SPDX-License-Identifier: Apache-2.0
"""NKI prototype for 512-wide materialized latent attention."""

from __future__ import annotations

import math

import torch
from nkilib.core.attention.attention_cte import attention_cte


def torch_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(
        query.shape[-1]
    )
    if causal:
        q_len, kv_len = query.shape[-2], key.shape[-2]
        qpos = torch.arange(kv_len - q_len, kv_len, device=query.device)[:, None]
        kpos = torch.arange(kv_len, device=query.device)[None, :]
        scores = scores.masked_fill(~(kpos <= qpos)[None], float("-inf"))
    return torch.matmul(scores.softmax(-1), value.float()).to(query.dtype)


def simulate_512_mla(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Execute nkilib's four-tile 512-d attention through the CPU simulator."""
    if query.shape[-1] != 512 or key.shape[-1] != 512 or value.shape[-1] != 512:
        raise ValueError("P2 NKI prototype requires a 512-wide Q/K/V materialization")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key and value must have shape [batch, sequence, 512]")
    if key.shape != value.shape or query.shape[0] != key.shape[0]:
        raise ValueError("NKI MLA batch and context dimensions do not agree")
    from vllm_neuron.nki.nki_cpu_sim import simulate_nki_kernel

    return simulate_nki_kernel(
        attention_cte,
        1,
        {
            "q": query.contiguous(),
            "k": key.transpose(1, 2).contiguous(),
            "v": value.contiguous(),
            "scale": 1 / math.sqrt(512),
            "causal_mask": causal,
            "tp_q": True,
            "tp_k": False,
            "tp_out": False,
        },
    )
