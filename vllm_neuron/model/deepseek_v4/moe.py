# SPDX-License-Identifier: Apache-2.0
"""Exact portable routing primitives for DeepSeek-V4 MoE."""

import torch
import torch.nn.functional as F


def routed_topk(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """noaux_tc routing: correction changes selection, never gate values."""
    if logits.shape[-1] != correction_bias.numel() or not 0 < topk <= logits.shape[-1]:
        raise ValueError("invalid expert routing dimensions")
    gates = F.softplus(logits.float()).sqrt()
    ids = torch.topk(gates + correction_bias.float(), topk, dim=-1).indices
    weights = torch.gather(gates, -1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids, (weights * routed_scaling_factor).to(logits.dtype)


def hash_experts(input_ids: torch.Tensor, tid2eid: torch.Tensor) -> torch.Tensor:
    if input_ids.min() < 0 or input_ids.max() >= tid2eid.shape[0]:
        raise ValueError("token id is outside tid2eid")
    return tid2eid[input_ids]
