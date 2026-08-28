# SPDX-License-Identifier: Apache-2.0
"""Exact portable routing primitives for DeepSeek-V4 MoE."""

import torch
import torch.nn.functional as F


def dense_expert_affinities(
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Aggregate every selected routing slot into ``[tokens, experts]``."""
    if expert_ids.shape != expert_weights.shape or expert_ids.ndim != 2:
        raise ValueError("expert ids and weights must have the same 2-D shape")
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    valid = (expert_ids >= 0) & (expert_ids < num_experts)
    safe_ids = torch.where(valid, expert_ids, torch.zeros_like(expert_ids)).long()
    affinities = expert_weights.new_zeros((expert_ids.shape[0], num_experts))
    return affinities.scatter_add(
        1, safe_ids, torch.where(valid, expert_weights, torch.zeros_like(expert_weights))
    )


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
    # Torch CPU returns a ``torch.return_types.topk`` named tuple while the
    # Torch-XLA 2.9 bridge returns a plain two-element list. Positional access
    # is part of both contracts and keeps this portable reference path usable
    # for the Trn2 component gate.
    ids = torch.topk(gates + correction_bias.float(), topk, dim=-1)[1]
    weights = torch.gather(gates, -1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids, (weights * routed_scaling_factor).to(logits.dtype)


def hash_experts(input_ids: torch.Tensor, tid2eid: torch.Tensor) -> torch.Tensor:
    # Reading a tensor's *values* to decide control flow is untraceable:
    # Dynamo raises "Data-dependent branching" rather than graph-breaking, so
    # this bound check has to be eager-only. It stays in eager because that is
    # where a bad `tid2eid` is diagnosable at all -- on device the same input
    # would silently gather garbage. Shape checks are safe either way and are
    # deliberately left outside this guard.
    if not torch.compiler.is_compiling():
        if input_ids.min() < 0 or input_ids.max() >= tid2eid.shape[0]:
            raise ValueError("token id is outside tid2eid")
    return tid2eid[input_ids]


def hash_topk(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    routed_scaling_factor: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hash-selected expert ids with learned gate weights from unmodified scores."""
    if logits.shape[-1] <= 0 or logits.numel() // logits.shape[-1] != input_ids.numel():
        raise ValueError("hash routing logits and token ids are not aligned")
    ids = hash_experts(input_ids.reshape(-1), tid2eid).long()
    if ids.ndim != 2 or ids.shape[0] != input_ids.numel():
        raise ValueError("tid2eid must provide a fixed top-k row for every token")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    # Eager-only for the same reason as hash_experts' bound check.
    if not torch.compiler.is_compiling():
        if ids.numel() and ids.max() >= flat_logits.shape[-1]:
            raise ValueError("tid2eid selects an expert outside the gate logits")
    scores = F.softplus(flat_logits.float()).sqrt()
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return ids, (weights * routed_scaling_factor).to(logits.dtype)
