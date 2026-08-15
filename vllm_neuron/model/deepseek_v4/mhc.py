# SPDX-License-Identifier: Apache-2.0
"""Decomposed multi-head residual (mHC) reference math."""

import torch


def sinkhorn(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    x = logits.float()
    for _ in range(iterations):
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)
        x = x - torch.logsumexp(x, dim=-2, keepdim=True)
    return x.exp().to(logits.dtype)


def mix_residual(residual: torch.Tensor, update: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Mix widened residual streams with a doubly-stochastic matrix."""
    if residual.shape != update.shape or residual.shape[-2] != logits.shape[-1]:
        raise ValueError("mHC stream dimensions do not agree")
    matrix = sinkhorn(logits)
    return torch.einsum("...ij,...jd->...id", matrix, residual) + update

