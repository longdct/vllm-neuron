# SPDX-License-Identifier: Apache-2.0
"""Decomposed multi-head residual (mHC) reference math."""

import torch
import torch.nn.functional as F


def sinkhorn(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    x = logits.float()
    for _ in range(iterations):
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)
        x = x - torch.logsumexp(x, dim=-2, keepdim=True)
    return x.exp().to(logits.dtype)


def sinkhorn_positive(
    matrix: torch.Tensor, iterations: int = 20, eps: float = 1e-6
) -> torch.Tensor:
    """Transformers-compatible projection from an already-positive matrix."""
    if iterations < 1 or eps <= 0:
        raise ValueError("iterations and eps must be positive")
    x = matrix.float()
    # A pure input-validation guard: the branch never changes the numerical
    # result on the valid (non-raising) path, only whether an invalid one is
    # caught early. Skipped while torch.compile is tracing -- a
    # data-dependent `if tensor.any():` is an unconditional graph break
    # (Dynamo: "Data-dependent branching... fundamental, unlikely Dynamo
    # will ever trace through it"), found compiling the full device-shaped
    # model on real Trn2 silicon (see
    # docs/model-dev/deepseek-v4-024-device-validation.md's device-graph-
    # capture section). Eager callers (every existing caller) keep the full
    # check.
    if not torch.compiler.is_compiling() and (x < 0).any():
        raise ValueError("Sinkhorn input matrix must be non-negative")
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x.to(matrix.dtype)


def mix_residual(residual: torch.Tensor, update: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Mix widened residual streams with a doubly-stochastic matrix."""
    if residual.shape != update.shape or residual.shape[-2] != logits.shape[-1]:
        raise ValueError("mHC stream dimensions do not agree")
    matrix = sinkhorn(logits)
    return torch.einsum("...ij,...jd->...id", matrix, residual) + update


def hyperconnection_reference(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    iterations: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Portable Transformers-equivalent mHC collapse and mixing weights."""
    if hidden_streams.ndim < 2 or scale.shape != (3,):
        raise ValueError("invalid mHC stream or scale shape")
    hc = hidden_streams.shape[-2]
    width = hidden_streams.shape[-1]
    mix = (2 + hc) * hc
    if fn.shape != (mix, hc * width) or base.shape != (mix,):
        raise ValueError("mHC parameters do not match the stream geometry")
    flat = hidden_streams.flatten(start_dim=-2).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    # Sliced, not `.split([...], dim=-1)`. `Tensor.split` with a *list* of
    # sizes returns the wrong data on Neuron for any dim other than 0 -- every
    # chunk, not just one, and silently: the un-split tensor is correct to
    # 2.4e-4 while the chunks are off by 717. Measured on trn2 with
    # `tools/deepseek_v4/check_mhc_device.py`; the FX graph is correct
    # (`split` -> `getitem`), so the defect is below it in the torch-xla /
    # neuronx-cc lowering. Basic slicing, `narrow`, `index_select`, `chunk`,
    # and `split` with an *int* size are all correct. CPU is unaffected, which
    # is why this cost a full real-weight device bring-up to find.
    projected = F.linear(flat, fn.float())
    pre_w = projected[..., :hc]
    post_w = projected[..., hc : 2 * hc]
    comb_w = projected[..., 2 * hc :]
    # `base` splits on dim 0, which measures correct on device -- sliced anyway
    # so the two adjacent decompositions cannot drift apart again.
    flat_base = base.float()
    pre_b = flat_base[:hc]
    post_b = flat_base[hc : 2 * hc]
    comb_b = flat_base[2 * hc :]
    pre = torch.sigmoid(pre_w * scale[0].float() + pre_b) + hc_eps
    post = 2 * torch.sigmoid(post_w * scale[1].float() + post_b)
    comb_logits = (
        comb_w.view(*comb_w.shape[:-1], hc, hc) * scale[2].float()
        + comb_b.view(hc, hc)
    )
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = sinkhorn_positive(comb, iterations, hc_eps).float()
    collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=-2)
    return post, comb, collapsed.to(hidden_streams.dtype)


def apply_hyperconnection(
    hidden_streams: torch.Tensor,
    update: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Place one sublayer update into streams and transpose-mix residuals."""
    if post.shape != hidden_streams.shape[:-1] or update.shape != hidden_streams.shape[:-2] + (
        hidden_streams.shape[-1],
    ):
        raise ValueError("mHC update shapes do not agree")
    dtype = hidden_streams.dtype
    return post.to(dtype).unsqueeze(-1) * update.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), hidden_streams
    )
