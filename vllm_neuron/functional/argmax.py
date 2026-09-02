# SPDX-License-Identifier: Apache-2.0
"""
Distributed argmax kernel for tensor-parallel inference.

This module provides a distributed argmax operation that works across
sharded tensors in a tensor-parallel setting.
"""

import logging
from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import all_gather_tensor
logger = logging.getLogger(__name__)


def argmax(
    tensor: Tensor,
    dim: int,
    gather_dim: int,
    keepdim: bool = False,
    process_group: Optional[ProcessGroup] = None,
) -> Tensor:
    """
    Performs distributed argmax on sharded tensors using 2-step algorithm.

    This function implements a distributed argmax operation for tensor-parallel
    inference where tensors are sharded across multiple devices. It uses a
    2-step approach:

    1. **Local argmax**: Each rank computes argmax on its local shard using
       NKI cascaded_max kernel (when conditions are met) or torch.max fallback
    2. **Global argmax**: Results are gathered and global argmax is computed

    **Sharding Layout**:
    The input tensor is assumed to be uniformly sharded along `gather_dim` across
    all ranks in the process group. For example, with TP=2 and vocab_size=256:
    - Rank 0: logits[:, 0:128]   (first 128 vocab tokens)
    - Rank 1: logits[:, 128:256] (last 128 vocab tokens)

    **When to Use**:

    This function is intended for distributed argmax. It will fall back to torch.argmax when process_group is not provided.

    **Kernel**:
    In step 1 we compute the local maxs using either a nki kernel or torch.max.
    NKI cascaded_max kernel is used when: dim=-1, 2D/3D tensor, size>=128, and
    not in CPU mode. Otherwise falls back to torch.max.

    Args:
        tensor: Input tensor to perform argmax on. Must be uniformly sharded
            along `gather_dim` across all ranks in `process_group`.
        dim: Dimension along which to find argmax.
        gather_dim: Dimension the tensor is sharded on. When `dim == gather_dim`,
            indices are corrected for sharding offset.
        keepdim: Whether to keep the reduced dimension. Defaults to False.
        process_group: Process group for distributed operations.
            Must be provided for distributed execution.

    Returns:
        Tensor with global argmax indices across all shards. All ranks return
        the same result.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from vllm_neuron.functional.argmax import argmax
        >>>
        >>> # Tensor-parallel inference with TP=2, vocab_size=256
        >>> # Each rank has logits of shape (batch=1, vocab_per_rank=128)
        >>> logits = torch.randn(1, 128)  # Local shard on this rank
        >>> pg = dist.new_group([0, 1])
        >>>
        >>> # Compute global argmax across both ranks
        >>> token_id = argmax(logits, dim=1, gather_dim=1, process_group=pg)
        >>> # Returns shape (1,) with global token index in range [0, 256)
        >>> # Rank 0 indices are in [0, 128), Rank 1 indices are in [128, 256)
    """
    if process_group is None:
        raise ValueError("process_group must be provided for distributed argmax")

    # ``all_gather_tensor`` is not negative-dimension safe: its internal
    # chunk/cat view construction treats ``-1 + 1`` as the start of the shape
    # and appends the original dimensions again. Normalize both dimensions
    # before gathering, matching the distributed top-k implementation.
    dim = dim % tensor.ndim
    gather_dim = gather_dim % tensor.ndim

    tp_degree = torch.distributed.get_world_size(group=process_group)

    # Fast path for single device
    if tp_degree == 1:
        return torch.argmax(tensor, dim=dim, keepdim=keepdim)

    # Gather the shards themselves and reduce locally, rather than exchanging
    # per-rank maxima. Exchanging maxima is the cheaper algorithm and was what
    # this function did, but it is not correct on this platform: see
    # `docs/model-dev/deepseek-v4-on-device-sampling.md`. A small all-gather --
    # the [batch, 1] local maximum -- silently returns a buffer in which each
    # rank sees only the shards belonging to its own physical Neuron device,
    # with the remaining slots left uninitialized. Measured at TP8 on trn2,
    # every rank in the first device agreed with its three neighbours, every
    # rank in the second agreed with its own three, and the two halves never
    # agreed. When an uninitialized slot won the reduction, the index that came
    # back with it was a float bit pattern, i.e. an out-of-vocabulary token id.
    #
    # The defect is a function of payload width, not of the gather dimension or
    # the index dtype: widths 1, 128 and 1024 are all wrong and 16160 is
    # correct, so there is no cheap padding that avoids it. Gathering the shard
    # is exactly what `nn/cpl.py` does on every host-sampled decode at this same
    # TP degree, and is correct there. It costs one vocabulary-width gather per
    # sampling call; on-device sampling still avoids the host round trip, which
    # is its main benefit.
    global_tensor = all_gather_tensor(
        tensor.contiguous(), gather_dim, group=process_group
    )

    # Indices into the gathered tensor are already global when the gather
    # concatenated the same axis the reduction runs over, so the per-rank
    # sharding offset the exchange-maxima algorithm needed is gone with it.
    return torch.argmax(global_tensor, dim=dim, keepdim=keepdim)
