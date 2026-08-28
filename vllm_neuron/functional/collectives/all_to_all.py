# SPDX-License-Identifier: Apache-2.0
import torch
import torch.distributed.distributed_c10d as c10d

from typing import Optional
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import _resolve_group_name

from vllm_neuron import envs


def _compute_use_allgather_fallback() -> bool:
    """Determine at import time whether to use allgather fallback.

    trn2's ENC engine does not support the Mesh algorithm required for
    cross-device all_to_all. We emulate it via all_gather (which uses
    ring/tree algorithms that work cross-device) followed by a local slice.

    The platform helper lives in ``TorchNeuron Native`` and is imported
    lazily so this module still imports cleanly on CPU/CI hosts where the
    Neuron runtime is absent.
    """
    if envs.VLLM_NEURON_CPU_MODE:
        return False
    try:
        from torch_neuronx.utils import get_platform_target

        return get_platform_target() == "trn2"
    except Exception:
        return False


_USE_ALLGATHER_FALLBACK = _compute_use_allgather_fallback()


def all_to_all(
    tensor: torch.Tensor,
    split_dim: int,
    concat_dim: int,
    group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    """All-to-all collective that redistributes tensor across ranks.

    Splits input tensor along split_dim, sends one piece to each rank,
    receives one piece from each rank, and concatenates along concat_dim.

    Args:
        tensor: Input tensor to redistribute.
        split_dim: Dimension to split for sending.
        concat_dim: Dimension to concatenate received pieces.
        group: Process group for the collective. If None, uses default group.

    Returns:
        Redistributed tensor with split_dim reduced and concat_dim increased by group_size.

    Example:
        >>> # world_size=2, input shape [4, 6]
        >>> result = all_to_all(tensor, split_dim=0, concat_dim=1, group=group)
        >>> # Output shape: [2, 12] (split dim 0 by 2, concat dim 1 by 2)
    """
    group_name = _resolve_group_name(group, "")
    group_size = c10d._get_group_size_by_name(group_name)

    # Normalize negative dimensions
    split_dim = split_dim % tensor.ndim
    concat_dim = concat_dim % tensor.ndim

    # trn2 fallback: all_gather + slice (ENC lacks Mesh for all_to_all)
    if _USE_ALLGATHER_FALLBACK:
        return _allgather_slice_fallback(
            tensor, split_dim, concat_dim, group, group_size
        )

    # Special case: split and concat on same dimension
    if split_dim == concat_dim:
        if split_dim != 0:
            tensor = tensor.transpose(0, split_dim).contiguous()

        output = _all_to_all_single_dim0(tensor, group_size, group, group_name)

        if split_dim != 0:
            output = output.transpose(0, split_dim)
        return output

    # Move split_dim to position 0
    if split_dim != 0:
        tensor = tensor.transpose(0, split_dim).contiguous()
        # Adjust concat_dim if needed
        if concat_dim == 0:
            concat_dim = split_dim
        elif concat_dim == split_dim:
            concat_dim = 0

    # Perform all_to_all_single on dim 0
    output = _all_to_all_single_dim0(tensor, group_size, group, group_name)

    # Reshape to separate ranks: [group_size*split_size, ...] → [group_size, split_size, ...]
    split_size = output.shape[0] // group_size
    output = output.view([group_size, split_size] + list(output.shape[1:]))

    # Move split_size to front: [group_size, split_size, ...] → [split_size, group_size, ...]
    output = output.transpose(0, 1).contiguous()

    # Merge group_size into concat_dim (currently at position 1)
    if concat_dim == 0:
        # Merge into dim 0: flatten [split_size, group_size] → [split_size*group_size]
        output = output.reshape([split_size * group_size] + list(output.shape[2:]))
    else:
        # Merge into another dim: move group_size there and flatten
        # concat_dim is relative to original tensor with split_dim at 0
        output = output.transpose(1, concat_dim).contiguous()
        shape = list(output.shape)
        shape[concat_dim] = shape[concat_dim] * shape[concat_dim + 1]
        del shape[concat_dim + 1]
        output = output.reshape(shape)

    # Move dim 0 back to original split_dim position
    if split_dim != 0:
        output = output.transpose(0, split_dim)

    return output


def _all_to_all_single_dim0(
    tensor: torch.Tensor,
    group_size: int,
    group: Optional[ProcessGroup],
    group_name: str,
) -> torch.Tensor:
    """Perform all_to_all_single on dimension 0.

    Args:
        tensor: Input tensor with split dimension at position 0.
        group_size: Size of the process group.
        group: Process group for CPU mode.
        group_name: Group name for non-CPU mode.

    Returns:
        Output tensor after all_to_all_single.
    """
    if envs.VLLM_NEURON_CPU_MODE:
        split_size = tensor.shape[0] // group_size
        output = torch.empty(
            [split_size * group_size] + list(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        torch.distributed.all_to_all_single(output, tensor, group=group)
        return output
    else:
        return torch.ops._c10d_functional.all_to_all_single(tensor, [], [], group_name)


def _allgather_slice_fallback(
    tensor: torch.Tensor,
    split_dim: int,
    concat_dim: int,
    group: Optional[ProcessGroup],
    group_size: int,
) -> torch.Tensor:
    """Emulate all_to_all via all_gather + slice for trn2.

    all_to_all(split_dim=S, concat_dim=C) is equivalent to:
      1. all_gather along concat_dim (every rank's data on C)
      2. slice split_dim to keep only this rank's portion

    This works because all_gather uses ring/tree algorithms that are
    supported cross-device on trn2, unlike the Mesh algorithm needed
    for native all_to_all.
    """
    rank = torch.distributed.get_rank(group)
    group_name = _resolve_group_name(group, "")

    # all_gather along concat_dim: transpose to dim 0, gather, transpose back
    if concat_dim != 0:
        tensor_t = tensor.transpose(0, concat_dim).contiguous()
    else:
        tensor_t = tensor.contiguous()

    gathered = torch.ops._c10d_functional.all_gather_into_tensor(
        tensor_t, group_size, group_name
    )
    torch.ops._c10d_functional.wait_tensor(gathered)

    if concat_dim != 0:
        gathered = gathered.transpose(0, concat_dim)

    # ``gathered`` stacks ``group_size`` source blocks along split_dim, one per
    # source rank, each holding that rank's full split_dim extent.
    block_size = gathered.shape[split_dim] // group_size

    if split_dim == concat_dim:
        # split_dim and concat_dim are the same axis, so a single contiguous
        # narrow would just return this rank's own gathered block (identity).
        # Instead take this rank's sub-chunk from every source block and
        # concatenate them in source-rank order along the shared axis, matching
        # native all_to_all's split_dim == concat_dim behavior.
        chunk_size = block_size // group_size
        offset = rank * chunk_size
        pieces = [
            gathered.narrow(split_dim, b * block_size + offset, chunk_size)
            for b in range(group_size)
        ]
        return torch.cat(pieces, dim=concat_dim).contiguous()

    # split_dim != concat_dim: gather and slice act on independent axes, so
    # keeping this rank's contiguous chunk along split_dim reproduces the
    # exchange.
    start = rank * block_size
    return gathered.narrow(split_dim, start, block_size).contiguous()
