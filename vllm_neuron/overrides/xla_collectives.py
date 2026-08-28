# SPDX-License-Identifier: Apache-2.0
"""
Custom implementations for _c10d_functional ops using torch.library.impl.
Registers custom XLA backend implementations.
"""

import logging
import os

import torch.distributed as dist
import torch_xla.core.xla_model as xm
from torch.distributed.distributed_c10d import _resolve_process_group
from torch.library import impl

# Set up logging
logger = logging.getLogger(__name__)

# Set XLA_ALWAYS_ALLREDUCE environment variable
os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"


def _get_reduce_type(reduceOp: str) -> str:
    """Map reduceOp string to xm reduce type constant."""
    reduce_type_map = {
        "sum": xm.REDUCE_SUM,
        "prod": xm.REDUCE_MUL,
        "mul": xm.REDUCE_MUL,
        "min": xm.REDUCE_MIN,
        "max": xm.REDUCE_MAX,
        "and": xm.REDUCE_AND,
        "or": xm.REDUCE_OR,
    }
    reduce_type = reduce_type_map.get(reduceOp.lower())
    if reduce_type is None:
        raise ValueError(f"Unsupported reduce operation: {reduceOp}")
    return reduce_type  # type: ignore[no-any-return]


def _get_replica_groups_from_group_name(group_name):
    """Extract replica groups from group name."""
    try:
        group = _resolve_process_group(group_name)
        ranks = dist.get_process_group_ranks(group)
        return [ranks]  # xm APIs expect list of lists
    except Exception as e:
        logger.error(f"Could not resolve process group '{group_name}': {e}")
        raise RuntimeError(f"Failed to resolve process group '{group_name}'") from e


@impl("_c10d_functional::all_reduce", "XLA")
def custom_all_reduce_xla(tensor, reduceOp, group_name):
    """Custom XLA implementation for all_reduce."""
    logger.debug(
        f"all_reduce on XLA: shape={tensor.shape}, op={reduceOp}, group={group_name}"
    )

    # Extract replica groups from group name
    replica_groups = _get_replica_groups_from_group_name(group_name)
    logger.debug(f"Using replica groups: {replica_groups}")

    # Get reduce type using helper function
    reduce_type = _get_reduce_type(reduceOp)
    logger.debug(f"Using reduce type: {reduce_type}")

    # Call xm.all_reduce - it should return a single tensor when given a single tensor
    result = xm.all_reduce(
        reduce_type,
        tensor,
        scale=1.0,
        groups=replica_groups,
        pin_layout=False,
    )

    # Ensure we return a single tensor, not a list
    if isinstance(result, list):
        logger.debug("xm.all_reduce returned list, extracting first element")
        result = result[0]

    logger.debug(
        f"all_reduce completed on XLA: result_shape={result.shape}, result_type={type(result)}"
    )
    return result


@impl("_c10d_functional::all_gather_into_tensor", "XLA")
def custom_all_gather_into_tensor_xla(input_tensor, group_size, group_name):
    """Custom XLA implementation for all_gather_into_tensor."""
    logger.debug(
        f"all_gather_into_tensor on XLA: shape={input_tensor.shape}, group_size={group_size}, group={group_name}"
    )

    # Extract replica groups from group name
    replica_groups = _get_replica_groups_from_group_name(group_name)
    logger.debug(f"Using replica groups: {replica_groups}")

    # Treat group name as channel id by hashing it to produce 64 bit int
    # Keep lower 31 bits to use as channel id which is 32 bits and must be positive
    # channel id is required when use_global_device_ids=True
    channel_id = hash(group_name) & 0x7FFFFFFF

    # Call xm.all_gather (gathers along dim 0 by default)
    # Need to pass use_global_device_ids=True as rank ids in replica group are global (relative to workload)
    # When you have multiple process groups with use_global_device_ids=True,
    # XLA needs to distinguish which collective operations should synchronize together.
    # The channel_id acts as a unique identifier.
    result = xm.all_gather(
        input_tensor,
        dim=0,
        groups=replica_groups,
        pin_layout=False,
        use_global_device_ids=True,
        channel_id=channel_id,
    )

    logger.debug(
        f"all_gather_into_tensor completed on XLA: result_shape={result.shape}"
    )
    return result


@impl("_c10d_functional::reduce_scatter_tensor", "XLA")
def custom_reduce_scatter_tensor_xla(input_tensor, reduceOp, group_size, group_name):
    """Custom XLA implementation for reduce_scatter_tensor."""
    logger.debug(
        f"reduce_scatter_tensor on XLA: shape={input_tensor.shape}, "
        f"op={reduceOp}, group_size={group_size}, group={group_name}"
    )

    # Extract replica groups from group name
    replica_groups = _get_replica_groups_from_group_name(group_name)
    logger.debug(f"Using replica groups: {replica_groups}")

    # Get reduce type using helper function
    reduce_type = _get_reduce_type(reduceOp)
    logger.debug(f"Using reduce type: {reduce_type}")
    shard_count = len(replica_groups[0])

    # Treat group name as channel id by hashing it to produce 64 bit int
    # Keep lower 31 bits to use as channel id which is 32 bits and must be positive
    # channel id is required when use_global_device_ids=True
    channel_id = hash(group_name) & 0x7FFFFFFF

    # Call xm.reduce_scatter
    # Need to pass use_global_device_ids=True as rank ids in replica group are global (relative to workload)
    # When you have multiple process groups with use_global_device_ids=True,
    # XLA needs to distinguish which collective operations should synchronize together.
    # The channel_id acts as a unique identifier.
    result = xm.reduce_scatter(
        reduce_type=reduce_type,
        input=input_tensor,
        scale=1.0,
        scatter_dim=0,
        shard_count=shard_count,
        groups=replica_groups,
        pin_layout=False,
        use_global_device_ids=True,
        channel_id=channel_id,
    )

    logger.debug(f"reduce_scatter_tensor completed: result_shape={result.shape}")
    return result


@impl("_c10d_functional::all_to_all_single", "XLA")
def custom_all_to_all_single_xla(
    input_tensor, output_split_sizes, input_split_sizes, group_name
):
    """Custom XLA implementation for all_to_all_single.

    xm.all_to_all does not support use_global_device_ids, so replica_groups
    must be a complete partition covering every rank exactly once.  We build
    the full partition from all registered sibling process groups.
    """
    logger.debug(
        f"all_to_all_single on XLA: input_shape={input_tensor.shape}, "
        f"input_split_sizes={input_split_sizes}, "
        f"group={group_name}"
    )

    # Build contiguous partition — xm.all_to_all requires all ranks to be covered
    replica_groups = _get_replica_groups_from_group_name(group_name)
    logger.debug(f"Using replica groups: {replica_groups}")

    # Determine split count from the group this rank belongs to
    split_count = (
        len(input_split_sizes) if input_split_sizes else len(replica_groups[0])
    )

    # Check if tensor can be split
    if input_tensor.shape[0] < split_count or input_tensor.shape[0] % split_count != 0:
        logger.warning(
            f"all_to_all_single: Cannot split tensor of size {input_tensor.shape[0]} "
            f"by split_count={split_count}. Returning input unchanged."
        )
        return input_tensor

    # Use xm.all_to_all with split and concat both on dim 0 (standard all_to_all_single behavior)
    result = xm.all_to_all(
        value=input_tensor,
        split_dimension=0,
        concat_dimension=0,
        split_count=split_count,
        groups=replica_groups,
        pin_layout=False,
    )

    logger.debug(f"all_to_all_single completed: result_shape={result.shape}")
    return result


logger.info(
    "✓ Custom XLA implementations registered via torch.library.impl (all_reduce, all_gather, reduce_scatter, all_to_all)"
)
