# SPDX-License-Identifier: Apache-2.0
"""FX Passes for vLLM Neuron compilation pipeline.

This module provides a scalable architecture for managing FX graph transformations
in the vLLM Neuron compilation pipeline.
"""

from .inplace_rewrite_pass import InPlaceToOutOfPlacePass
from .aliasing_pass import AliasingOutputRewritePass
from .collective_replica_groups_pass import CollectiveReplicaGroupsPass
from .device_rewriter import DeviceRewriterPass
from .backend_config_pass import NkiKernelWriteBackendConfigPass
from .pass_manager import FXPassManager


def get_default_pass_manager() -> FXPassManager:
    """Get the default pass manager with standard passes configured.

    Returns:
        FXPassManager: Configured pass manager with DeviceRewriterPass
    """
    manager = FXPassManager()
    manager.add_pass(DeviceRewriterPass())
    manager.add_pass(NkiKernelWriteBackendConfigPass())
    manager.add_pass(AliasingOutputRewritePass())
    manager.add_pass(InPlaceToOutOfPlacePass())
    manager.add_pass(CollectiveReplicaGroupsPass())
    return manager


__all__ = [
    "CollectiveReplicaGroupsPass",
    "DeviceRewriterPass",
    "FXPassManager",
    "get_default_pass_manager",
]
