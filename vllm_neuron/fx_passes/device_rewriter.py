# SPDX-License-Identifier: Apache-2.0
"""Device rewriting pass for XLA compilation compatibility."""

import logging

import torch
from torch.fx import Node

from .base import FXPass


class DeviceRewriterPass(FXPass):
    """FX pass that rewrites device metadata for any operation with device parameter.

    Simple logic: For any node that has a device parameter in kwargs, if the device
    is not XLA, replace it with XLA.
    """

    def __init__(self):
        """Initialize the device rewriter pass."""
        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        """Return the pass name."""
        return "device_rewriter"

    def _get_device_type(self, device) -> str:
        """Extract device type from various device formats.

        Args:
            device: Device in various formats (str, torch.device, etc.)

        Returns:
            str: Device type string (e.g., 'cpu', 'cuda', 'xla')
        """
        if hasattr(device, "type"):
            return device.type
        elif isinstance(device, str):
            return device
        else:
            return str(device)

    def run(
        self, gm: torch.fx.GraphModule, **kwargs
    ) -> tuple[torch.fx.GraphModule, dict]:
        """Execute the device rewriting pass.

        Args:
            gm: The PyTorch FX GraphModule to transform
            **kwargs: Additional arguments (target_device, etc.)

        Returns:
            tuple[torch.fx.GraphModule, dict]: The transformed GraphModule and metadata

        Raises:
            RuntimeError: If the transformation fails
        """
        target_device = kwargs.get("target_device", "xla")

        try:
            # Single pass through all nodes
            rewrite_count = 0
            for node in gm.graph.nodes:
                if "device" in node.kwargs:
                    if self._rewrite_device_in_kwargs(node, target_device):
                        rewrite_count += 1
                    else:
                        # If we found a device parameter but didn't rewrite it, this is unexpected
                        # because our simple logic should rewrite ALL non-XLA devices to XLA
                        current_device = node.kwargs["device"]
                        current_device_type = self._get_device_type(current_device)

                        raise RuntimeError(
                            f"Unexpected: Found node {node.target} with device parameter "
                            f"'{current_device_type}' that was not rewritten. This should not "
                            f"happen in the device rewriter"
                        )

            # Recompile the graph module after modifications
            gm.recompile()

            self.logger.debug(f"Rewrote device metadata for {rewrite_count} operations")

            # Prepare metadata
            metadata = {
                "rewrite_count": rewrite_count,
            }

            return gm, metadata
        except Exception as e:
            raise RuntimeError(f"Device rewriting failed: {str(e)}") from e

    def _rewrite_device_in_kwargs(self, node: Node, target_device: str) -> bool:
        """Rewrite device metadata in node kwargs by creating a new node copy.

        Args:
            node: The FX node to modify
            target_device: Target device type

        Returns:
            bool: True if device was rewritten, False otherwise
        """
        if "device" not in node.kwargs:
            return False

        current_device = node.kwargs["device"]

        # Determine the current device type
        current_device_type = self._get_device_type(current_device)

        # Only rewrite if current device is NOT the target device
        if current_device_type != target_device:
            # Create new kwargs with updated device
            new_kwargs = dict(node.kwargs)

            # Preserve the original device format (string vs torch.device object)
            if isinstance(current_device, str):
                # Original was a string, replace with string
                new_kwargs["device"] = target_device
            elif hasattr(current_device, "type"):
                # Original was a torch.device object, replace with torch.device object
                new_kwargs["device"] = torch.device(target_device, index=0)
            else:
                # Fallback: use string format
                new_kwargs["device"] = target_device

            node.kwargs = new_kwargs

            self.logger.debug(
                f"Rewrote device for {node.target} from {current_device} to {target_device}"
            )
            return True
        else:
            # Already an XLA tensor - no need to re-write
            return True

        return False
