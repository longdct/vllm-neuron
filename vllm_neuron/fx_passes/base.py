# SPDX-License-Identifier: Apache-2.0
"""Base interface for FX passes in the vLLM Neuron compilation pipeline."""

from abc import ABC, abstractmethod

import torch


class FXPass(ABC):
    """Abstract base class for FX graph transformation passes.

    All FX passes must implement this interface to be used in the pass manager.
    """

    @abstractmethod
    def run(
        self, gm: torch.fx.GraphModule, **kwargs
    ) -> tuple[torch.fx.GraphModule, dict]:
        """Transform the GraphModule and return the modified version with metadata.

        Args:
            gm: The PyTorch FX GraphModule to transform
            **kwargs: Additional arguments passed from the pass manager

        Returns:
            tuple[torch.fx.GraphModule, dict]: The transformed GraphModule and metadata dict

        Raises:
            RuntimeError: If the transformation fails
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the pass name for logging and debugging.

        Returns:
            str: A unique identifier for this pass
        """
        pass
