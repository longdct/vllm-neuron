# SPDX-License-Identifier: Apache-2.0
"""TorchNeuron Native backend selection for the vLLM Neuron plugin."""

import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)


class NeuronBackend(Enum):
    """Supported Neuron backend implementations."""

    NEURON_NATIVE = "neuron_native"


def get_backend() -> NeuronBackend:
    """Return the only supported backend."""
    value = os.environ.get("VLLM_NEURON_BACKEND", "").lower()
    if value == "vllm_neuron":
        raise ValueError(
            "VLLM_NEURON_BACKEND=vllm_neuron selected the retired lite/XLA "
            "backend. Install a compatible torch-neuronx build and unset "
            "VLLM_NEURON_BACKEND (or temporarily use neuron_native)."
        )
    if value not in ("", "neuron_native"):
        raise ValueError(
            f"Invalid VLLM_NEURON_BACKEND value: {value!r}. "
            "Valid values are unset and neuron_native."
        )
    logger.info("Using TorchNeuron Native backend")
    return NeuronBackend.NEURON_NATIVE


def get_platform_class() -> str:
    """Return the bundled vLLM platform implementation."""
    get_backend()
    return "vllm_neuron.vllm.platform.NeuronPlatform"
