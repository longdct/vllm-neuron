# SPDX-License-Identifier: Apache-2.0
"""
Overrides module for vLLM Neuron.

This module contains dispatch overrides for various PyTorch components
"""

# Auto-import the functional collectives library implementation when overrides module is imported
from . import (
    neuron_collectives,  # noqa: F401
    xla_collectives,  # noqa: F401
)
