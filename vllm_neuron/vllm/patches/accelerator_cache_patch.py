# SPDX-License-Identifier: Apache-2.0
"""Compatibility for upstream cleanup on the Neuron Lite backend."""

from __future__ import annotations

import torch


def apply_neuron_empty_cache_patch() -> None:
    """Make upstream's generic accelerator cleanup safe on Neuron Lite."""
    if getattr(torch.accelerator.empty_cache, "_vllm_neuron_noop", False) is True:
        return

    def _neuron_empty_cache() -> None:
        return None

    _neuron_empty_cache._vllm_neuron_noop = True  # type: ignore[attr-defined]
    torch.accelerator.empty_cache = _neuron_empty_cache
