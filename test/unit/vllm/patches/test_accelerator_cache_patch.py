# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import torch

from vllm_neuron.vllm.patches.accelerator_cache_patch import (
    apply_neuron_empty_cache_patch,
)


def test_neuron_empty_cache_patch_is_idempotent(monkeypatch):
    original = Mock()
    monkeypatch.setattr(torch.accelerator, "empty_cache", original)

    apply_neuron_empty_cache_patch()
    patched = torch.accelerator.empty_cache
    patched()
    apply_neuron_empty_cache_patch()

    assert torch.accelerator.empty_cache is patched
    original.assert_not_called()
