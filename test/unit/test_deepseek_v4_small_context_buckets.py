# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_neuron.utils.bucket_utils import resolve_segmented_prefill_config


@pytest.mark.parametrize("query", [8, 64])
def test_dsv4_small_context_diagnostic_skips_segmented_prefill(monkeypatch, query):
    monkeypatch.setenv("VLLM_NEURON_DSV4_SMALL_CONTEXT", "1")
    assert resolve_segmented_prefill_config(query, 256) == (None, None)


def test_small_prefill_still_rejected_without_diagnostic(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_DSV4_SMALL_CONTEXT", raising=False)
    with pytest.raises(ValueError, match="not a supported chunked prefill size"):
        resolve_segmented_prefill_config(8, 256)
