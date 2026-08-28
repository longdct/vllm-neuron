# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
from transformers import DeepseekV4Config

from vllm_neuron.model.deepseek_v4.dense_csa import DenseCsaUnsupportedError
from vllm_neuron.vllm.platform import NeuronPlatform


def _configure(monkeypatch, enabled):
    config = DeepseekV4Config(
        num_hidden_layers=1, layer_types=["compressed_sparse_attention"]
    )
    if enabled:
        monkeypatch.setenv("VLLM_NEURON_DEEPSEEK_V4_DENSE_CSA_BOUND", "1")
    else:
        monkeypatch.delenv("VLLM_NEURON_DEEPSEEK_V4_DENSE_CSA_BOUND", raising=False)
    NeuronPlatform._configure_deepseek_v4_dense_csa_bound(
        SimpleNamespace(hf_config=config)
    )


@pytest.fixture(autouse=True)
def configured_bound(monkeypatch):
    """The opt-in fallback ceiling, on.

    Off is the default now that the lightning indexer exists -- selection
    happens at any length, so there is nothing to protect. These tests cover
    the escape hatch; ``test_the_bound_is_off_by_default`` covers the default.
    """
    _configure(monkeypatch, enabled=True)
    yield
    NeuronPlatform._deepseek_dense_csa_bound = None


def test_the_bound_is_off_by_default(monkeypatch):
    """No ceiling without the opt-in: the indexer removed its premise."""
    _configure(monkeypatch, enabled=False)
    assert NeuronPlatform._deepseek_dense_csa_bound is None
    # Far past the old 2051-token limit, and uncapped besides.
    NeuronPlatform.validate_request(
        {"prompt_token_ids": list(range(100_000))}, SimpleNamespace(max_tokens=None)
    )


def test_request_at_2051_total_tokens_is_admitted():
    NeuronPlatform.validate_request(
        {"prompt_token_ids": list(range(2000))}, SimpleNamespace(max_tokens=51)
    )


def test_request_at_2052_total_tokens_is_rejected():
    with pytest.raises(DenseCsaUnsupportedError, match="2052"):
        NeuronPlatform.validate_request(
            {"prompt_token_ids": list(range(2000))}, SimpleNamespace(max_tokens=52)
        )


def test_uncapped_request_is_rejected():
    with pytest.raises(DenseCsaUnsupportedError, match="no output-length cap"):
        NeuronPlatform.validate_request(
            {"prompt_token_ids": [1, 2]}, SimpleNamespace(max_tokens=None)
        )


def test_prompt_embeds_use_their_token_dimension():
    prompt_embeds = SimpleNamespace(shape=(2000, 64))
    NeuronPlatform.validate_request(
        {"prompt_embeds": prompt_embeds}, SimpleNamespace(max_tokens=51)
    )
    with pytest.raises(DenseCsaUnsupportedError, match="2052"):
        NeuronPlatform.validate_request(
            {"prompt_embeds": prompt_embeds}, SimpleNamespace(max_tokens=52)
        )


def test_missing_prompt_length_is_rejected_before_scheduling():
    with pytest.raises(ValueError, match="requires prompt token count"):
        NeuronPlatform.validate_request({}, SimpleNamespace(max_tokens=1))
