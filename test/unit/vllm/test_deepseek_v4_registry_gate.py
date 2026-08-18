# SPDX-License-Identifier: Apache-2.0
"""DeepseekV4ForCausalLM registration stays opt-in, not default.

Mirrors the existing ``VLLM_NEURON_SYNTHETIC_MODEL`` gate: registering the
model unconditionally would advertise production support that doesn't exist
yet (roadmap steps 4-5 -- real checkpoint loading, quantization, memory
calibration -- are still open). See ``vllm_neuron/model/registry.py``.
"""

import importlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")


def _get_models(monkeypatch, env_value):
    if env_value is None:
        monkeypatch.delenv("VLLM_NEURON_ENABLE_DEEPSEEK_V4", raising=False)
    else:
        monkeypatch.setenv("VLLM_NEURON_ENABLE_DEEPSEEK_V4", env_value)
    from vllm_neuron.model import registry

    importlib.reload(registry)
    return dict(registry.get_models())


def test_deepseek_v4_absent_by_default(monkeypatch):
    assert "DeepseekV4ForCausalLM" not in _get_models(monkeypatch, None)


def test_deepseek_v4_absent_when_flag_is_not_exactly_one(monkeypatch):
    assert "DeepseekV4ForCausalLM" not in _get_models(monkeypatch, "true")


def test_deepseek_v4_present_when_explicitly_enabled(monkeypatch):
    models = _get_models(monkeypatch, "1")
    assert "DeepseekV4ForCausalLM" in models
    from vllm_neuron.model.deepseek_v4.factory import (
        DeepseekV4ForCausalLM as Factory,
    )

    assert models["DeepseekV4ForCausalLM"] is Factory
