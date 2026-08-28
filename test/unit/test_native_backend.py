# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "vllm_neuron" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backend_default_and_alias(monkeypatch):
    backend = _load("backend")
    monkeypatch.delenv("VLLM_NEURON_BACKEND", raising=False)
    assert backend.get_backend() is backend.NeuronBackend.NEURON_NATIVE
    monkeypatch.setenv("VLLM_NEURON_BACKEND", "neuron_native")
    assert backend.get_backend() is backend.NeuronBackend.NEURON_NATIVE


def test_legacy_backend_is_actionable_error(monkeypatch):
    backend = _load("backend")
    monkeypatch.setenv("VLLM_NEURON_BACKEND", "vllm_neuron")
    with pytest.raises(ValueError, match="retired.*torch-neuronx"):
        backend.get_backend()


def test_native_option_translation(monkeypatch):
    native = _load("native")
    monkeypatch.delenv("NEURON_COMPILER_OPT_LEVEL", raising=False)
    assert native.native_compile_options(
        model_name="tiny", optimization_level=1, use_tensorizer=True
    ) == {"model_name": "tiny", "use_tensorizer_backend": True}
    assert os.environ["NEURON_COMPILER_OPT_LEVEL"] == "-O1"
    with pytest.raises(ValueError, match="optimization level"):
        native.native_compile_options(model_name="tiny", optimization_level=9)


def test_rank_isolated_cache_directories(monkeypatch, tmp_path):
    native = _load("native")
    for name in (
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_NEURONX_HLO_CACHE_DIR",
        "TORCH_NEURONX_NEFF_CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    rank_root = native.configure_cache_environment(str(tmp_path), 3)
    assert rank_root == tmp_path / "rank_3"
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(rank_root / "inductor")
    assert os.environ["TORCH_NEURONX_HLO_CACHE_DIR"] == str(rank_root / "hlo")
    assert os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(rank_root / "neff")


def test_retired_lite_settings_are_rejected(monkeypatch):
    native = _load("native")
    monkeypatch.setenv("VLLM_NEURON_DISABLE_PARALLEL_TRACE", "1")
    with pytest.raises(ValueError, match="in-process warmup"):
        native.reject_retired_configuration()
