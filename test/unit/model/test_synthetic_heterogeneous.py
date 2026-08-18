# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.kv_cache import CacheKind
from vllm_neuron.model.synthetic.synthetic import SyntheticConfig, SyntheticNeuronModel


def test_synthetic_model_declares_every_deepseek_cache_layout():
    layouts = ["swa", "mla", "c4", "c128", "carry", "rswa"]
    model = SyntheticNeuronModel(
        SyntheticConfig(
            num_hidden_layers=len(layouts),
            hidden_size=512,
            num_attention_heads=1,
            num_key_value_heads=1,
            sliding_window=128,
            cache_layouts=layouts,
        )
    )
    layers = model.get_kv_spec().layers
    assert [layer.cache_kind for layer in layers] == [
        CacheKind.SLIDING_WINDOW,
        CacheKind.MLA,
        CacheKind.MLA,
        CacheKind.MLA,
        CacheKind.COMPRESSOR_STATE,
        CacheKind.RSWA,
    ]
    assert [layers[i].compress_ratio for i in (1, 2, 3)] == [1, 4, 128]


def test_layout_count_must_match_layer_count():
    model = SyntheticNeuronModel(
        SyntheticConfig(num_hidden_layers=2, cache_layouts=["mla"])
    )
    with pytest.raises(ValueError, match="length"):
        model.get_kv_spec()


def test_single_tensor_latent_can_be_bound_for_lifecycle_tests():
    model = SyntheticNeuronModel(SyntheticConfig(num_hidden_layers=1))
    latent = torch.zeros(2, 1, 1, 512)
    model.bind_kv_cache({"layers.0.self_attn": [latent]})
    assert model._kv_caches["layers.0.self_attn"] == [latent]
