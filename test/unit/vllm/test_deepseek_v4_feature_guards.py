# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm_neuron.vllm.platform import NeuronPlatform


def config(*, model_type="deepseek_v4", prefix=False, speculative=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix),
        speculative_config=speculative,
    )


def test_prefix_caching_is_rejected_for_deepseek_v4():
    with pytest.raises(NotImplementedError, match="prefix caching"):
        NeuronPlatform._validate_deepseek_v4_unsupported_features(config(prefix=True))


def test_speculative_decode_is_rejected_for_deepseek_v4():
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        NeuronPlatform._validate_deepseek_v4_unsupported_features(
            config(speculative=object())
        )


def test_other_models_keep_their_existing_feature_support():
    NeuronPlatform._validate_deepseek_v4_unsupported_features(
        config(model_type="llama", prefix=True, speculative=object())
    )
