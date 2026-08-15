# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 config normalization against its named oracle (plan P4).

The plan names the oracle for this component explicitly: *"Config normalization
→ Transformers config loader — it is the thing being normalized."* The unit
tests in ``test/unit/model/deepseek_v4/test_config.py`` run on a bare
interpreter against hand-built dicts, which pins the normalizer's *logic* but
cannot pin its *assumptions* about what a real config contains. These do.

That distinction is not academic. Every assumption checked below was originally
guessed wrong, and every one of those guesses passed the hand-built unit tests:

* the ``layer_types`` vocabulary (``compressed_attention`` does not exist)
* ``compress_rates`` being a per-layer list (it is a dict keyed by layer type)
* ``num_hash_layers`` being a live field (it is a legacy kwarg upstream consumes)

So this file exists to make the oracle, not our own fixtures, the thing the
normalizer is judged against.
"""

import pytest

pytest.importorskip("transformers", reason="oracle tests require Transformers")

from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    _COMPRESS_RATIO_TO_LAYER_TYPE,
    DEEPSEEK_V4_LAYER_TYPES,
    DEEPSEEK_V4_MLP_LAYER_TYPES,
)

from vllm_neuron.model.deepseek_v4.config import (
    SUPPORTED_COMPRESS_RATIOS,
    AttentionKind,
    MLPKind,
    normalize_layer_specs,
)
from vllm_neuron.model.deepseek_v4.dense_csa import (
    CountingMode,
    eligible_entries,
    geometry_from_config,
    max_dense_csa_tokens,
)


class TestCompressorGeometryOracle:
    @pytest.mark.parametrize(
        ("layer_type", "rate"),
        [
            ("compressed_sparse_attention", 4),
            ("heavily_compressed_attention", 128),
        ],
    )
    def test_entry_count_matches_complete_window_cadence(self, layer_type, rate):
        config = DeepseekV4Config()
        geometry = geometry_from_config(config, layer_type)
        for total_tokens in range(0, rate * 3 + 2):
            assert eligible_entries(
                total_tokens, geometry, mode=CountingMode.COMPLETE
            ) == total_tokens // rate

    def test_default_csa_dense_equivalence_bound_is_derived(self):
        config = DeepseekV4Config()
        geometry = geometry_from_config(config, "compressed_sparse_attention")
        limit = max_dense_csa_tokens(
            geometry, config.index_topk, mode=CountingMode.COMPLETE
        )
        assert limit == config.index_topk * 4 + 3
        assert eligible_entries(limit, geometry, mode=CountingMode.COMPLETE) == 512
        assert eligible_entries(limit + 1, geometry, mode=CountingMode.COMPLETE) == 513


class TestVocabularyMatchesUpstream:
    """Our tables must be the upstream tables, not a parallel guess at them."""

    def test_layer_type_vocabulary_is_complete(self):
        from vllm_neuron.model.deepseek_v4.config import _LAYER_TYPE_TO_RATIO

        assert set(_LAYER_TYPE_TO_RATIO) == set(DEEPSEEK_V4_LAYER_TYPES), (
            "layer_types vocabulary has drifted from upstream; an unknown "
            "spelling would raise at startup, and a stale one would mis-type layers"
        )

    def test_ratio_mapping_matches_upstream_inverse(self):
        from vllm_neuron.model.deepseek_v4.config import _LAYER_TYPE_TO_RATIO

        upstream = {v: k for k, v in _COMPRESS_RATIO_TO_LAYER_TYPE.items()}
        assert _LAYER_TYPE_TO_RATIO == upstream

    def test_mlp_vocabulary_is_complete(self):
        from vllm_neuron.model.deepseek_v4.config import _MLP_TYPE_KINDS

        assert set(_MLP_TYPE_KINDS) == set(DEEPSEEK_V4_MLP_LAYER_TYPES)

    def test_every_upstream_ratio_is_supported(self):
        """Upstream's ratios and our cache layouts must cover the same set.

        If upstream adds a compression ratio we have no layout for, the
        normalizer rejects it -- correct, but it should be a deliberate decision
        rather than a surprise at checkpoint-load time.
        """
        assert set(_COMPRESS_RATIO_TO_LAYER_TYPE) == set(SUPPORTED_COMPRESS_RATIOS)


class TestNormalizingTheRealConfig:
    def test_default_config_normalizes(self):
        config = DeepseekV4Config()
        layers = normalize_layer_specs(config)
        assert len(layers) == config.num_hidden_layers

    def test_ratios_follow_layer_types(self):
        config = DeepseekV4Config()
        layers = normalize_layer_specs(config)

        for spec, layer_type in zip(layers, config.layer_types):
            expected = 0 if layer_type == "sliding_attention" else (
                config.compress_rates[layer_type]
            )
            assert spec.compress_ratio == expected, (
                f"layer {spec.index} typed {layer_type!r} normalized to ratio "
                f"{spec.compress_ratio}, expected {expected}"
            )

    def test_attention_kind_follows_compression(self):
        layers = normalize_layer_specs(DeepseekV4Config())
        for spec in layers:
            expected = (
                AttentionKind.SLIDING_WINDOW
                if spec.compress_ratio == 0
                else AttentionKind.COMPRESSED
            )
            assert spec.attention is expected

    def test_mlp_kinds_follow_mlp_layer_types(self):
        """MLP structure comes from its own list, not from the attention type."""
        config = DeepseekV4Config()
        layers = normalize_layer_specs(config)

        expected = [
            MLPKind.HASH_MOE if t == "hash_moe" else MLPKind.ROUTED_MOE
            for t in config.mlp_layer_types
        ]
        assert [spec.mlp for spec in layers] == expected

    def test_attention_and_mlp_structure_are_independent(self):
        """Guard against re-deriving one from the other.

        In the default V4 config the first three layers are ``hash_moe`` while
        being ``heavily_compressed_attention`` -- so hash-MoE layers are *not*
        the sliding-window layers, and any implementation that inferred one from
        the other would be wrong here. Asserted so that inference cannot be
        reintroduced unnoticed.
        """
        config = DeepseekV4Config()
        layers = normalize_layer_specs(config)

        hash_layers = {s.index for s in layers if s.mlp is MLPKind.HASH_MOE}
        sliding_layers = {
            s.index for s in layers if s.attention is AttentionKind.SLIDING_WINDOW
        }
        assert hash_layers, "expected the default config to have hash-MoE layers"
        assert hash_layers != sliding_layers


class TestLegacyCheckpointForms:
    """The legacy fields survive only on raw JSON -- upstream consumes them."""

    def test_upstream_consumes_compress_ratios(self):
        """Pinning *why* the raw-form path cannot be dropped.

        ``compress_ratios`` is folded into ``layer_types`` and not retained, so a
        loaded config never carries it. The normalizer still supports it because
        raw ``config.json`` does.
        """
        ratios = [0, 4, 128] + [4] * (DeepseekV4Config().num_hidden_layers - 3)
        config = DeepseekV4Config(compress_ratios=ratios)

        assert not hasattr(config, "compress_ratios")
        assert config.layer_types[:3] == [
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ]

    def test_raw_ratios_and_loaded_config_agree(self):
        """The equivalence the plan requires, against the real loader.

        Feed the same structure in both forms -- raw ratios into upstream, and
        the resulting normalized config into ours -- and the per-layer result
        must match. This is the end-to-end form of P3a.1's equivalence
        requirement, with upstream rather than a fixture as the reference.
        """
        num_layers = DeepseekV4Config().num_hidden_layers
        ratios = [0, 4, 128, 4] * (num_layers // 4)
        ratios += [4] * (num_layers - len(ratios))

        loaded = DeepseekV4Config(compress_ratios=ratios)
        from_loaded = normalize_layer_specs(loaded)

        assert [s.compress_ratio for s in from_loaded] == ratios

    def test_upstream_consumes_num_hash_layers(self):
        config = DeepseekV4Config(num_hash_layers=5)

        assert not hasattr(config, "num_hash_layers")
        assert config.mlp_layer_types[:5] == ["hash_moe"] * 5
        assert config.mlp_layer_types[5] == "moe"

        layers = normalize_layer_specs(config)
        assert [s.index for s in layers if s.mlp is MLPKind.HASH_MOE] == list(range(5))
