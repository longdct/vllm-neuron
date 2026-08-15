# SPDX-License-Identifier: Apache-2.0
"""Tests for DeepSeek-V4 config normalization and validation.

Two things are being pinned here. First, that the raw-checkpoint form
(``compress_ratios``) and the normalized Transformers form (``layer_types`` +
``compress_rates``) produce *identical* per-layer structure -- the equivalence
the plan requires, since which form arrives depends on the pinned Transformers
version. Second, that every unrecognized or internally inconsistent config
raises instead of being interpreted, because a mis-read layer type produces a
model that loads and runs and is silently wrong.
"""

from types import SimpleNamespace

import pytest

# Structural stand-in for the V4-Flash pattern: leading sliding-window layers,
# an alternating c4/c128 body, a trailing sliding-window layer. Deliberately
# short. This is *not* the pinned checkpoint's ratio list -- these tests cover
# the normalizer's handling of the shape, not the checkpoint's contents.
RATIOS = [0, 0, 4, 128, 4, 128, 0]

LAYER_TYPES = [
    "sliding_attention",
    "sliding_attention",
    "compressed_attention",
    "compressed_attention",
    "compressed_attention",
    "compressed_attention",
    "sliding_attention",
]


def base_fields(**overrides):
    """A minimal config that validates, before per-layer fields are added."""
    fields = {
        "num_hidden_layers": len(RATIOS),
        "num_hash_layers": 3,
        "sliding_window": 128,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "index_topk": 512,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "num_nextn_predict_layers": 1,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
    }
    fields.update(overrides)
    return fields


def raw_form(**overrides):
    """Raw checkpoint form: ``compress_ratios``."""
    return base_fields(compress_ratios=list(RATIOS), **overrides)


def normalized_form(**overrides):
    """Normalized Transformers form: ``layer_types`` + ``compress_rates``."""
    return base_fields(
        layer_types=list(LAYER_TYPES), compress_rates=list(RATIOS), **overrides
    )


class TestFormEquivalence:
    def test_raw_and_normalized_forms_produce_identical_layers(
        self, deepseek_v4_config
    ):
        """The core requirement: which form arrived must not change the model."""
        from_raw = deepseek_v4_config.normalize_layer_specs(raw_form())
        from_normalized = deepseek_v4_config.normalize_layer_specs(normalized_form())
        assert from_raw == from_normalized

    def test_equivalence_holds_for_the_whole_normalized_config(
        self, deepseek_v4_config
    ):
        assert deepseek_v4_config.normalize_config(
            raw_form()
        ) == deepseek_v4_config.normalize_config(normalized_form())

    def test_accepts_mapping_and_object_configs_alike(self, deepseek_v4_config):
        """``AutoConfig`` yields an object; ``config.json`` yields a dict."""
        as_dict = deepseek_v4_config.normalize_config(raw_form())
        as_object = deepseek_v4_config.normalize_config(SimpleNamespace(**raw_form()))
        assert as_dict == as_object

    def test_both_forms_present_and_agreeing_is_accepted(self, deepseek_v4_config):
        config = base_fields(
            compress_ratios=list(RATIOS),
            layer_types=list(LAYER_TYPES),
            compress_rates=list(RATIOS),
        )
        assert deepseek_v4_config.normalize_layer_specs(config) == (
            deepseek_v4_config.normalize_layer_specs(raw_form())
        )


class TestLayerStructure:
    def test_compress_ratio_selects_attention_kind(self, deepseek_v4_config):
        layers = deepseek_v4_config.normalize_layer_specs(raw_form())
        kinds = [l.attention for l in layers]
        sliding = deepseek_v4_config.AttentionKind.SLIDING_WINDOW
        compressed = deepseek_v4_config.AttentionKind.COMPRESSED
        assert kinds == [sliding, sliding, compressed, compressed, compressed, compressed, sliding]
        assert [l.compress_ratio for l in layers] == RATIOS

    def test_leading_layers_are_hash_moe(self, deepseek_v4_config):
        layers = deepseek_v4_config.normalize_layer_specs(raw_form())
        hash_moe = deepseek_v4_config.MLPKind.HASH_MOE
        routed = deepseek_v4_config.MLPKind.ROUTED_MOE
        assert [l.mlp for l in layers[:3]] == [hash_moe] * 3
        assert all(l.mlp is routed for l in layers[3:])

    def test_layer_indices_are_positional(self, deepseek_v4_config):
        layers = deepseek_v4_config.normalize_layer_specs(raw_form())
        assert [l.index for l in layers] == list(range(len(RATIOS)))

    def test_zero_hash_layers_leaves_every_layer_routed(self, deepseek_v4_config):
        layers = deepseek_v4_config.normalize_layer_specs(
            raw_form(num_hash_layers=0)
        )
        routed = deepseek_v4_config.MLPKind.ROUTED_MOE
        assert all(l.mlp is routed for l in layers)


class TestCacheGrouping:
    """Grouping is what the heterogeneous cache registration consumes (plan P1.1)."""

    def test_distinct_groups_are_layout_distinct_and_ordered(self, deepseek_v4_config):
        normalized = deepseek_v4_config.normalize_config(raw_form())
        assert normalized.distinct_cache_groups() == (
            ("sliding_window", 0),
            ("compressed", 4),
            ("compressed", 128),
        )

    def test_layer_indices_map_back_to_their_group(self, deepseek_v4_config):
        normalized = deepseek_v4_config.normalize_config(raw_form())
        assert normalized.layer_indices_for(("sliding_window", 0)) == (0, 1, 6)
        assert normalized.layer_indices_for(("compressed", 4)) == (2, 4)
        assert normalized.layer_indices_for(("compressed", 128)) == (3, 5)

    def test_every_layer_belongs_to_exactly_one_group(self, deepseek_v4_config):
        normalized = deepseek_v4_config.normalize_config(raw_form())
        covered = [
            index
            for key in normalized.distinct_cache_groups()
            for index in normalized.layer_indices_for(key)
        ]
        assert sorted(covered) == list(range(normalized.num_hidden_layers))
        assert len(covered) == len(set(covered))


class TestUnrecognizedForms:
    def test_neither_form_present_raises(self, deepseek_v4_config):
        with pytest.raises(deepseek_v4_config.DeepseekV4ConfigError, match="neither"):
            deepseek_v4_config.normalize_layer_specs(base_fields())

    @pytest.mark.parametrize("dropped", ["layer_types", "compress_rates"])
    def test_half_of_the_normalized_form_raises(self, deepseek_v4_config, dropped):
        config = normalized_form()
        del config[dropped]
        with pytest.raises(
            deepseek_v4_config.DeepseekV4ConfigError, match="requires both"
        ):
            deepseek_v4_config.normalize_layer_specs(config)

    def test_unknown_layer_type_string_raises_and_names_the_value(
        self, deepseek_v4_config
    ):
        """The guess-refusal the module exists to make."""
        types = list(LAYER_TYPES)
        types[2] = "linear_attention"
        config = normalized_form()
        config["layer_types"] = types
        with pytest.raises(
            deepseek_v4_config.DeepseekV4ConfigError, match="linear_attention"
        ):
            deepseek_v4_config.normalize_layer_specs(config)

    def test_forms_present_but_disagreeing_raises(self, deepseek_v4_config):
        conflicting = list(RATIOS)
        conflicting[2] = 128
        config = base_fields(
            compress_ratios=conflicting,
            layer_types=list(LAYER_TYPES),
            compress_rates=list(RATIOS),
        )
        with pytest.raises(
            deepseek_v4_config.DeepseekV4ConfigError, match="disagree"
        ):
            deepseek_v4_config.normalize_layer_specs(config)

    def test_layer_type_inconsistent_with_its_rate_raises(self, deepseek_v4_config):
        """The two encodings of one fact must agree."""
        types = list(LAYER_TYPES)
        types[0] = "compressed_attention"  # but compress_rates[0] == 0
        config = normalized_form()
        config["layer_types"] = types
        with pytest.raises(
            deepseek_v4_config.DeepseekV4ConfigError, match="inconsistent"
        ):
            deepseek_v4_config.normalize_layer_specs(config)


class TestValidation:
    def test_ratio_list_length_must_match_layer_count(self, deepseek_v4_config):
        with pytest.raises(
            deepseek_v4_config.DeepseekV4ConfigError, match="num_hidden_layers"
        ):
            deepseek_v4_config.normalize_layer_specs(
                raw_form(num_hidden_layers=len(RATIOS) + 1)
            )

    def test_unsupported_compression_ratio_raises(self, deepseek_v4_config):
        ratios = list(RATIOS)
        ratios[2] = 8
        with pytest.raises(deepseek_v4_config.DeepseekV4ConfigError, match=r"\[8\]"):
            deepseek_v4_config.normalize_layer_specs(base_fields(compress_ratios=ratios))

    def test_hash_layers_beyond_layer_count_raises(self, deepseek_v4_config):
        with pytest.raises(
            deepseek_v4_config.DeepseekV4ConfigError, match="exceeds"
        ):
            deepseek_v4_config.normalize_layer_specs(
                raw_form(num_hash_layers=len(RATIOS) + 1)
            )

    def test_non_integer_ratio_raises(self, deepseek_v4_config):
        with pytest.raises(
            deepseek_v4_config.DeepseekV4ConfigError, match="must be an integer"
        ):
            deepseek_v4_config.normalize_layer_specs(
                base_fields(compress_ratios=[0, 0, "4", 128, 4, 128, 0])
            )

    def test_missing_required_field_raises(self, deepseek_v4_config):
        config = raw_form()
        del config["index_topk"]
        with pytest.raises(
            deepseek_v4_config.DeepseekV4ConfigError, match="index_topk"
        ):
            deepseek_v4_config.normalize_config(config)

    def test_multi_head_kv_is_rejected(self, deepseek_v4_config):
        """MLA keeps one latent KV per token; anything else is another model."""
        with pytest.raises(
            deepseek_v4_config.DeepseekV4ConfigError, match="expected 1"
        ):
            deepseek_v4_config.normalize_config(raw_form(num_key_value_heads=8))

    def test_topk_exceeding_expert_count_is_rejected(self, deepseek_v4_config):
        with pytest.raises(deepseek_v4_config.DeepseekV4ConfigError, match="exceeds"):
            deepseek_v4_config.normalize_config(
                raw_form(n_routed_experts=4, num_experts_per_tok=6)
            )

    @pytest.mark.parametrize("field", ["n_group", "topk_group"])
    def test_v3_grouped_routing_is_rejected(self, deepseek_v4_config, field):
        """V4 dropped node-limited routing; a config carrying it is not a V4."""
        with pytest.raises(
            deepseek_v4_config.DeepseekV4ConfigError, match="DeepSeek-V3"
        ):
            deepseek_v4_config.normalize_config(raw_form(**{field: 8}))

    @pytest.mark.parametrize(
        "field,value",
        [("scoring_func", "sigmoid"), ("topk_method", "greedy")],
    )
    def test_unsupported_routing_variants_are_rejected(
        self, deepseek_v4_config, field, value
    ):
        with pytest.raises(
            deepseek_v4_config.DeepseekV4ConfigError, match="unsupported"
        ):
            deepseek_v4_config.normalize_config(raw_form(**{field: value}))

    def test_booleans_are_not_accepted_as_integers(self, deepseek_v4_config):
        """``True == 1`` in Python; a bool here means the config is malformed."""
        with pytest.raises(deepseek_v4_config.DeepseekV4ConfigError):
            deepseek_v4_config.normalize_config(raw_form(hc_mult=True))


class TestTinyAndFullScaleConfigs:
    def test_tiny_synthetic_config_normalizes(self, deepseek_v4_config):
        """Plan P3 builds tiny models; the normalizer must not assume real sizes."""
        tiny = base_fields(
            num_hidden_layers=3,
            num_hash_layers=1,
            compress_ratios=[0, 4, 128],
            num_attention_heads=4,
            head_dim=64,
            n_routed_experts=8,
            num_experts_per_tok=2,
            index_topk=16,
        )
        normalized = deepseek_v4_config.normalize_config(tiny)
        assert normalized.num_hidden_layers == 3
        assert normalized.num_hash_layers == 1
        assert normalized.distinct_cache_groups() == (
            ("sliding_window", 0),
            ("compressed", 4),
            ("compressed", 128),
        )

    def test_flash_scale_layer_count_normalizes(self, deepseek_v4_config):
        """43 layers, the V4-Flash count, in the documented structural shape."""
        ratios = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(40)] + [0]
        assert len(ratios) == 43
        normalized = deepseek_v4_config.normalize_config(
            base_fields(num_hidden_layers=43, compress_ratios=ratios)
        )
        assert normalized.num_hidden_layers == 43
        assert normalized.num_hash_layers == 3
        assert len(normalized.layer_indices_for(("compressed", 4))) == 20
        assert len(normalized.layer_indices_for(("compressed", 128))) == 20
        assert normalized.layer_indices_for(("sliding_window", 0)) == (0, 1, 42)
