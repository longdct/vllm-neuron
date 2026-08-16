# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import DeepseekV4Config

from vllm_neuron.model.deepseek_v4.model import DeepseekV4ForCausalLM
from vllm_neuron.model.kv_cache import CacheKind


def hf_config():
    return DeepseekV4Config(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        num_hidden_layers=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=512,
        q_lora_rank=16,
        layer_types=[
            "heavily_compressed_attention",
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        mlp_layer_types=["hash_moe", "moe", "moe", "moe"],
    )


def test_hf_config_builds_production_shaped_multi_variant_model():
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    assert model.model.embed_tokens.num_embeddings == 64
    assert [(layer.ratio, layer.moe.kind) for layer in model.model.layers] == [
        (128, "hash_moe"),
        (0, "routed_moe"),
        (4, "routed_moe"),
        (128, "routed_moe"),
    ]
    assert "model.embed_tokens.weight" in dict(model.named_parameters())
    assert "lm_head.weight" in dict(model.named_parameters())


def test_production_shaped_forward_decode_is_chunk_invariant():
    torch.manual_seed(41)
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    tokens = torch.tensor([1, 7, 3, 11, 2])
    expected, _ = model(tokens)
    first, state = model(tokens[:3])
    decoded, state = model(tokens[3:], state)
    torch.testing.assert_close(torch.cat((first, decoded)), expected, rtol=0, atol=0)
    assert state.num_tokens == len(tokens)


def test_production_namespace_loader_binds_embedding_and_head():
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    embed = torch.randn_like(model.model.embed_tokens.weight)
    head = torch.randn_like(model.lm_head.weight)
    loaded = model.load_weights([("embed.weight", embed), ("head.weight", head)])
    assert loaded == {"model.embed_tokens.weight", "lm_head.weight"}
    torch.testing.assert_close(model.model.embed_tokens.weight, embed)
    torch.testing.assert_close(model.lm_head.weight, head)


def test_production_model_declares_exact_heterogeneous_cache_inventory():
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    specs = model.get_kv_spec().layers
    assert len(specs) == 10  # four SWA + three compressed + three carry caches
    assert [spec.cache_kind for spec in specs].count(
        CacheKind.SLIDING_WINDOW_MLA
    ) == 4
    assert [
        spec.compress_ratio for spec in specs if spec.cache_kind is CacheKind.MLA
    ] == [128, 4, 128]
    carry = [spec for spec in specs if spec.cache_kind is CacheKind.COMPRESSOR_STATE]
    assert [
        (spec.block_size, spec.sliding_window_size, spec.head_size) for spec in carry
    ] == [
        (8, 128, 1024),
        (4, 8, 2048),
        (8, 128, 1024),
    ]


def test_production_model_strictly_binds_single_tensor_caches():
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    caches = {
        spec.name: [torch.empty(1, dtype=spec.dtype)]
        for spec in model.get_kv_spec().layers
    }
    model.bind_kv_cache(caches)
    assert model._kv_caches == caches
    missing = dict(caches)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing="):
        model.bind_kv_cache(missing)
    bad_arity = dict(caches)
    name = next(iter(bad_arity))
    bad_arity[name] = [torch.empty(1), torch.empty(1)]
    with pytest.raises(ValueError, match="one latent tensor"):
        model.bind_kv_cache(bad_arity)
