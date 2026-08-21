# SPDX-License-Identifier: Apache-2.0
"""Device-shaped model assembly: real forward contract, real cache I/O.

Covers the Step 1-3 rewrite in ``vllm_neuron/model/deepseek_v4/model.py``:
batched ``attn_metadata``-driven forward (no Python token loop at the model
level), ``bind_kv_cache`` attaching tensors to submodules, and paged cache
I/O for the SWA / compressed-MLA / compressor-carry cache groups. See
``docs/model-dev/deepseek-v4-carry-cache-design.md`` for the addressing
design these tests exercise.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import DeepseekV4Config

from vllm_neuron.model.deepseek_v4.model import DeepseekV4ForCausalLM
from vllm_neuron.model.deepseek_v4.weight_loaders import load_checkpoint_weights
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
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        layer_types=[
            "heavily_compressed_attention",
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        mlp_layer_types=["hash_moe", "moe", "moe", "moe"],
    )


def fresh_caches(model, num_blocks=64):
    caches = {}
    for spec in model.get_kv_spec().layers:
        ratio = getattr(spec, "compress_ratio", 1) or 1
        physical_slots = (spec.block_size // ratio) if ratio > 1 else (spec.block_size or 32)
        shape = (num_blocks, 1, physical_slots, spec.head_size)
        caches[spec.name] = [torch.zeros(shape, dtype=spec.dtype)]
    return caches


def build_attn_metadata(specs, cached_seq_len_by_group, new_tokens, max_blocks=8):
    """One request (index 0), sequential block allocation from block 0.

    Mirrors the shape ``neuron_model_runner.py`` hands the model
    (``block_table_tensor``, ``slot_mapping``, ``cached_seq_len``, ...), with
    a plain sequential (non-evicting) block allocator standing in for the
    real ``KVCacheManager`` -- sufficient for exercising the model's own
    cache-I/O logic in isolation.
    """
    metadata = {}
    for spec in specs:
        block_size = spec.block_size or 32
        prior = cached_seq_len_by_group[spec.name]
        needed_blocks = max(1, (prior + new_tokens + block_size - 1) // block_size)
        block_table = torch.arange(needed_blocks, dtype=torch.int32).unsqueeze(0)
        block_table = torch.nn.functional.pad(
            block_table, (0, max_blocks - block_table.shape[1]), value=0
        )
        raw_positions = torch.arange(prior, prior + new_tokens)
        blk_idx = raw_positions // block_size
        pos_idx = raw_positions % block_size
        slot_mapping = block_table[0, blk_idx] * block_size + pos_idx
        metadata[spec.name] = {
            "block_table_tensor": block_table,
            "slot_mapping": slot_mapping,
            "max_query_len": new_tokens,
            "block_size": block_size,
            "max_blocks_per_seq": block_table.shape[1],
            "decode_token_threshold": 1,
            "cached_seq_len": torch.tensor([[prior]], dtype=torch.int32),
            "kv_segment_size": 0,
        }
    return metadata


def run_chunked(model, specs, tokens_list):
    cached = {spec.name: 0 for spec in specs}
    outputs = []
    for chunk in tokens_list:
        n = len(chunk)
        attn_metadata = build_attn_metadata(specs, cached, n)
        input_ids = torch.tensor(chunk)
        positions = torch.arange(cached[specs[0].name], cached[specs[0].name] + n)
        sampling_positions = torch.arange(n)
        logits = model(input_ids, positions, attn_metadata, sampling_positions)
        outputs.append(logits)
        for spec in specs:
            cached[spec.name] += n
    return torch.cat(outputs, dim=0)


def test_hf_config_builds_device_shaped_multi_variant_model():
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    assert model.model.embed_tokens.num_embeddings == 64
    assert [
        (layer.ratio, layer.moe.kind) for layer in model.model.layers
    ] == [
        (128, "hash_moe"),
        (0, "routed_moe"),
        (4, "routed_moe"),
        (128, "routed_moe"),
    ]
    assert "model.embed_tokens.weight" in dict(model.named_parameters())
    assert "lm_head.weight" in dict(model.named_parameters())


def test_device_model_declares_exact_heterogeneous_cache_inventory():
    """get_kv_spec is untouched by the Step 1-3 rewrite -- still 10 groups."""
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    specs = model.get_kv_spec().layers
    assert len(specs) == 10  # four SWA + three compressed + three carry caches
    assert [spec.cache_kind for spec in specs].count(
        CacheKind.SLIDING_WINDOW_MLA
    ) == 4
    mla_specs = [spec for spec in specs if spec.cache_kind is CacheKind.MLA]
    assert [spec.compress_ratio for spec in mla_specs] == [128, 4, 128]
    # Explicit block_size=128 on every compressed group -- see the comment in
    # get_kv_spec: the platform default of 32 does not divide 128.
    assert [spec.block_size for spec in mla_specs] == [128, 128, 128]
    carry = [spec for spec in specs if spec.cache_kind is CacheKind.COMPRESSOR_STATE]
    # head_size = 2*(1+overlap)*latent_size; latent_size == head_dim == 16 here.
    assert [
        (spec.block_size, spec.sliding_window_size, spec.head_size) for spec in carry
    ] == [
        (8, 128, 32),
        (4, 8, 64),
        (8, 128, 32),
    ]


def test_bind_kv_cache_attaches_tensors_to_attention_and_compressor_modules():
    """Real bind_kv_cache: attaches onto submodules, not a validate-only dict."""
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    caches = fresh_caches(model)
    model.bind_kv_cache(caches)

    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}.self_attn"
        assert layer.attention.swa_cache is caches[f"{prefix}.swa_cache"][0]
        if layer.attention.compressor is not None:
            assert layer.attention.mla_cache is caches[prefix][0]
            assert (
                layer.attention.compressor.state_cache
                is caches[f"{prefix}.compressor.state_cache"][0]
            )
        else:
            assert layer.attention.mla_cache is None


def test_bind_kv_cache_still_rejects_key_and_arity_mismatches():
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    caches = fresh_caches(model)
    missing = dict(caches)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing="):
        model.bind_kv_cache(missing)

    bad_arity = dict(caches)
    name = next(iter(bad_arity))
    bad_arity[name] = [torch.empty(1), torch.empty(1)]
    with pytest.raises(ValueError, match="one latent tensor"):
        model.bind_kv_cache(bad_arity)


def test_production_namespace_loader_binds_embedding_and_head():
    """The official-checkpoint-name mapping in weight_loaders.py, unchanged
    by this pass. model.load_weights() itself now matches the runner's real
    ``(checkpoint_path, device, cache_dir)`` contract (reads a safetensors
    directory) rather than taking an iterator directly -- see its docstring.
    """
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    embed = torch.randn_like(model.model.embed_tokens.weight)
    head = torch.randn_like(model.lm_head.weight)
    loaded = load_checkpoint_weights(
        model, [("embed.weight", embed), ("head.weight", head)]
    )
    assert loaded == {"model.embed_tokens.weight", "lm_head.weight"}
    torch.testing.assert_close(model.model.embed_tokens.weight, embed)
    torch.testing.assert_close(model.lm_head.weight, head)


def test_batched_forward_is_chunk_invariant_against_single_shot():
    """The real Step 1-3 correctness bar: attn_metadata-driven batched
    forward, run one token at a time or in arbitrary chunks, matches a
    single-shot forward over the whole sequence -- through real paged SWA,
    compressed-MLA, and compressor-carry cache I/O, not a Python state
    object. Small float32 tolerance because attention itself is exact
    per-token (see model.py's per-token attention loop and its docstring)
    but the mHC/MoE stages remain batched across a chunk, which is not
    guaranteed bit-identical to a differently-shaped batched call.
    """
    torch.manual_seed(41)
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    specs = model.get_kv_spec().layers
    tokens = [1, 7, 3, 11, 2, 9, 4]

    model.bind_kv_cache(fresh_caches(model))
    whole = run_chunked(model, specs, [tokens])

    model.bind_kv_cache(fresh_caches(model))
    per_token = run_chunked(model, specs, [[t] for t in tokens])
    torch.testing.assert_close(whole, per_token, rtol=1e-3, atol=1e-4)

    model.bind_kv_cache(fresh_caches(model))
    arbitrary_chunks = run_chunked(model, specs, [tokens[:3], tokens[3:5], tokens[5:]])
    torch.testing.assert_close(whole, arbitrary_chunks, rtol=1e-3, atol=1e-4)
