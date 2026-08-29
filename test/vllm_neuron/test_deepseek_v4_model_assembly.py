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
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

from vllm_neuron.model.deepseek_v4.model import (
    DeepseekV4Attention,
    DeepseekV4ForCausalLM,
    DeepseekV4MoE,
    NeuronDeepseekV4RotaryEmbedding,
)
from vllm_neuron.model.deepseek_v4.weight_loaders import load_checkpoint_weights
from vllm_neuron.model.kv_cache import CacheKind
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.vllm.worker.kv_spec_conversion import layer_spec_to_vllm_spec


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


def run_chunked(model, specs, tokens_list, max_blocks=8):
    cached = {spec.name: 0 for spec in specs}
    outputs = []
    for chunk in tokens_list:
        n = len(chunk)
        attn_metadata = build_attn_metadata(
            specs, cached, n, max_blocks=max_blocks
        )
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


def test_load_weights_restores_nonpersistent_buffers_after_meta_materialization(
    tmp_path,
):
    """HF checkpoints omit derived buffers, so load must rebuild them."""
    reference = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    with torch.device("meta"):
        materialized = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()

    # An empty directory exercises materialization without masking the result
    # by loading parameters. The deterministic buffers must still be valid.
    materialized.load_weights(str(tmp_path), torch.device("cpu"), None)

    reference_attention = [
        module
        for module in reference.modules()
        if isinstance(module, DeepseekV4Attention)
    ]
    materialized_attention = [
        module
        for module in materialized.modules()
        if isinstance(module, DeepseekV4Attention)
    ]
    for expected, actual in zip(reference_attention, materialized_attention):
        torch.testing.assert_close(actual.identity_kv_weight, expected.identity_kv_weight)

    expected_rope = next(
        module
        for module in reference.modules()
        if isinstance(module, NeuronDeepseekV4RotaryEmbedding)
    )
    actual_rope = next(
        module
        for module in materialized.modules()
        if isinstance(module, NeuronDeepseekV4RotaryEmbedding)
    )
    for layer_type in expected_rope.layer_types:
        torch.testing.assert_close(
            getattr(actual_rope, f"{layer_type}_inv_freq"),
            getattr(expected_rope, f"{layer_type}_inv_freq"),
        )

    expected_moe = next(
        module
        for module in reference.modules()
        if isinstance(module, DeepseekV4MoE)
    )
    actual_moe = next(
        module
        for module in materialized.modules()
        if isinstance(module, DeepseekV4MoE)
    )
    torch.testing.assert_close(actual_moe.tid2eid, expected_moe.tid2eid)


def test_device_model_declares_exact_heterogeneous_cache_inventory():
    """Four SWA + four compressed + four carry caches.

    Ten before the lightning indexer; the single c4 layer now adds two more --
    the indexer runs a whole second compressor at ``index_head_dim``, so it
    needs its own compressed-entry pages and its own carry pages. c128 layers
    have no indexer and are unchanged.
    """
    config = hf_config()
    model = DeepseekV4ForCausalLM.from_configs(config).eval()
    specs = model.get_kv_spec().layers
    assert len(specs) == 12
    assert [spec.cache_kind for spec in specs].count(
        CacheKind.SLIDING_WINDOW_MLA
    ) == 4
    mla_specs = [spec for spec in specs if spec.cache_kind is CacheKind.MLA]
    # The second ratio-4 group is the c4 layer's indexer, alongside its own.
    assert [spec.compress_ratio for spec in mla_specs] == [128, 4, 4, 128]
    assert [spec.name.endswith(".indexer") for spec in mla_specs] == [
        False,
        False,
        True,
        False,
    ]
    # The indexer's entries are index_head_dim wide, not latent_size.
    assert [spec.head_size for spec in mla_specs] == [
        config.head_dim,
        config.head_dim,
        config.index_head_dim,
        config.head_dim,
    ]
    # Explicit block_size=128 on every compressed group -- see the comment in
    # get_kv_spec: the platform default of 32 does not divide 128.
    assert [spec.block_size for spec in mla_specs] == [128, 128, 128, 128]
    carry = [spec for spec in specs if spec.cache_kind is CacheKind.COMPRESSOR_STATE]
    # head_size = 2*(1+overlap)*width; width is latent_size (16) for the outer
    # compressors and index_head_dim (128) for the indexer's.
    assert [
        (spec.block_size, spec.sliding_window_size, spec.head_size) for spec in carry
    ] == [
        (8, 128, 32),
        (4, 8, 64),
        (4, 8, 4 * config.index_head_dim),
        (8, 128, 32),
    ]


def test_compressed_cache_pages_scale_with_public_block_size():
    model = DeepseekV4ForCausalLM.from_configs(
        hf_config(), NeuronConfig(kv_cache_block_size=256)
    ).eval()
    mla_specs = [
        spec
        for spec in model.get_kv_spec().layers
        if spec.cache_kind is CacheKind.MLA
    ]
    assert [spec.block_size for spec in mla_specs] == [1024, 1024, 1024, 1024]
    assert all(spec.block_size % spec.compress_ratio == 0 for spec in mla_specs)
    converted = [
        layer_spec_to_vllm_spec(spec, 256, torch.bfloat16)
        for spec in model.get_kv_spec().layers
    ]
    full_page_sizes = [
        spec.page_size_bytes for spec in converted if isinstance(spec, MLAAttentionSpec)
    ]
    swa_page_sizes = [
        spec.page_size_bytes
        for spec in converted
        if isinstance(spec, SlidingWindowMLASpec)
    ]
    assert max(swa_page_sizes) <= max(full_page_sizes)


def test_only_c4_layers_declare_an_indexer():
    """c128/HCA attends to all of its compressed history and has no indexer."""
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    kinds = [
        (layer.attention.ratio, layer.attention.indexer is not None)
        for layer in model.model.layers
    ]
    assert kinds == [(128, False), (0, False), (4, True), (128, False)]


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


def indexer_hf_config(index_topk):
    """One CSA layer, with an indexer budget small enough to actually bite.

    ``sliding_window`` is kept short so the compressed entries carry real
    signal rather than being shadowed by a local window covering everything.
    """
    return DeepseekV4Config(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        sliding_window=8,
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["moe"],
        index_topk=index_topk,
        index_n_heads=2,
        index_head_dim=8,
    )


def _indexer_model(index_topk, seed=17):
    torch.manual_seed(seed)
    model = DeepseekV4ForCausalLM.from_configs(indexer_hf_config(index_topk)).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    return model


#: 40 tokens at ratio 4 gives 10 compressed entries -- comfortably past a
#: budget of 2, so selection has to discard real content.
_INDEXER_TOKENS = [t % 64 for t in range(40)]

#: The carry-state group pages at block_size 4, so 40 tokens needs 10 blocks.
_INDEXER_BLOCKS = 16


def test_a_small_indexer_budget_changes_the_output():
    """Guards against the indexer being wired in as a silent no-op.

    With the stock ``index_topk`` nothing is ever pruned at test scale, so an
    indexer that selected everything -- or was never called -- would look
    exactly like a correct one. Shrinking the budget is what makes the
    difference observable.
    """
    unbounded = _indexer_model(999)
    bounded = _indexer_model(2)
    specs = unbounded.get_kv_spec().layers

    unbounded.bind_kv_cache(fresh_caches(unbounded))
    dense = run_chunked(unbounded, specs, [_INDEXER_TOKENS], max_blocks=_INDEXER_BLOCKS)
    bounded.bind_kv_cache(fresh_caches(bounded))
    sparse = run_chunked(
        bounded,
        bounded.get_kv_spec().layers,
        [_INDEXER_TOKENS],
        max_blocks=_INDEXER_BLOCKS,
    )

    assert torch.isfinite(dense).all() and torch.isfinite(sparse).all()
    assert not torch.allclose(dense, sparse), "indexer pruned nothing"


def test_indexer_selection_is_chunk_invariant():
    """Selection must depend on the token, not on how tokens were batched.

    The indexer runs inside the per-token loop and reads a cache that is
    written by that same loop, so a chunk-boundary bug here would show up as
    prefill and decode disagreeing -- which is exactly how this model is
    served.
    """
    model = _indexer_model(2)
    specs = model.get_kv_spec().layers

    model.bind_kv_cache(fresh_caches(model))
    whole = run_chunked(model, specs, [_INDEXER_TOKENS], max_blocks=_INDEXER_BLOCKS)

    model.bind_kv_cache(fresh_caches(model))
    per_token = run_chunked(
        model, specs, [[t] for t in _INDEXER_TOKENS], max_blocks=_INDEXER_BLOCKS
    )
    torch.testing.assert_close(whole, per_token, rtol=1e-3, atol=1e-4)

    model.bind_kv_cache(fresh_caches(model))
    chunks = run_chunked(
        model,
        specs,
        [_INDEXER_TOKENS[:9], _INDEXER_TOKENS[9:23], _INDEXER_TOKENS[23:]],
        max_blocks=_INDEXER_BLOCKS,
    )
    torch.testing.assert_close(whole, chunks, rtol=1e-3, atol=1e-4)


def test_indexer_budget_above_the_entry_count_is_dense_attention():
    """The equivalence the admission bound rests on, end to end.

    40 tokens at ratio 4 is 10 entries; any budget at or above that must
    select all of them, so the model output must match a budget of 10 exactly.
    """
    at_bound = _indexer_model(10)
    over_bound = _indexer_model(4096)

    at_bound.bind_kv_cache(fresh_caches(at_bound))
    tight = run_chunked(
        at_bound,
        at_bound.get_kv_spec().layers,
        [_INDEXER_TOKENS],
        max_blocks=_INDEXER_BLOCKS,
    )
    over_bound.bind_kv_cache(fresh_caches(over_bound))
    loose = run_chunked(
        over_bound,
        over_bound.get_kv_spec().layers,
        [_INDEXER_TOKENS],
        max_blocks=_INDEXER_BLOCKS,
    )

    torch.testing.assert_close(tight, loose)


def test_indexer_checkpoint_tensors_load_onto_the_indexer_module():
    """The names the real checkpoint ships for the indexer, end to end.

    Spelled exactly as the official index JSON spells them. Until the indexer
    existed these were skipped outright; the risk now is the opposite one --
    that they land on the *layer's* compressor instead of the indexer's, since
    both are ``DeepseekV4Compressor``s with identically-named parameters.
    """
    model = DeepseekV4ForCausalLM.from_configs(hf_config()).eval()
    indexer = model.model.layers[2].attention.indexer
    assert indexer is not None
    outer = model.model.layers[2].attention.compressor

    before = outer.fused_wkv_wgate.weight.detach().clone()
    fused = indexer.compressor.fused_wkv_wgate.weight
    rows, columns = fused.shape[0] // 2, fused.shape[1]
    values = {
        "layers.2.attn.indexer.wq_b.weight": torch.randn_like(indexer.q_b_proj.weight),
        "layers.2.attn.indexer.weights_proj.weight": torch.randn_like(
            indexer.weights_proj.weight
        ),
        "layers.2.attn.indexer.compressor.wkv.weight": torch.randn(
            rows, columns, dtype=fused.dtype
        ),
        "layers.2.attn.indexer.compressor.wgate.weight": torch.randn(
            rows, columns, dtype=fused.dtype
        ),
        "layers.2.attn.indexer.compressor.ape": torch.randn_like(
            indexer.compressor.ape
        ),
        "layers.2.attn.indexer.compressor.norm.weight": torch.randn_like(
            indexer.compressor.norm_weight
        ),
    }
    loaded = load_checkpoint_weights(model, list(values.items()))

    assert loaded == {
        "model.layers.2.attention.indexer.q_b_proj.weight",
        "model.layers.2.attention.indexer.weights_proj.weight",
        "model.layers.2.attention.indexer.compressor.fused_wkv_wgate.weight",
        "model.layers.2.attention.indexer.compressor.ape",
        "model.layers.2.attention.indexer.compressor.norm_weight",
    }
    torch.testing.assert_close(
        indexer.q_b_proj.weight, values["layers.2.attn.indexer.wq_b.weight"]
    )
    torch.testing.assert_close(
        indexer.weights_proj.weight,
        values["layers.2.attn.indexer.weights_proj.weight"],
    )
    torch.testing.assert_close(
        indexer.compressor.ape, values["layers.2.attn.indexer.compressor.ape"]
    )
    torch.testing.assert_close(
        indexer.compressor.norm_weight,
        values["layers.2.attn.indexer.compressor.norm.weight"],
    )
    # wkv is shard 0, wgate shard 1, concatenated on dim 0.
    torch.testing.assert_close(
        fused[:rows], values["layers.2.attn.indexer.compressor.wkv.weight"]
    )
    torch.testing.assert_close(
        fused[rows:], values["layers.2.attn.indexer.compressor.wgate.weight"]
    )
    # And the layer's own compressor is untouched by any of it.
    torch.testing.assert_close(outer.fused_wkv_wgate.weight, before)
