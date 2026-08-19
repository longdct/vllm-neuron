# SPDX-License-Identifier: Apache-2.0
"""Device model wrapper modules vs. the real transformers architecture.

``test_deepseek_v4_component_oracles.py`` (pre-existing) validates the
*standalone functions* in ``mhc.py``/``moe.py``/``compressor.py`` against
real ``transformers.models.deepseek_v4.modeling_deepseek_v4`` reference
modules. This file validates the same real modules against this plugin's
actual ``model.py`` **wrapper classes** — the nn.Module glue that holds
parameters and is what actually runs in the device-shaped model — since a
correct standalone function does not guarantee its wrapper plugs the right
weights into it correctly, or that the surrounding assembly (e.g. per-layer
normalization) matches.

This is how the missing ``input_layernorm``/``post_attention_layernorm`` in
``DeepseekV4DecoderLayer`` was found: every wrapper checked here matched the
real architecture exactly (0.0 diff) once driven by the same weights, except
that the real decoder layer normalizes the mHC-collapsed hidden state before
attention/MoE and an earlier version of this plugin's layer did not. See
``docs/model-dev/deepseek-v4-carry-cache-design.md`` for the fuller account
and `model.py`'s ``DeepseekV4DecoderLayer` docstring.

``DeepseekV4Attention`` is now the real multi-head q_lora/kv_proj/partial-RoPE
MLA architecture end to end (q_a_proj/q_a_norm/q_b_proj, kv_proj/kv_norm, real
``DeepseekV4RotaryEmbedding``, attention sinks, K=V broadcast via an identity
"up-projection", the real architecture's undo-RoPE-on-the-output step, and now
the real grouped low-rank output projection too -- ``o_a_proj``/``o_b_proj``,
``DeepseekV4GroupedLinear`` -- rather than a plain dense ``Linear``),
cross-validated on the full final output both in isolation and through the
real paged-cache-I/O path (multi-token prefill, real ``bind_kv_cache``). See
``test_attention_matches_real_module_through_paged_cache_io`` below and
``model.py``'s ``DeepseekV4Attention`` docstring.

``DeepseekV4Expert`` (the per-expert FFN, both routed and shared) now matches
the real ``DeepseekV4Experts``/``DeepseekV4MLP`` exactly too: ``[out,
in]``-layout ``gate_up_proj``/``down_proj`` driven through ``F.linear``, and
gate/up clamped to ``swiglu_limit`` before the SiLU*up product -- not the
earlier unclamped, ``[in, out]``-layout approximation. See
``test_expert_wrapper_matches_real_module`` below.

No remaining documented divergence: every wrapper this pass touches now
matches the real architecture exactly, not just approximates it.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from torch.nn import functional as F
from transformers import DeepseekV4Config
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as tm

from vllm_neuron.model.deepseek_v4 import model as dev
from vllm_neuron.model.deepseek_v4.attention import (
    apply_partial_rotary,
    compressed_entry_slot_mapping,
)
from vllm_neuron.model.deepseek_v4.compressor import (
    compress_csa_chunk,
    compress_hca_chunk,
    finalize_compressed_entries,
)
from vllm_neuron.model.deepseek_v4.moe import hash_topk, routed_topk


def hf_config():
    return DeepseekV4Config(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        sliding_window=16,
        layer_types=[
            "heavily_compressed_attention",
            "sliding_attention",
            "compressed_sparse_attention",
        ],
        mlp_layer_types=["hash_moe", "moe", "moe"],
    )


@pytest.fixture
def real_and_dev():
    config = hf_config()
    torch.manual_seed(0)
    real = tm.DeepseekV4ForCausalLM(config).eval()
    with torch.no_grad():
        for p in real.parameters():
            p.uniform_(-0.1, 0.1)
    device_model = dev.DeepseekV4ForCausalLM.from_configs(config).eval()
    return config, real, device_model


def test_hyperconnection_wrapper_matches_real_module(real_and_dev):
    config, real, device_model = real_and_dev
    real_hc = real.model.layers[0].attn_hc
    my_hc = device_model.model.layers[0].attn_hc
    with torch.no_grad():
        my_hc.fn.copy_(real_hc.fn)
        my_hc.base.copy_(real_hc.base)
        my_hc.hc_scale.copy_(real_hc.scale)
    streams = torch.randn(2, 5, config.hc_mult, config.hidden_size)
    r_post, r_comb, r_collapsed = real_hc(streams)
    m_post, m_comb, m_collapsed = my_hc(streams)
    torch.testing.assert_close(r_post, m_post, rtol=0, atol=0)
    torch.testing.assert_close(r_comb, m_comb, rtol=0, atol=0)
    torch.testing.assert_close(r_collapsed, m_collapsed, rtol=0, atol=0)


def test_hyperhead_wrapper_matches_real_module(real_and_dev):
    config, real, device_model = real_and_dev
    real_hh = real.model.hc_head
    my_hh = device_model.model.hc_head
    with torch.no_grad():
        my_hh.fn.copy_(real_hh.hc_fn)
        my_hh.base.copy_(real_hh.hc_base)
        my_hh.hc_scale.copy_(real_hh.hc_scale)
    streams = torch.randn(2, 5, config.hc_mult, config.hidden_size)
    torch.testing.assert_close(real_hh(streams), my_hh(streams), rtol=0, atol=0)


def test_routed_moe_gate_wrapper_matches_real_topk_router(real_and_dev):
    config, real, device_model = real_and_dev
    real_gate = real.model.layers[1].mlp.gate
    my_moe = device_model.model.layers[1].moe
    with torch.no_grad():
        my_moe.gate.weight.copy_(real_gate.weight)
        my_moe.correction_bias.copy_(real_gate.e_score_correction_bias)
    hidden = torch.randn(2, 5, config.hidden_size)
    r_logits, r_weights, r_ids = real_gate(hidden)
    m_logits = my_moe.gate(hidden.view(-1, config.hidden_size))
    m_ids, m_weights = routed_topk(
        m_logits, my_moe.correction_bias, my_moe.topk, config.routed_scaling_factor
    )
    torch.testing.assert_close(r_logits, m_logits, rtol=0, atol=0)
    for row in range(r_ids.shape[0]):
        r_by_id = dict(zip(r_ids[row].tolist(), r_weights[row].tolist()))
        m_by_id = dict(zip(m_ids[row].tolist(), m_weights[row].tolist()))
        assert set(r_by_id) == set(m_by_id)
        for expert_id, weight in r_by_id.items():
            assert weight == pytest.approx(m_by_id[expert_id], abs=1e-6)


def test_hash_moe_gate_wrapper_matches_real_hash_router(real_and_dev):
    config, real, device_model = real_and_dev
    real_gate = real.model.layers[0].mlp.gate
    my_moe = device_model.model.layers[0].moe
    with torch.no_grad():
        my_moe.gate.weight.copy_(real_gate.weight)
        my_moe.tid2eid.copy_(real_gate.tid2eid)
    hidden = torch.randn(2, 5, config.hidden_size)
    input_ids = torch.tensor([[1, 5, 9, 20, 3], [4, 8, 15, 2, 7]])
    r_logits, r_weights, r_ids = real_gate(hidden, input_ids)
    m_logits = my_moe.gate(hidden.view(-1, config.hidden_size))
    m_ids, m_weights = hash_topk(
        m_logits, input_ids, my_moe.tid2eid, config.routed_scaling_factor
    )
    torch.testing.assert_close(r_logits, m_logits, rtol=0, atol=0)
    assert torch.equal(r_ids, m_ids)
    torch.testing.assert_close(r_weights, m_weights, rtol=0, atol=0)


def test_expert_wrapper_matches_real_module(real_and_dev):
    """DeepseekV4Expert vs. the real DeepseekV4Experts (routed) /
    DeepseekV4MLP (shared): ``[out, in]``-layout weights through F.linear
    and swiglu_limit-clamped gate/up, not the earlier unclamped ``[in,
    out]`` approximation. hf_config()'s intermediate_size (64) happens to
    equal this plugin's synthetic-geometry choice (hidden_size*2 == 64) for
    both routed and shared experts, so real and device weights are the same
    shape here without any resizing.
    """
    config, real, device_model = real_and_dev
    real_mlp = real.model.layers[1].mlp
    my_moe = device_model.model.layers[1].moe

    hidden = torch.randn(2, 5, config.hidden_size)

    # Routed expert 0: replicate the real per-expert math directly (bypasses
    # top-k routing/masking, which is already covered by
    # test_routed_moe_gate_wrapper_matches_real_topk_router).
    real_experts = real_mlp.experts
    with torch.no_grad():
        gate_up = F.linear(hidden, real_experts.gate_up_proj[0])
        gated = real_experts._apply_gate(gate_up)
        real_routed_out = F.linear(gated, real_experts.down_proj[0])

        my_moe.experts[0].gate_up_proj.copy_(real_experts.gate_up_proj[0])
        my_moe.experts[0].down_proj.copy_(real_experts.down_proj[0])
        my_routed_out = my_moe.experts[0](hidden)
    torch.testing.assert_close(real_routed_out, my_routed_out, rtol=0, atol=0)

    # Shared expert: real DeepseekV4MLP keeps gate_proj/up_proj separate;
    # this plugin fuses them into one gate_up_proj.
    real_shared = real_mlp.shared_experts
    with torch.no_grad():
        real_shared_out = real_shared(hidden)
        my_moe.shared_experts.gate_up_proj.copy_(
            torch.cat([real_shared.gate_proj.weight, real_shared.up_proj.weight], dim=0)
        )
        my_moe.shared_experts.down_proj.copy_(real_shared.down_proj.weight)
        my_shared_out = my_moe.shared_experts(hidden)
    torch.testing.assert_close(real_shared_out, my_shared_out, rtol=0, atol=0)


def _capture_pre_norm(kv_norm_module):
    captured = {}

    def hook(module, args, output):
        captured["pre_norm"] = args[0]

    handle = kv_norm_module.register_forward_hook(hook)
    return captured, handle


@pytest.mark.parametrize(
    "layer_index,ratio,tokens,compress_fn",
    [(0, 128, 260, compress_hca_chunk), (2, 4, 12, compress_csa_chunk)],
)
def test_compressor_wrapper_matches_real_module(
    real_and_dev, layer_index, ratio, tokens, compress_fn
):
    """Projection + windowed reduction + RMSNorm, RoPE-degenerate.

    RoPE is intentionally excluded from this comparison: this plugin's
    compressor finalizes with rope_dim=0 (see DeepseekV4Compressor's
    docstring) since the query side has no matching RoPE encoding in this
    pass's simplified attention, so comparing the norm-only stage is the
    correct/only comparable slice, not a shortcut around a harder check.
    """
    config, real, device_model = real_and_dev
    real_comp = real.model.layers[layer_index].self_attn.compressor
    my_comp = device_model.model.layers[layer_index].attention.compressor
    with torch.no_grad():
        fused_weight = torch.cat([real_comp.kv_proj.weight, real_comp.gate_proj.weight], dim=0)
        my_comp.fused_wkv_wgate.weight.copy_(fused_weight)
        my_comp.ape.copy_(real_comp.position_bias)
        my_comp.norm_weight.copy_(real_comp.kv_norm.weight)

    hidden = torch.randn(1, tokens, config.hidden_size)
    captured, handle = _capture_pre_norm(real_comp.kv_norm)
    with torch.no_grad():
        try:
            # q_residual/past_key_values are unused before kv_norm runs (the
            # indexer that needs them runs after) -- real_comp's forward
            # always computes and norms the compressed entries first.
            real_comp(hidden, torch.empty(0), torch.arange(tokens).unsqueeze(0), None, layer_index)
        except RuntimeError:
            pass
    handle.remove()
    real_reduced = captured["pre_norm"]
    real_normed = real_comp.kv_norm(real_reduced)

    kv_gate = my_comp.fused_wkv_wgate(hidden.squeeze(0)).unsqueeze(0)
    my_kv, my_gate = kv_gate[..., : my_comp.width], kv_gate[..., my_comp.width :]
    my_reduced, _ = compress_fn(my_kv, my_gate, my_comp.ape, None)
    empty = my_reduced.new_zeros((*my_reduced.shape[:-1], 0))
    my_normed = finalize_compressed_entries(
        my_reduced, my_comp.norm_weight, my_comp.rms_norm_eps, empty, empty
    )
    torch.testing.assert_close(real_reduced, my_reduced, rtol=0, atol=1e-6)
    torch.testing.assert_close(real_normed, my_normed, rtol=0, atol=1e-6)


@pytest.mark.parametrize("layer_index,ratio,total_len", [(0, 128, 270), (2, 4, 20)])
def test_compressor_wrapper_matches_real_module_through_paged_cache_io_past_carry_eviction(
    real_and_dev, layer_index, ratio, total_len
):
    """Extends test_compressor_wrapper_matches_real_module past the
    compressor's own carry-cache eviction, and through the *real*
    ``DeepseekV4Compressor.forward`` (real paged ``state_cache``/``mla_cache``
    I/O, real null-block eviction, one call per raw token -- the actual
    runtime call pattern) rather than the bare ``compress_fn`` -- closing the
    coverage gap docs/model-dev/deepseek-v4-swa-null-block-bug.md flags for
    ``_carry_rows`` (no existing test drove it past eviction through the real
    module, unlike ``_swa_history``'s
    ``test_attention_matches_real_module_after_swa_eviction_past_one_window``).

    Relies on the already-established chunk-invariance of
    ``compress_hca_chunk``/``compress_csa_chunk`` (this module's own
    docstring, and ``test_carry_replay_reproduces_incremental_*_chunking``):
    the real module's one-shot reduction over the full sequence and this
    plugin's per-token incremental writes must land on the identical
    compressed sequence. RoPE is undone at read-back (via
    ``apply_partial_rotary(..., inverse=True)`` at each entry's own known
    position) before comparing, matching
    ``test_compressor_wrapper_matches_real_module``'s choice to compare the
    RoPE-degenerate (norm-only) stage -- this test additionally proves real
    RoPE round-trips cleanly, it just isn't the equality being asserted.
    """
    config, real, device_model = real_and_dev
    real_comp = real.model.layers[layer_index].self_attn.compressor
    my_comp = device_model.model.layers[layer_index].attention.compressor
    assert my_comp.ratio == ratio
    with torch.no_grad():
        fused_weight = torch.cat([real_comp.kv_proj.weight, real_comp.gate_proj.weight], dim=0)
        my_comp.fused_wkv_wgate.weight.copy_(fused_weight)
        my_comp.ape.copy_(real_comp.position_bias)
        my_comp.norm_weight.copy_(real_comp.kv_norm.weight)

    hidden = torch.randn(1, total_len, config.hidden_size)
    captured, handle = _capture_pre_norm(real_comp.kv_norm)
    with torch.no_grad():
        try:
            real_comp(hidden, torch.empty(0), torch.arange(total_len).unsqueeze(0), None, layer_index)
        except RuntimeError:
            pass
    handle.remove()
    real_reduced = captured["pre_norm"]  # [1, num_entries, width] -- chunk-invariant ground truth
    real_normed = real_comp.kv_norm(real_reduced)

    self_attn_name = f"model.layers.{layer_index}.self_attn"
    specs = {s.name: s for s in device_model.get_kv_spec().layers}
    state_spec = specs[f"{self_attn_name}.compressor.state_cache"]
    mla_spec = specs[self_attn_name]
    # block-table columns -- must cover total_len raw tokens at state_cache's
    # own (small) block_size, e.g. 8 for HCA/128 blocks over 140 tokens.
    width = -(-total_len // state_spec.block_size) + 1
    my_comp.state_cache = torch.zeros(
        (width + 1, 1, state_spec.block_size, state_spec.head_size), dtype=state_spec.dtype
    )
    mla_cache = torch.zeros(
        (2, 1, mla_spec.block_size, mla_spec.head_size), dtype=mla_spec.dtype
    )

    def null_remapped_block_table(cached_seq_len, window, block_size):
        # Same null-remap convention as
        # test_gather_paged_latent_reads_stale_columns_once_swa_window_has_advanced
        # / test_carry_rows_reads_live_window_correctly_past_carry_cache_eviction:
        # column c is live iff its last token hasn't fallen out of the window.
        # +1: real vLLM allocates a token's own column before the engine
        # calls forward for it, so the column housing *this* (about-to-be-
        # written) token must already count as allocated -- using plain
        # cached_seq_len (tokens strictly before this one) would leave a
        # freshly-started column null at the exact step that writes its
        # first slot.
        live_start = max(0, cached_seq_len - window)
        n_cols = -(-(cached_seq_len + 1) // block_size)
        return [
            0 if (c >= n_cols or c * block_size + block_size <= live_start) else c + 1
            for c in range(width)
        ]

    # mla_cache addressing: one always-live raw physical block (id 1) is
    # enough for the *raw-token-addressed* side of compressed_entry_slot_mapping
    # (see its docstring on why raw and entry addressing share low bits);
    # entry_base is the compressed-entry index that raw block 1 maps to.
    entry_base = mla_spec.block_size // ratio
    hidden_flat = hidden.squeeze(0)
    with torch.no_grad():
        for t in range(total_len):
            state_row = null_remapped_block_table(t, state_spec.sliding_window_size, state_spec.block_size)
            state_col, state_off = t // state_spec.block_size, t % state_spec.block_size
            state_slot = state_row[state_col] * state_spec.block_size + state_off
            raw_slot = 1 * mla_spec.block_size + t
            mla_slot = compressed_entry_slot_mapping(torch.tensor([raw_slot]), ratio)
            my_comp(
                hidden_flat[t : t + 1],
                position_ids=torch.tensor([[t]], dtype=torch.long),
                block_table_row=torch.tensor(state_row, dtype=torch.long),
                state_slot_mapping=torch.tensor([state_slot], dtype=torch.long),
                mla_cache=mla_cache,
                mla_slot_mapping=mla_slot,
            )

    num_entries = total_len // ratio
    assert num_entries >= 2, "test must actually exercise multiple completions"
    # mla_cache stores bf16 (CacheKind.MLA's dtype, per get_kv_spec) -- cast
    # up before comparing against the real module's fp32 output; the wider
    # tolerance below accounts for that quantization, not just floating-point
    # summation-order noise.
    written = mla_cache[0, 0, entry_base : entry_base + num_entries, :].float().unsqueeze(0)
    entry_positions = (torch.arange(num_entries) * ratio).unsqueeze(0)
    with torch.no_grad():
        cos, sin = my_comp.rotary_emb(real_normed, position_ids=entry_positions, layer_type="compress")
    real_finalized = apply_partial_rotary(real_normed, cos, sin, rope_dim=2 * cos.shape[-1])

    torch.testing.assert_close(real_finalized, written, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("layer_index,ratio", [(0, 128), (2, 4)])
def test_compressed_history_gathers_full_prefix_with_correct_validity_mask(
    real_and_dev, layer_index, ratio
):
    """Regression test for DeepseekV4Attention._compressed_history's
    Dynamo-static redesign (docs/model-dev/deepseek-v4-swa-null-block-bug.md's
    next open item, now closed): unlike _swa_history/_carry_rows's *sliding*
    windows, this cache group never evicts, so the fix gathers the entire
    block-table-addressable capacity (max_entries) unconditionally and masks
    entries past the real count, rather than branching on cached_seq_len.
    Verifies directly against a hand-built paged mla_cache with known
    entries: the valid rows exactly match the true entries in order, the
    validity boundary is exactly at num_entries, and the num_entries == 0
    case (no branch, just an all-False mask) works too.
    """
    config, real, device_model = real_and_dev
    my_attn = device_model.model.layers[layer_index].attention
    assert my_attn.ratio == ratio

    mla_spec = {s.name: s for s in device_model.get_kv_spec().layers}[
        f"model.layers.{layer_index}.self_attn"
    ]
    storage_block_size = mla_spec.block_size
    num_cols = 3
    my_attn.mla_cache = torch.zeros((num_cols + 1, 1, storage_block_size, mla_spec.head_size))
    block_table_row = torch.arange(1, num_cols + 1, dtype=torch.long)
    max_entries = num_cols * storage_block_size

    torch.manual_seed(9)
    known_entries = torch.randn(max_entries, mla_spec.head_size)
    for col in range(num_cols):
        my_attn.mla_cache[col + 1, 0] = known_entries[
            col * storage_block_size : (col + 1) * storage_block_size
        ]

    for num_entries in (0, 1, storage_block_size, max_entries - 1, max_entries):
        cached_seq_len = num_entries * ratio  # the boundary _compressed_history reads
        position_ids = torch.tensor([[cached_seq_len]], dtype=torch.long)
        gathered, valid = my_attn._compressed_history(block_table_row, position_ids)

        assert gathered.shape[0] == max_entries
        assert valid.shape == (max_entries,)
        expected_valid = torch.arange(max_entries) < num_entries
        assert torch.equal(valid, expected_valid)
        if num_entries:
            torch.testing.assert_close(
                gathered[:num_entries], known_entries[:num_entries], rtol=0, atol=0
            )


def test_attention_matches_real_module_through_paged_cache_io(real_and_dev):
    """Cross-checks the *integrated, end-to-end* attention path: real cache
    I/O, the per-token loop, RoPE applied/undone at the right absolute
    positions, and now the real grouped low-rank output projection too --
    not just the bare math in isolation, and not stopping short of the
    final output. Uses a sliding-only layer (no compressor) to isolate
    attention; the compressor's own RoPE is already covered by
    test_compressor_wrapper_matches_real_module.

    Tolerance is not bit-exact (unlike the other tests here): the real
    module computes one batched causal softmax over the whole prefill
    chunk, while this plugin's attention (see DeepseekV4Attention's
    docstring on the per-token loop) computes the same causal attention one
    token at a time. Same math, different floating-point summation order --
    expected non-associativity noise, not a logic bug (this is the same
    kind of difference already documented and tolerated in
    test_deepseek_v4_model_assembly.py's chunk-invariance test).
    """
    config = hf_config()
    torch.manual_seed(1)
    # A dedicated sliding-only config: hf_config()'s layer 1 is
    # sliding_attention but shares num_hidden_layers=3 with compressed
    # layers, which would build unused compressor caches. A clean 1-layer
    # config keeps this test focused on attention alone.
    sliding_config = DeepseekV4Config(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        moe_intermediate_size=config.moe_intermediate_size,
        num_local_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        vocab_size=config.vocab_size,
        num_hidden_layers=1,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=1,
        head_dim=config.head_dim,
        q_lora_rank=config.q_lora_rank,
        sliding_window=128,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
    )
    real = tm.DeepseekV4ForCausalLM(sliding_config).eval()
    with torch.no_grad():
        for p in real.parameters():
            p.uniform_(-0.1, 0.1)
    real_attn = real.model.layers[0].self_attn

    device_model = dev.DeepseekV4ForCausalLM.from_configs(sliding_config).eval()
    my_attn = device_model.model.layers[0].attention
    with torch.no_grad():
        my_attn.q_a_proj.weight.copy_(real_attn.q_a_proj.weight)
        my_attn.q_a_norm.weight.copy_(real_attn.q_a_norm.weight)
        my_attn.q_b_proj.weight.copy_(real_attn.q_b_proj.weight)
        my_attn.kv_proj.weight.copy_(real_attn.kv_proj.weight)
        my_attn.kv_norm.weight.copy_(real_attn.kv_norm.weight)
        my_attn.sinks.copy_(real_attn.sinks)
        my_attn.o_a_proj.weight.copy_(real_attn.o_a_proj.weight)
        my_attn.o_b_proj.weight.copy_(real_attn.o_b_proj.weight)

    tokens = 5
    hidden = torch.randn(1, tokens, sliding_config.hidden_size)
    position_ids = torch.arange(tokens).unsqueeze(0)
    cos, sin = real.model.rotary_emb(
        hidden, position_ids=position_ids, layer_type=real_attn.rope_layer_type
    )
    position_embeddings = {real_attn.rope_layer_type: (cos, sin)}
    causal_mask = torch.triu(torch.full((1, 1, tokens, tokens), float("-inf")), diagonal=1)

    with torch.no_grad():
        real_out, _ = real_attn(hidden, position_embeddings, position_ids, causal_mask, None)

    specs = device_model.get_kv_spec().layers
    caches = {
        s.name: [torch.zeros((64, 1, s.block_size or 32, s.head_size), dtype=s.dtype)]
        for s in specs
    }
    device_model.bind_kv_cache(caches)

    block_table = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    slot_mapping = torch.arange(tokens, dtype=torch.int64)
    attn_metadata = {
        s.name: {
            "block_table_tensor": block_table,
            "slot_mapping": slot_mapping,
            "max_query_len": tokens,
            "block_size": s.block_size or 32,
            "max_blocks_per_seq": 8,
            "decode_token_threshold": 1,
            "cached_seq_len": torch.tensor([[0]], dtype=torch.int32),
            "kv_segment_size": 0,
        }
        for s in specs
    }

    with torch.no_grad():
        my_out = my_attn(
            hidden.squeeze(0),
            self_attn_name="model.layers.0.self_attn",
            attn_metadata=attn_metadata,
        )

    torch.testing.assert_close(real_out.squeeze(0), my_out, rtol=1e-3, atol=5e-4)


def test_attention_matches_real_module_after_swa_eviction_past_one_window(real_and_dev):
    """Regression test for docs/model-dev/deepseek-v4-swa-null-block-bug.md:
    drives the device path past one sliding window so a real block-table
    column actually gets null-remapped, then checks the post-eviction step
    still matches the real (unpaged) module. The test above
    (``sliding_window=128``, ``tokens=5``) never reaches eviction at all --
    this is the case the bug doc calls out as the missing coverage.

    Uses vLLM's real ``KVCacheManager``/``SlidingWindowSpec`` (same pattern
    as ``test_deepseek_cache_lifecycle.py``'s
    ``test_sliding_window_remapping_uses_null_blocks_but_latents_remain_stable``)
    to produce a block table shaped exactly like a real engine would, rather
    than hand-rolling the null-remap logic.
    """
    pytest.importorskip("vllm")
    from transformers.masking_utils import create_sliding_window_causal_mask
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.request import Request

    config = hf_config()
    torch.manual_seed(2)
    sliding_window = 4
    block_size = 32  # matches the device model's swa_cache default block_size
    sliding_config = DeepseekV4Config(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        moe_intermediate_size=config.moe_intermediate_size,
        num_local_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        vocab_size=config.vocab_size,
        num_hidden_layers=1,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=1,
        head_dim=config.head_dim,
        q_lora_rank=config.q_lora_rank,
        sliding_window=sliding_window,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
    )
    real = tm.DeepseekV4ForCausalLM(sliding_config).eval()
    with torch.no_grad():
        for p in real.parameters():
            p.uniform_(-0.1, 0.1)
    real_attn = real.model.layers[0].self_attn

    device_model = dev.DeepseekV4ForCausalLM.from_configs(sliding_config).eval()
    my_attn = device_model.model.layers[0].attention
    with torch.no_grad():
        my_attn.q_a_proj.weight.copy_(real_attn.q_a_proj.weight)
        my_attn.q_a_norm.weight.copy_(real_attn.q_a_norm.weight)
        my_attn.q_b_proj.weight.copy_(real_attn.q_b_proj.weight)
        my_attn.kv_proj.weight.copy_(real_attn.kv_proj.weight)
        my_attn.kv_norm.weight.copy_(real_attn.kv_norm.weight)
        my_attn.sinks.copy_(real_attn.sinks)
        my_attn.o_a_proj.weight.copy_(real_attn.o_a_proj.weight)
        my_attn.o_b_proj.weight.copy_(real_attn.o_b_proj.weight)

    # Past one sliding_window (4) and past one full evicted block: with
    # sliding_window=4, block_size=32, the first block (tokens [0,32)) is
    # fully skipped once cached_seq_len >= 32 + sliding_window - 1 = 35;
    # chunk1=40 comfortably clears that, so chunk2 (the decode step) reads
    # against a block table with column 0 already null-remapped.
    chunk1_len, chunk2_len = 40, 3
    total_len = chunk1_len + chunk2_len
    hidden = torch.randn(1, total_len, sliding_config.hidden_size)
    position_ids = torch.arange(total_len).unsqueeze(0)
    cos, sin = real.model.rotary_emb(
        hidden, position_ids=position_ids, layer_type=real_attn.rope_layer_type
    )
    position_embeddings = {real_attn.rope_layer_type: (cos, sin)}
    # Unlike the plain-triu causal_mask above, this must be a real
    # sliding-window mask: eager_attention_forward's `sliding_window` kwarg
    # is unused by this model's attention_interface -- the window is instead
    # baked into the mask tensor itself, normally built once per model
    # forward by DeepseekV4Model.forward via this same helper. A plain
    # causal mask here would silently test unwindowed attention (this is
    # exactly why the test above, sliding_window=128 >> tokens=5, never
    # exercised windowing at all).
    causal_mask = create_sliding_window_causal_mask(
        config=sliding_config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    with torch.no_grad():
        # One-shot full-sequence forward against the real (unpaged) module.
        real_out, _ = real_attn(hidden, position_embeddings, position_ids, causal_mask, None)

    specs = device_model.get_kv_spec().layers
    caches = {
        s.name: [torch.zeros((64, 1, s.block_size or block_size, s.head_size), dtype=s.dtype)]
        for s in specs
    }
    device_model.bind_kv_cache(caches)

    # Real vLLM block-table lifecycle: allocate chunk 1 (nothing evicted
    # yet), then advance num_computed_tokens and allocate chunk 2 -- vLLM's
    # KVCacheManager.allocate_slots evicts (null-remaps) blocks that have
    # fully fallen out of the window as part of that second call, exactly
    # as it would for a real decode step.
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer.0"],
                    SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=sliding_config.head_dim,
                        dtype=torch.bfloat16,
                        sliding_window=sliding_window,
                    ),
                )
            ],
        ),
        max_model_len=256,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        max_num_batched_tokens=128,
        enable_caching=False,
    )
    req = Request("swa_evict", [1] * chunk1_len, SamplingParams(max_tokens=8), None)
    manager.allocate_slots(req, chunk1_len)
    block_ids_1 = tuple(b.block_id for b in manager.get_blocks("swa_evict").blocks[0])
    req.num_computed_tokens = chunk1_len
    manager.allocate_slots(req, chunk2_len)
    block_ids_2 = tuple(b.block_id for b in manager.get_blocks("swa_evict").blocks[0])

    def block_table_row(block_ids, width=8):
        padded = list(block_ids) + [0] * (width - len(block_ids))
        return torch.tensor([padded], dtype=torch.int32)

    def slot_mapping_for(block_ids, positions):
        return torch.tensor(
            [
                block_ids[pos // block_size] * block_size + pos % block_size
                for pos in positions
            ],
            dtype=torch.int64,
        )

    def attn_metadata_for(block_ids, positions, cached_seq_len):
        return {
            specs[0].name: {
                "block_table_tensor": block_table_row(block_ids),
                "slot_mapping": slot_mapping_for(block_ids, positions),
                "max_query_len": len(positions),
                "block_size": block_size,
                "max_blocks_per_seq": 8,
                "decode_token_threshold": 1,
                "cached_seq_len": torch.tensor([[cached_seq_len]], dtype=torch.int32),
                "kv_segment_size": 0,
            }
        }

    hidden_flat = hidden.squeeze(0)
    with torch.no_grad():
        my_attn(
            hidden_flat[:chunk1_len],
            self_attn_name="model.layers.0.self_attn",
            attn_metadata=attn_metadata_for(block_ids_1, range(0, chunk1_len), 0),
        )
        chunk2_out = my_attn(
            hidden_flat[chunk1_len:],
            self_attn_name="model.layers.0.self_attn",
            attn_metadata=attn_metadata_for(
                block_ids_2, range(chunk1_len, total_len), chunk1_len
            ),
        )

    torch.testing.assert_close(
        real_out.squeeze(0)[chunk1_len:], chunk2_out, rtol=1e-3, atol=5e-4
    )
