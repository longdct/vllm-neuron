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

**What this does NOT cover** (documented divergences, not bugs to fix here):

- Attention structure: this plugin's ``DeepseekV4Attention`` is a
  deliberately simplified single-global-head, no-q_lora, no-RoPE stand-in
  (see `model.py`'s module docstring) — real DeepSeek-V4 attention is
  multi-head with q_lora/kv_lora down+up projections and partial RoPE. Not
  comparable at any granularity finer than "both take a hidden state and
  return an attention output."
- Expert FFN internals: this plugin's ``DeepseekV4Expert`` uses a plain
  unclamped SiLU(gate)*up SwiGLU with ``[hidden, 2*intermediate]``-layout
  weights; the real ``DeepseekV4Experts``/``DeepseekV4MLP`` clamp
  gate/up to ``swiglu_limit`` and store weights ``[out, in]``-layout (as
  ``F.linear`` expects). The *routing decision* (which experts, what
  weights) is verified here; the expert computation itself is not.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import DeepseekV4Config
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as tm

from vllm_neuron.model.deepseek_v4 import model as dev
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
        num_attention_heads=1,
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
