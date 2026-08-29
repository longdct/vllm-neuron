# SPDX-License-Identifier: Apache-2.0
"""Per-element parity of the Qwen3.5 full-attention numerics against HuggingFace.

Everything runs on CPU in fp32 and is compared element by element, never in
aggregate: the DeepSeek-V4 compressed-entry off-by-one was bit-exact at six of
eight positions, so every summary statistic read it as floating-point noise
(docs/model-dev/deepseek-v4-real-weight-validation.md).

The reference is the installed
``transformers.models.qwen3_5.modeling_qwen3_5``, which is the same code the
released checkpoints were validated against.
"""

import pytest
import torch

hf_modeling = pytest.importorskip(
    "transformers.models.qwen3_5.modeling_qwen3_5",
    reason="requires a transformers build carrying the Qwen3.5 architecture",
)
from transformers.models.qwen3_5.configuration_qwen3_5 import (  # noqa: E402
    Qwen3_5TextConfig as HFQwen3_5TextConfig,
)

from vllm_neuron.model.qwen3_5.attention import (  # noqa: E402
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5RotaryEmbedding,
    apply_output_gate,
    apply_partial_rotary_pos_emb,
    compute_interleaved_mrope,
    double_freqs,
    split_query_and_gate,
)
from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig  # noqa: E402

# fp32 on CPU: a real defect must not be able to hide inside a bf16 tolerance.
EXACT = dict(rtol=0.0, atol=1e-6)


def _small_config(**overrides):
    """A small but structurally faithful config: 3:1 schedule, partial RoPE."""
    base = dict(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=256,
        rms_norm_eps=1e-6,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        torch_dtype=torch.float32,
    )
    base.update(overrides)
    return Qwen3_5TextConfig(**base)


def _hf_config(ours: Qwen3_5TextConfig):
    return HFQwen3_5TextConfig(
        hidden_size=ours.hidden_size,
        intermediate_size=ours.intermediate_size,
        num_hidden_layers=ours.num_hidden_layers,
        num_attention_heads=ours.num_attention_heads,
        num_key_value_heads=ours.num_key_value_heads,
        head_dim=ours.head_dim,
        vocab_size=ours.vocab_size,
        rms_norm_eps=ours.rms_norm_eps,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": ours.rope_theta,
            "mrope_interleaved": True,
            "mrope_section": ours.mrope_section,
            "partial_rotary_factor": ours.partial_rotary_factor,
        },
        partial_rotary_factor=ours.partial_rotary_factor,
    )


# ---------------------------------------------------------------------------
# Normalization: the two conventions
# ---------------------------------------------------------------------------


def test_rmsnorm_matches_hf_once_the_plus_one_is_folded():
    """HF stores a zero-init weight and applies (1 + w); we fold the 1 in."""
    torch.manual_seed(0)
    dim, eps = 64, 1e-6
    x = torch.randn(3, 7, dim, dtype=torch.float32)
    hf_weight = torch.randn(dim, dtype=torch.float32) * 0.1

    hf = hf_modeling.Qwen3_5RMSNorm(dim, eps=eps)
    with torch.no_grad():
        hf.weight.copy_(hf_weight)

    ours = Qwen3_5RMSNorm(dim, eps, torch.float32)
    with torch.no_grad():
        ours.weight.copy_(1.0 + hf_weight)  # the fold

    torch.testing.assert_close(ours(x), hf(x), **EXACT)


def test_unfolded_weight_would_be_wrong():
    """Guard the trap: loading HF's tensor as-is must NOT match."""
    torch.manual_seed(0)
    dim = 64
    x = torch.randn(2, 5, dim)
    hf_weight = torch.randn(dim) * 0.1

    hf = hf_modeling.Qwen3_5RMSNorm(dim, eps=1e-6)
    with torch.no_grad():
        hf.weight.copy_(hf_weight)

    naive = Qwen3_5RMSNorm(dim, 1e-6, torch.float32)
    with torch.no_grad():
        naive.weight.copy_(hf_weight)  # forgot the fold

    assert not torch.allclose(naive(x), hf(x), rtol=1e-3, atol=1e-3)


def test_gated_rmsnorm_matches_hf_with_no_fold():
    """The gated norm uses the ordinary convention -- folding here would break it."""
    torch.manual_seed(0)
    dim, eps = 32, 1e-6
    x = torch.randn(12, dim, dtype=torch.float32)
    gate = torch.randn(12, dim, dtype=torch.float32)
    weight = torch.randn(dim, dtype=torch.float32) * 0.1 + 1.0

    hf = hf_modeling.Qwen3_5RMSNormGated(dim, eps=eps)
    with torch.no_grad():
        hf.weight.copy_(weight)

    ours = Qwen3_5RMSNormGated(dim, eps, torch.float32)
    with torch.no_grad():
        ours.weight.copy_(weight)

    torch.testing.assert_close(ours(x, gate), hf(x, gate), **EXACT)


def test_gated_rmsnorm_sharded_variance_matches_whole():
    """Splitting the value dim and all-reducing sum-of-squares must be exact."""
    torch.manual_seed(0)
    dim = 32
    x = torch.randn(6, dim, dtype=torch.float32)
    gate = torch.randn(6, dim, dtype=torch.float32)

    ours = Qwen3_5RMSNormGated(dim, 1e-6, torch.float32)
    whole = ours(x, gate)

    # Emulate a 2-way value-dim split: each rank holds half the columns but the
    # all-reduced sum of squares over the full width.
    sum_sq = x.float().pow(2).sum(-1, keepdim=True)
    halves = []
    for lo, hi in ((0, dim // 2), (dim // 2, dim)):
        shard = Qwen3_5RMSNormGated(hi - lo, 1e-6, torch.float32)
        with torch.no_grad():
            shard.weight.copy_(ours.weight[lo:hi])
        halves.append(shard(x[:, lo:hi], gate[:, lo:hi], sum_squares=sum_sq, dim_size=dim))

    torch.testing.assert_close(torch.cat(halves, dim=-1), whole, **EXACT)


# ---------------------------------------------------------------------------
# Rotary embedding
# ---------------------------------------------------------------------------


def test_interleaved_mrope_matches_hf():
    torch.manual_seed(0)
    config = _small_config()
    section = config.mrope_section
    band = config.rotary_dim // 2
    assert sum(section) == band

    freqs = torch.randn(3, 2, 9, band, dtype=torch.float32)

    hf_rope = hf_modeling.Qwen3_5TextRotaryEmbedding(_hf_config(config))
    expected = hf_rope.apply_interleaved_mrope(freqs.clone(), section)
    actual = compute_interleaved_mrope(freqs.clone(), section)

    torch.testing.assert_close(actual, expected, **EXACT)


def test_rotary_cos_sin_matches_hf():
    config = _small_config()
    hf_rope = hf_modeling.Qwen3_5TextRotaryEmbedding(_hf_config(config))
    ours = Qwen3_5RotaryEmbedding(config)

    positions = torch.arange(11, dtype=torch.long)
    dummy = torch.zeros(1, 11, config.hidden_size, dtype=torch.float32)

    hf_cos, hf_sin = hf_rope(dummy, positions[None, :].expand(3, -1))
    our_cos, our_sin = ours(positions, dtype=torch.float32)

    # HF returns the already-doubled rotary_dim width; ours returns the band.
    our_cos_full, our_sin_full = double_freqs(our_cos, our_sin)
    assert our_cos_full.shape[-1] == config.rotary_dim

    torch.testing.assert_close(our_cos_full, hf_cos[0], **EXACT)
    torch.testing.assert_close(our_sin_full, hf_sin[0], **EXACT)


def test_partial_rotary_matches_hf_and_leaves_the_tail_untouched():
    torch.manual_seed(0)
    config = _small_config()
    rotary_dim = config.rotary_dim
    heads, seq, head_dim = 4, 9, config.head_dim

    q = torch.randn(1, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(1, 2, seq, head_dim, dtype=torch.float32)

    ours_rope = Qwen3_5RotaryEmbedding(config)
    cos, sin = ours_rope(torch.arange(seq), dtype=torch.float32)
    cos_full, sin_full = double_freqs(cos, sin)

    hf_q, hf_k = hf_modeling.apply_rotary_pos_emb(
        q, k, cos_full[None, ...], sin_full[None, ...], unsqueeze_dim=1
    )

    our_q = apply_partial_rotary_pos_emb(q, cos_full, sin_full, rotary_dim)
    our_k = apply_partial_rotary_pos_emb(k, cos_full, sin_full, rotary_dim)

    torch.testing.assert_close(our_q, hf_q, **EXACT)
    torch.testing.assert_close(our_k, hf_k, **EXACT)

    # The non-rotated tail must survive byte-for-byte -- this is the channel
    # range Neuron once lowered to zeros in a rank-4 cat (divergence #2).
    torch.testing.assert_close(our_q[..., rotary_dim:], q[..., rotary_dim:], **EXACT)
    assert torch.count_nonzero(our_q[..., rotary_dim:]) == our_q[..., rotary_dim:].numel()


def test_partial_rotary_actually_rotates_only_the_leading_channels():
    """A trailing-channel implementation would pass a shape check but be wrong."""
    torch.manual_seed(0)
    config = _small_config()
    rotary_dim = config.rotary_dim
    x = torch.randn(1, 2, 5, config.head_dim, dtype=torch.float32)

    cos, sin = Qwen3_5RotaryEmbedding(config)(torch.arange(5), dtype=torch.float32)
    cos_full, sin_full = double_freqs(cos, sin)
    out = apply_partial_rotary_pos_emb(x, cos_full, sin_full, rotary_dim)

    # Leading channels changed (position 0 is identity, so look past it).
    assert not torch.allclose(out[:, :, 1:, :rotary_dim], x[:, :, 1:, :rotary_dim])
    # Trailing channels did not.
    torch.testing.assert_close(out[..., rotary_dim:], x[..., rotary_dim:], **EXACT)


# ---------------------------------------------------------------------------
# Query/gate interleaving
# ---------------------------------------------------------------------------


def test_split_query_and_gate_matches_hf_chunk():
    """HF views as [..., heads, 2*head_dim] then chunks; the gate is per head."""
    torch.manual_seed(0)
    heads, head_dim, tokens = 4, 64, 7
    projected = torch.randn(1, tokens, heads * head_dim * 2, dtype=torch.float32)

    hf_q, hf_gate = torch.chunk(
        projected.view(1, tokens, -1, head_dim * 2), 2, dim=-1
    )
    hf_gate_flat = hf_gate.reshape(1, tokens, -1)

    our_q, our_gate = split_query_and_gate(projected, heads, head_dim)

    torch.testing.assert_close(our_q, hf_q, **EXACT)
    torch.testing.assert_close(our_gate.reshape(1, tokens, -1), hf_gate_flat, **EXACT)


def test_flat_half_split_would_mix_heads():
    """Guard the trap: splitting the projection in half is not the same thing."""
    torch.manual_seed(0)
    heads, head_dim, tokens = 4, 64, 7
    projected = torch.randn(1, tokens, heads * head_dim * 2, dtype=torch.float32)

    our_q, _ = split_query_and_gate(projected, heads, head_dim)
    naive_q = projected[..., : heads * head_dim].view(1, tokens, heads, head_dim)

    assert not torch.allclose(our_q, naive_q)


def test_output_gate_matches_hf():
    torch.manual_seed(0)
    attn = torch.randn(3, 8, dtype=torch.float32)
    gate = torch.randn(3, 8, dtype=torch.float32)
    torch.testing.assert_close(
        apply_output_gate(attn, gate), attn * torch.sigmoid(gate), **EXACT
    )


# ---------------------------------------------------------------------------
# End-to-end attention layer
# ---------------------------------------------------------------------------


def test_full_attention_layer_matches_hf():
    """Whole-layer parity: projections, QK-norm, partial RoPE, gate, o_proj."""
    torch.manual_seed(0)
    config = _small_config()
    hf_config = _hf_config(config)
    hf_config._attn_implementation = "eager"

    hf_attn = hf_modeling.Qwen3_5Attention(hf_config, layer_idx=0).eval().float()

    tokens = 9
    hidden = torch.randn(1, tokens, config.hidden_size, dtype=torch.float32)
    positions = torch.arange(tokens)

    rope = Qwen3_5RotaryEmbedding(config)
    cos, sin = rope(positions, dtype=torch.float32)
    cos_full, sin_full = double_freqs(cos, sin)

    causal = torch.full((tokens, tokens), float("-inf")).triu(1)
    with torch.no_grad():
        hf_out, _ = hf_attn(
            hidden,
            position_embeddings=(cos_full[None, ...], sin_full[None, ...]),
            attention_mask=causal[None, None, ...],
        )

    # --- our path, reusing HF's weights ---
    head_dim = config.head_dim
    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads

    with torch.no_grad():
        qg = hidden @ hf_attn.q_proj.weight.T
        k = hidden @ hf_attn.k_proj.weight.T
        v = hidden @ hf_attn.v_proj.weight.T

        query, gate = split_query_and_gate(qg, n_q, head_dim)
        gate = gate.reshape(1, tokens, -1)

        q_norm = Qwen3_5RMSNorm(head_dim, config.rms_norm_eps, torch.float32)
        k_norm = Qwen3_5RMSNorm(head_dim, config.rms_norm_eps, torch.float32)
        q_norm.weight.copy_(1.0 + hf_attn.q_norm.weight)
        k_norm.weight.copy_(1.0 + hf_attn.k_norm.weight)

        query = q_norm(query).transpose(1, 2)
        key = k_norm(k.view(1, tokens, n_kv, head_dim)).transpose(1, 2)
        value = v.view(1, tokens, n_kv, head_dim).transpose(1, 2)

        query = apply_partial_rotary_pos_emb(
            query, cos_full, sin_full, config.rotary_dim
        )
        key = apply_partial_rotary_pos_emb(key, cos_full, sin_full, config.rotary_dim)

        key = key.repeat_interleave(n_q // n_kv, dim=1)
        value = value.repeat_interleave(n_q // n_kv, dim=1)

        scores = (query @ key.transpose(-1, -2)) * (head_dim**-0.5)
        scores = scores + causal
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        attn_out = (probs @ value).transpose(1, 2).reshape(1, tokens, -1)

        attn_out = apply_output_gate(attn_out, gate)
        our_out = attn_out @ hf_attn.o_proj.weight.T

    torch.testing.assert_close(our_out, hf_out, rtol=0.0, atol=1e-5)


def test_full_attention_layer_diverges_without_the_gate():
    """The output gate is not cosmetic -- dropping it must change the result."""
    torch.manual_seed(0)
    config = _small_config()
    hf_config = _hf_config(config)
    hf_config._attn_implementation = "eager"
    hf_attn = hf_modeling.Qwen3_5Attention(hf_config, layer_idx=0).eval().float()

    tokens = 6
    hidden = torch.randn(1, tokens, config.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        qg = hidden @ hf_attn.q_proj.weight.T
        _, gate = split_query_and_gate(qg, config.num_attention_heads, config.head_dim)
    gate = gate.reshape(1, tokens, -1)

    fake_attn = torch.randn(1, tokens, gate.shape[-1], dtype=torch.float32)
    assert not torch.allclose(apply_output_gate(fake_attn, gate), fake_attn)
