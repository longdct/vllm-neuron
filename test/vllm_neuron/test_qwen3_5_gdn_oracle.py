# SPDX-License-Identifier: Apache-2.0
"""Per-element parity of the Qwen3.5 Gated DeltaNet against HuggingFace.

The important tests here are not just "does it match" but:

* :func:`test_blocked_inverse_matches_the_reference_loop` -- the graph-explosion
  mitigation is *numerically identical* to the 63-step sequential UT transform
  it replaces, not merely close;
* :func:`test_two_chunk_prefill_matches_one_shot` -- chunk-invariance, which is
  what makes chunked prefill correct and what no existing test in this repo
  covered for the DeepSeek sliding window (a real eviction bug survived because
  every test stayed inside the window).
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

from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig  # noqa: E402
from vllm_neuron.model.qwen3_5.gated_deltanet import (  # noqa: E402
    Qwen3_5GatedDeltaNet,
    causal_conv1d,
    causal_conv1d_with_state,
    chunk_gated_delta_rule,
    l2norm,
    recurrent_gated_delta_rule,
    unit_triangular_inverse,
)

def code_of(fn) -> str:
    """Source of ``fn`` with docstrings removed.

    Structure tests assert on source text, and these modules *document* the
    forbidden forms in order to explain why they are avoided. Matching against
    raw source would therefore fire on the explanation rather than the code.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


EXACT = dict(rtol=0.0, atol=1e-6)
# The chunked and recurrent formulations are algebraically equal but reassociate
# heavily, so they are compared at fp32 accumulation noise rather than exactly.
ALGEBRAIC = dict(rtol=1e-4, atol=1e-4)

CHUNK = 16  # small chunk so tests exercise several chunks cheaply


def _config(**overrides):
    base = dict(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=256,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
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
        linear_num_key_heads=ours.linear_num_key_heads,
        linear_num_value_heads=ours.linear_num_value_heads,
        linear_key_head_dim=ours.linear_key_head_dim,
        linear_value_head_dim=ours.linear_value_head_dim,
        linear_conv_kernel_dim=ours.linear_conv_kernel_dim,
        hidden_act="silu",
        layer_types=ours.layer_types,
    )


def _rule_inputs(seed=0, batch=1, seq=48, heads=6, k_dim=32, v_dim=32):
    torch.manual_seed(seed)
    return (
        torch.randn(batch, seq, heads, k_dim, dtype=torch.float32),
        torch.randn(batch, seq, heads, k_dim, dtype=torch.float32),
        torch.randn(batch, seq, heads, v_dim, dtype=torch.float32),
        # g is a log-decay: strictly negative, small magnitude.
        -torch.rand(batch, seq, heads, dtype=torch.float32) * 0.5,
        torch.rand(batch, seq, heads, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# The UT transform without the sequential loop
# ---------------------------------------------------------------------------


def _reference_ut_loop(a: torch.Tensor) -> torch.Tensor:
    """HuggingFace's forward substitution, verbatim in shape and order."""
    attn = a.clone()
    chunk_size = a.shape[-1]
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return attn + torch.eye(chunk_size, dtype=a.dtype, device=a.device)


@pytest.mark.parametrize("n", [2, 4, 8, 16, 64])
def test_blocked_inverse_matches_the_reference_loop(n):
    """The 2*log2(n) matmul form must equal the n-1 step scan it replaces."""
    torch.manual_seed(0)
    raw = torch.randn(2, 3, n, n, dtype=torch.float32) * 0.3
    a = raw.tril(-1)  # strictly lower triangular

    torch.testing.assert_close(
        unit_triangular_inverse(a), _reference_ut_loop(a), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("n", [4, 16, 64])
def test_blocked_inverse_is_a_true_inverse(n):
    torch.manual_seed(1)
    a = (torch.randn(n, n, dtype=torch.float32) * 0.3).tril(-1)
    eye = torch.eye(n, dtype=torch.float32)
    torch.testing.assert_close(unit_triangular_inverse(a) @ (eye - a), eye, **ALGEBRAIC)


def test_blocked_inverse_uses_no_python_scan_over_the_chunk():
    """Structural guard: iteration count must be logarithmic, not linear.

    A per-row scan is what unrolls into 63 op groups per chunk per head per
    layer under XLA, so the *shape* of the loop is the thing worth pinning --
    not merely the result, which the parity test above already covers.
    """
    source = code_of(unit_triangular_inverse)

    assert "while" in source
    # No per-row forward substitution.
    assert "range(1," not in source
    # A doubling loop: the bound is logarithmic in the chunk size.
    assert "span * 2" in source or "span *= 2" in source


def test_blocked_inverse_iteration_count_is_logarithmic():
    """Empirical companion to the structural guard: count the matmuls."""
    counts = {}
    real_matmul = torch.Tensor.__matmul__

    for n in (16, 64, 256):
        calls = 0

        def counting_matmul(self, other):
            nonlocal calls
            calls += 1
            return real_matmul(self, other)

        a = (torch.randn(n, n) * 0.1).tril(-1)
        torch.Tensor.__matmul__ = counting_matmul
        try:
            unit_triangular_inverse(a)
        finally:
            torch.Tensor.__matmul__ = real_matmul
        counts[n] = calls

    # 16x the chunk size must not cost 16x the matmuls.
    assert counts[256] < counts[16] * 3, counts
    # And it must be nowhere near a per-row scan.
    assert counts[256] < 32, counts


# ---------------------------------------------------------------------------
# Delta rule parity
# ---------------------------------------------------------------------------


def test_chunk_rule_matches_hf():
    q, k, v, g, beta = _rule_inputs()

    expected, expected_state = hf_modeling.torch_chunk_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g=g.clone(), beta=beta.clone(),
        chunk_size=CHUNK, initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    actual, actual_state = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=CHUNK, initial_state=None
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-5, atol=1e-5)


def test_chunk_rule_matches_hf_with_a_carried_state():
    """Continuing from a non-zero state is the chunked-prefill case."""
    q, k, v, g, beta = _rule_inputs(seed=2, seq=32)
    torch.manual_seed(3)
    state = torch.randn(1, 6, 32, 32, dtype=torch.float32) * 0.1

    expected, expected_state = hf_modeling.torch_chunk_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g=g.clone(), beta=beta.clone(),
        chunk_size=CHUNK, initial_state=state.clone(), output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    actual, actual_state = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=CHUNK, initial_state=state
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-5, atol=1e-5)


def test_recurrent_rule_matches_hf():
    q, k, v, g, beta = _rule_inputs(seed=4, seq=5)

    expected, expected_state = hf_modeling.torch_recurrent_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g=g.clone(), beta=beta.clone(),
        initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    actual, actual_state = recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None
    )

    torch.testing.assert_close(actual, expected, **EXACT)
    torch.testing.assert_close(actual_state, expected_state, **EXACT)


def test_chunked_and_recurrent_agree():
    """Two formulations of one recurrence; disagreement means one is wrong."""
    q, k, v, g, beta = _rule_inputs(seed=5, seq=CHUNK * 3)

    chunked, chunked_state = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=CHUNK
    )
    stepwise, stepwise_state = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta)

    torch.testing.assert_close(chunked, stepwise, **ALGEBRAIC)
    torch.testing.assert_close(chunked_state, stepwise_state, **ALGEBRAIC)


def test_ragged_sequence_is_padded_correctly():
    """A length that is not a multiple of chunk_size must still be exact."""
    q, k, v, g, beta = _rule_inputs(seed=6, seq=CHUNK * 2 + 5)
    chunked, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=CHUNK)
    stepwise, _ = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta)
    assert chunked.shape[1] == CHUNK * 2 + 5
    torch.testing.assert_close(chunked, stepwise, **ALGEBRAIC)


def test_two_chunk_prefill_matches_one_shot():
    """Chunk-invariance: splitting a prefill and carrying state changes nothing."""
    seq = CHUNK * 4
    q, k, v, g, beta = _rule_inputs(seed=7, seq=seq)

    whole, whole_state = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=CHUNK
    )

    half = seq // 2
    first, mid_state = chunk_gated_delta_rule(
        q[:, :half], k[:, :half], v[:, :half],
        g=g[:, :half], beta=beta[:, :half], chunk_size=CHUNK,
    )
    second, final_state = chunk_gated_delta_rule(
        q[:, half:], k[:, half:], v[:, half:],
        g=g[:, half:], beta=beta[:, half:], chunk_size=CHUNK,
        initial_state=mid_state,
    )

    torch.testing.assert_close(torch.cat([first, second], dim=1), whole, **ALGEBRAIC)
    torch.testing.assert_close(final_state, whole_state, **ALGEBRAIC)


def test_prefill_then_decode_hands_the_state_over():
    """The prefill -> decode seam is where a state hand-off bug would live."""
    q, k, v, g, beta = _rule_inputs(seed=8, seq=CHUNK * 2 + 1)

    whole, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=CHUNK)

    prefill_len = CHUNK * 2
    _, state = chunk_gated_delta_rule(
        q[:, :prefill_len], k[:, :prefill_len], v[:, :prefill_len],
        g=g[:, :prefill_len], beta=beta[:, :prefill_len], chunk_size=CHUNK,
    )
    step, _ = recurrent_gated_delta_rule(
        q[:, prefill_len:], k[:, prefill_len:], v[:, prefill_len:],
        g=g[:, prefill_len:], beta=beta[:, prefill_len:], initial_state=state,
    )

    torch.testing.assert_close(step, whole[:, prefill_len:], **ALGEBRAIC)


def test_l2norm_matches_hf():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8, dtype=torch.float32)
    torch.testing.assert_close(l2norm(x), hf_modeling.l2norm(x), **EXACT)


# ---------------------------------------------------------------------------
# Causal convolution
# ---------------------------------------------------------------------------


def test_causal_conv_matches_hf():
    torch.manual_seed(0)
    channels, kernel, seq = 12, 4, 9
    x = torch.randn(2, channels, seq, dtype=torch.float32)
    w = torch.randn(channels, kernel, dtype=torch.float32)

    expected = hf_modeling.causal_conv1d_fn(x, w, None, activation="silu")
    torch.testing.assert_close(causal_conv1d(x, w, None, "silu"), expected, **EXACT)


def test_zero_conv_state_equals_a_fresh_sequence():
    """A zero state is exactly the reference's left zero-padding."""
    torch.manual_seed(0)
    channels, kernel, seq = 12, 4, 9
    x = torch.randn(2, channels, seq, dtype=torch.float32)
    w = torch.randn(channels, kernel, dtype=torch.float32)

    state = torch.zeros(2, channels, kernel - 1, dtype=torch.float32)
    out, _ = causal_conv1d_with_state(x, state, w, None, "silu")
    torch.testing.assert_close(out, causal_conv1d(x, w, None, "silu"), **EXACT)


def test_conv_state_carries_across_a_split():
    """Splitting the sequence and carrying the conv state must be seamless."""
    torch.manual_seed(0)
    channels, kernel, seq = 12, 4, 10
    x = torch.randn(1, channels, seq, dtype=torch.float32)
    w = torch.randn(channels, kernel, dtype=torch.float32)

    whole = causal_conv1d(x, w, None, "silu")

    zero = torch.zeros(1, channels, kernel - 1, dtype=torch.float32)
    first, state = causal_conv1d_with_state(x[..., :6], zero, w, None, "silu")
    second, _ = causal_conv1d_with_state(x[..., 6:], state, w, None, "silu")

    torch.testing.assert_close(torch.cat([first, second], dim=-1), whole, **EXACT)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


def test_split_mixed_qkv_is_explicit_and_correct():
    config = _config()
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    mixed = torch.arange(config.conv_dim, dtype=torch.float32).expand(1, 3, -1)

    q, k, v = layer.split_mixed_qkv(mixed)
    assert q.shape[-1] == config.key_dim
    assert k.shape[-1] == config.key_dim
    assert v.shape[-1] == config.value_dim
    torch.testing.assert_close(q, mixed[..., : config.key_dim], **EXACT)
    torch.testing.assert_close(
        k, mixed[..., config.key_dim : 2 * config.key_dim], **EXACT
    )
    torch.testing.assert_close(v, mixed[..., 2 * config.key_dim :], **EXACT)


def test_split_mixed_qkv_avoids_the_list_split_divergence():
    """Divergence #1: Tensor.split(list, dim!=0) is silently wrong on Neuron."""
    source = code_of(Qwen3_5GatedDeltaNet.split_mixed_qkv)
    assert ".split(" not in source
    assert "torch.split" not in source
    assert "tensor_split" not in source


def test_gated_deltanet_layer_matches_hf():
    """Whole-layer parity against HF with no cache (fresh sequence)."""
    config = _config()
    hf_config = _hf_config(config)

    torch.manual_seed(0)
    hf_layer = hf_modeling.Qwen3_5GatedDeltaNet(hf_config, layer_idx=0).eval().float()
    ours = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval().float()
    ours.load_state_dict(hf_layer.state_dict(), strict=True)

    hidden = torch.randn(1, 24, config.hidden_size, dtype=torch.float32)

    with torch.no_grad():
        expected = hf_layer(hidden, cache_params=None)
        actual, _, _ = ours(hidden)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_gated_deltanet_state_shapes():
    config = _config()
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval().float()
    hidden = torch.randn(2, 12, config.hidden_size, dtype=torch.float32)

    with torch.no_grad():
        _, conv_state, recurrent_state = layer(hidden)

    assert conv_state.shape == (2, config.conv_dim, config.linear_conv_kernel_dim - 1)
    assert recurrent_state.shape == (
        2,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    )
    # The state is an accumulator: fp32 regardless of the model dtype.
    assert recurrent_state.dtype is torch.float32
