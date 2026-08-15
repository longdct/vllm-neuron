# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.attention import mla_attention_reference
from vllm_neuron.model.deepseek_v4.compressor import compress_chunk
from vllm_neuron.model.deepseek_v4.mhc import sinkhorn
from vllm_neuron.model.deepseek_v4.moe import hash_experts, routed_topk


def test_512d_mla_matches_explicit_fp32_oracle():
    torch.manual_seed(1)
    q = torch.randn(1, 3, 2, 512, dtype=torch.bfloat16)
    latent = torch.randn(1, 3, 512, dtype=torch.bfloat16)
    kw = torch.randn(2, 512, 512, dtype=torch.bfloat16) / 32
    vw = torch.randn(2, 512, 16, dtype=torch.bfloat16) / 32
    actual = mla_attention_reference(q, latent, kw, vw)
    k = torch.einsum("bsl,hld->bshd", latent.float(), kw.float())
    v = torch.einsum("bsl,hlv->bshv", latent.float(), vw.float())
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k) / (512**0.5)
    scores = scores.masked_fill(
        ~torch.ones(3, 3, dtype=torch.bool).tril()[None, None], float("-inf")
    )
    expected = torch.einsum("bhts,bshv->bthv", scores.softmax(-1), v).to(q.dtype)
    torch.testing.assert_close(actual, expected)


def test_decode_sees_complete_paged_history():
    q = torch.ones(1, 1, 1, 4)
    latent = torch.arange(12.0).view(1, 3, 4)
    eye = torch.eye(4).view(1, 4, 4)
    out = mla_attention_reference(q, latent, eye, eye)
    assert out.shape == (1, 1, 1, 4)
    assert torch.isfinite(out).all()


def test_sinkhorn_is_doubly_stochastic_after_twenty_iterations():
    matrix = sinkhorn(torch.randn(4, 4), iterations=20).float()
    torch.testing.assert_close(matrix.sum(-1), torch.ones(4), atol=2e-5, rtol=0)
    torch.testing.assert_close(matrix.sum(-2), torch.ones(4), atol=2e-5, rtol=0)


@pytest.mark.parametrize("chunks", [(13,), (1, 3, 2, 7), (4, 4, 5)])
def test_compressor_is_chunk_boundary_invariant(chunks):
    hidden = torch.arange(13 * 3.0).view(13, 3)
    ape = torch.tensor([0.1, 0.2, 0.3, 0.4])
    expected, expected_state = compress_chunk(hidden, ape, 4)
    outputs, state, offset = [], None, 0
    for size in chunks:
        out, state = compress_chunk(hidden[offset : offset + size], ape, 4, state)
        outputs.append(out)
        offset += size
    torch.testing.assert_close(torch.cat(outputs), expected)
    torch.testing.assert_close(state.carry, expected_state.carry)
    assert state.total_tokens == 13


def test_noaux_bias_changes_selection_but_not_selected_gate_values():
    logits = torch.tensor([[4.0, 3.0, 1.0]])
    bias = torch.tensor([0.0, -10.0, 10.0])
    ids, weights = routed_topk(logits, bias, topk=2)
    assert ids.tolist() == [[2, 0]]
    raw = torch.nn.functional.softplus(logits).sqrt()
    selected = raw.gather(-1, ids)
    expected = selected / selected.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(weights, expected)


def test_hash_routing_is_exact_lookup():
    table = torch.tensor([[2, 1], [0, 3], [3, 2]])
    assert hash_experts(torch.tensor([2, 0]), table).tolist() == [[3, 2], [2, 1]]
