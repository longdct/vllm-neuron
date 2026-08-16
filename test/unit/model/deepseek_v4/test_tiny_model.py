# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.tiny_model import TinyDeepseekV4ForCausalLM


@pytest.fixture
def model():
    torch.manual_seed(7)
    return TinyDeepseekV4ForCausalLM().eval()


def test_all_structural_variants_forward_and_decode(model):
    assert [(layer.ratio, layer.moe.kind) for layer in model.layers] == [
        (128, "hash_moe"),
        (0, "routed_moe"),
        (4, "routed_moe"),
        (128, "routed_moe"),
    ]
    logits, state = model(torch.tensor([1, 2, 3, 4]))
    decoded, state = model(torch.tensor([5]), state)
    assert logits.shape == (4, model.config.vocab_size)
    assert decoded.shape == (1, model.config.vocab_size)
    assert state.num_tokens == 5
    assert torch.isfinite(decoded).all()


@pytest.mark.parametrize("chunks", [(5,), (1, 4), (2, 1, 2)])
def test_chunked_logits_match_unchunked_exactly(model, chunks):
    tokens = torch.tensor([1, 2, 3, 4, 5])
    expected, _ = model(tokens)
    state, outputs, offset = None, [], 0
    for size in chunks:
        output, state = model(tokens[offset : offset + size], state)
        outputs.append(output)
        offset += size
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)


def test_abort_state_is_not_reused(model):
    _, abandoned = model(torch.tensor([1, 2, 3]))
    fresh, fresh_state = model(torch.tensor([4]))
    leaked, _ = model(torch.tensor([4]), abandoned)
    assert fresh_state.num_tokens == 1
    assert not torch.equal(fresh, leaked)
