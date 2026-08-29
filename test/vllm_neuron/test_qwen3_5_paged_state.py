# SPDX-License-Identifier: Apache-2.0
"""The Gated DeltaNet paged-state seam.

Exercises ``forward_paged`` against the module's own unpaged form: reading a
per-request state slot, masking a fresh sequence to zero, and writing the new
state back. This is where a state hand-off bug would live, and it is checked
across the prefill/decode boundary rather than only inside one step.
"""

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig  # noqa: E402
from vllm_neuron.model.qwen3_5.gated_deltanet import (  # noqa: E402
    Qwen3_5GatedDeltaNet,
)

ALGEBRAIC = dict(rtol=1e-4, atol=1e-4)


def _config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        torch_dtype=torch.float32,
    )


def _layer(config, num_slots=4):
    torch.manual_seed(0)
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval().float()
    layer.conv_state_cache = torch.zeros(
        num_slots, config.conv_dim, config.linear_conv_kernel_dim - 1
    )
    layer.recurrent_state_cache = torch.zeros(
        num_slots,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    )
    return layer


def _metadata(slots, cached_seq_len=0):
    """Minimal per-layer metadata: a one-column block table plus the seq len."""
    return {
        "layers.0.linear_attn": {
            "block_table_tensor": torch.tensor(slots, dtype=torch.int32).view(-1, 1),
            "cached_seq_len": torch.tensor([[cached_seq_len]], dtype=torch.int32),
        }
    }


def test_state_index_is_the_single_block_table_column():
    config = _config()
    layer = _layer(config)
    meta = _metadata([2, 0, 3])
    torch.testing.assert_close(
        layer.state_index(meta), torch.tensor([2, 0, 3], dtype=torch.long)
    )


def test_unbound_state_cache_fails_loudly():
    layer = Qwen3_5GatedDeltaNet(_config(), layer_idx=0)
    with pytest.raises(RuntimeError, match="state cache was never bound"):
        layer.forward_paged(torch.zeros(2, 64), _metadata([0, 1]), is_decode=False)


def test_fresh_sequence_matches_the_unpaged_form():
    """cached_seq_len == 0 must zero the slot, whatever garbage it held."""
    config = _config()
    layer = _layer(config)

    # Poison the slot: a reused block is not zeroed by the allocator.
    layer.conv_state_cache[1].normal_()
    layer.recurrent_state_cache[1].normal_()

    hidden = torch.randn(6, config.hidden_size)
    with torch.no_grad():
        paged = layer.forward_paged(hidden, _metadata([1]), is_decode=False)
        expected, _, _ = layer(hidden.unsqueeze(0))

    torch.testing.assert_close(paged, expected.squeeze(0), **ALGEBRAIC)


def test_state_is_written_back_to_the_right_slot():
    config = _config()
    layer = _layer(config)
    hidden = torch.randn(6, config.hidden_size)

    with torch.no_grad():
        layer.forward_paged(hidden, _metadata([2]), is_decode=False)

    assert torch.any(layer.recurrent_state_cache[2] != 0)
    # Every other slot is untouched.
    for slot in (0, 1, 3):
        assert torch.all(layer.recurrent_state_cache[slot] == 0), slot
        assert torch.all(layer.conv_state_cache[slot] == 0), slot


def test_prefill_then_decode_matches_one_shot():
    """The seam the whole design exists to get right."""
    config = _config()
    layer = _layer(config)

    total = 9
    hidden = torch.randn(total, config.hidden_size)

    with torch.no_grad():
        whole, _, _ = layer(hidden.unsqueeze(0))

        prefill = layer.forward_paged(
            hidden[:-1], _metadata([0], cached_seq_len=0), is_decode=False
        )
        decode = layer.forward_paged(
            hidden[-1:], _metadata([0], cached_seq_len=total - 1), is_decode=True
        )

    torch.testing.assert_close(prefill, whole.squeeze(0)[:-1], **ALGEBRAIC)
    torch.testing.assert_close(decode, whole.squeeze(0)[-1:], **ALGEBRAIC)


def test_two_requests_keep_separate_state():
    config = _config()
    layer = _layer(config)

    torch.manual_seed(1)
    a = torch.randn(4, config.hidden_size)
    b = torch.randn(4, config.hidden_size)

    with torch.no_grad():
        # Batched through two slots at once.
        batched = layer.forward_paged(
            torch.cat([a, b]), _metadata([0, 1]), is_decode=False
        )
        solo_a, _, _ = layer(a.unsqueeze(0))
        solo_b, _, _ = layer(b.unsqueeze(0))

    torch.testing.assert_close(batched[:4], solo_a.squeeze(0), **ALGEBRAIC)
    torch.testing.assert_close(batched[4:], solo_b.squeeze(0), **ALGEBRAIC)
    # And the two slots hold different state.
    assert not torch.allclose(
        layer.recurrent_state_cache[0], layer.recurrent_state_cache[1]
    )


def test_uneven_token_split_across_requests_is_rejected():
    config = _config()
    layer = _layer(config)
    with pytest.raises(ValueError, match="do not divide evenly"):
        layer.forward_paged(
            torch.randn(5, config.hidden_size), _metadata([0, 1]), is_decode=False
        )


def test_fresh_mask_uses_no_python_branch_on_tensor_values():
    """Structural: the fresh-sequence decision must be a mask, not an if."""
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(Qwen3_5GatedDeltaNet.forward_paged))
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # The only permitted branches are on None-ness and on Python ints,
            # never on the contents of a device tensor.
            rendered = ast.unparse(node.test)
            assert (
                "is None" in rendered
                or "cached is not None" in rendered
                or "%" in rendered
            ), rendered
