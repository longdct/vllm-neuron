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


#: Branch subjects ``forward_paged`` may legitimately test, all of them known
#: at trace time. ``is_decode`` is a Python bool argument and ``world_size`` a
#: process-group size, which is what the sequence-parallel collectives switch
#: on -- the same Python-level branch ``Qwen3_5MLP.forward`` makes. Nothing
#: here reads the *contents* of a device tensor, which is the actual hazard.
PERMITTED_BRANCH_SUBJECTS = (
    "is None",
    "cached is not None",
    "%",
    "is_decode",
    "world_size",
)


# ---------------------------------------------------------------------------
# Bucket padding
# ---------------------------------------------------------------------------
#
# The runner pads every request's token block up to a bucket, so a 12-token
# prompt arrives as 12 real rows and ~2000 filler ones. Full attention shrugs
# that off -- a padded key is a column the causal mask drops. A recurrence
# cannot: every row it walks over mutates the state decode resumes from. These
# pin the three things that have to happen, by comparing against the only
# unambiguous reference, a prefill of exactly the real tokens.


def _padded_metadata(slots, real, cached_seq_len=0):
    meta = _metadata(slots, cached_seq_len=cached_seq_len)
    meta["layers.0.linear_attn"]["num_valid_tokens"] = torch.tensor(
        real, dtype=torch.int32
    )
    return meta


def test_padded_prefill_leaves_the_same_state_as_an_unpadded_one():
    """The property the whole mask exists for.

    Filler rows are deliberately *not* zeros: zeros would flatter the mask,
    since a zero row perturbs the state far less than a real one. Repeating one
    token is what the runner actually emits, and it is the worst case -- the
    l2-normalised keys of identical tokens are identical, so the delta rule's
    update is at its most aggressive precisely where the data is meaningless.
    """
    config = _config()
    real = 6
    padded = 40

    hidden = torch.randn(padded, config.hidden_size)
    hidden[real:] = hidden[real - 1]  # the runner repeats the last position

    layer = _layer(config)
    with torch.no_grad():
        layer.forward_paged(hidden[:real], _metadata([0]), is_decode=False)
    want_conv = layer.conv_state_cache[0].clone()
    want_rec = layer.recurrent_state_cache[0].clone()

    layer = _layer(config)
    with torch.no_grad():
        layer.forward_paged(hidden, _padded_metadata([0], [real]), is_decode=False)

    torch.testing.assert_close(layer.conv_state_cache[0], want_conv, **ALGEBRAIC)
    torch.testing.assert_close(layer.recurrent_state_cache[0], want_rec, **ALGEBRAIC)


def test_an_unmasked_padded_prefill_really_does_corrupt_the_state():
    """The negative control: without the mask the states must *not* agree.

    Without this, the test above would still pass if padding happened to be
    harmless, and would then be pinning nothing.
    """
    config = _config()
    real = 6
    hidden = torch.randn(40, config.hidden_size)
    hidden[real:] = hidden[real - 1]

    layer = _layer(config)
    with torch.no_grad():
        layer.forward_paged(hidden[:real], _metadata([0]), is_decode=False)
    want = layer.recurrent_state_cache[0].clone()

    layer = _layer(config)
    with torch.no_grad():
        layer.forward_paged(hidden, _metadata([0]), is_decode=False)

    assert not torch.allclose(layer.recurrent_state_cache[0], want, **ALGEBRAIC)


def test_padded_prefill_then_decode_matches_the_unpadded_sequence():
    """End to end across the seam: the state is only ever read by decode."""
    config = _config()
    real = 6
    total = real + 1

    hidden = torch.randn(40, config.hidden_size)
    hidden[real:] = hidden[real - 1]
    nxt = torch.randn(1, config.hidden_size)

    layer = _layer(config)
    with torch.no_grad():
        whole, _, _ = layer(torch.cat([hidden[:real], nxt]).unsqueeze(0))
        layer.forward_paged(hidden, _padded_metadata([0], [real]), is_decode=False)
        decode = layer.forward_paged(
            nxt, _padded_metadata([0], [1], cached_seq_len=real), is_decode=True
        )

    torch.testing.assert_close(decode, whole.squeeze(0)[-1:], **ALGEBRAIC)


def test_each_request_is_masked_at_its_own_length():
    """Two requests with different real lengths in one padded batch."""
    config = _config()
    rows = 20
    lengths = [7, 3]

    hidden = torch.randn(2, rows, config.hidden_size)
    for i, n in enumerate(lengths):
        hidden[i, n:] = hidden[i, n - 1]

    want = []
    for i, n in enumerate(lengths):
        solo = _layer(config)
        with torch.no_grad():
            solo.forward_paged(hidden[i, :n], _metadata([0]), is_decode=False)
        want.append(solo.recurrent_state_cache[0].clone())

    layer = _layer(config)
    with torch.no_grad():
        layer.forward_paged(
            hidden.reshape(2 * rows, -1),
            _padded_metadata([1, 2], lengths),
            is_decode=False,
        )

    torch.testing.assert_close(layer.recurrent_state_cache[1], want[0], **ALGEBRAIC)
    torch.testing.assert_close(layer.recurrent_state_cache[2], want[1], **ALGEBRAIC)


def test_the_conv_window_is_gathered_at_a_tensor_offset():
    """Structural: the padded conv window must not be a Python-int slice.

    ``new_state = extended[..., n : n + width]`` is the obvious spelling and is
    a data-dependent shape the moment ``n`` is an int -- §4.2 of the plan, and
    the class of bug that cost DeepSeek-V4 nine device runs. The gather has a
    fixed width and reads its offset from a tensor instead.
    """
    import inspect

    from vllm_neuron.model.qwen3_5 import nki_gdn

    source = inspect.getsource(nki_gdn._trailing_window)
    assert "torch.gather" in source
    assert ".item()" not in source
    assert "int(" not in source


def test_fresh_mask_uses_no_python_branch_on_tensor_values():
    """Structural: the fresh-sequence decision must be a mask, not an if.

    Branching on ``cached_seq_len``'s value instead of masking with it is what
    produced nine consecutive DeepSeek-V4 Dynamo blockers, so this asserts the
    shape of the code rather than only its output.
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(Qwen3_5GatedDeltaNet.forward_paged))
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            rendered = ast.unparse(node.test)
            assert any(
                subject in rendered for subject in PERMITTED_BRANCH_SUBJECTS
            ), rendered


# ---------------------------------------------------------------------------
# State storage dtype
# ---------------------------------------------------------------------------
#
# Both paged states are stored at one dtype -- see Qwen3_5TextConfig's
# mamba_state_dtype for why they cannot differ. The layer must not care which
# one it is: every consumer upcasts the state to fp32 on entry, so the storage
# dtype rounds the state once per step rather than degrading the arithmetic.


def _typed_layer(config, state_dtype, num_slots=4):
    layer = _layer(config, num_slots=num_slots)
    layer.conv_state_cache = layer.conv_state_cache.to(state_dtype)
    layer.recurrent_state_cache = layer.recurrent_state_cache.to(state_dtype)
    return layer


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_prefill_decode_round_trip_at_either_state_dtype(state_dtype):
    """Drives the write-back, which is where a dtype mismatch would surface.

    A forward-only test never reaches ``index_copy_`` into the cache, so it
    would miss exactly the failure this guards.
    """
    config = _config()
    layer = _typed_layer(config, state_dtype)
    torch.manual_seed(1)
    hidden = torch.randn(7, config.hidden_size)

    with torch.no_grad():
        prefill = layer.forward_paged(
            hidden[:-1], _metadata([0], cached_seq_len=0), is_decode=False
        )
        assert layer.conv_state_cache.dtype == state_dtype
        assert layer.recurrent_state_cache.dtype == state_dtype
        assert layer.recurrent_state_cache.abs().sum() > 0

        decode = layer.forward_paged(
            hidden[-1:], _metadata([0], cached_seq_len=6), is_decode=True
        )

    assert prefill.shape == (6, config.hidden_size)
    assert decode.shape == (1, config.hidden_size)
    assert torch.isfinite(prefill).all()
    assert torch.isfinite(decode).all()
    assert layer.conv_state_cache.dtype == state_dtype
    assert layer.recurrent_state_cache.dtype == state_dtype


def test_a_bfloat16_state_tracks_the_fp32_one():
    """bf16 storage is a rounding of the same computation, not a different one.

    Tolerance is bf16's own resolution at this scale, not a blanket rtol: the
    state is rounded once on write-back and read straight back, so the decode
    output may differ by that rounding and nothing more. A structural bug --
    a state read from the wrong slot, a mask applied at the wrong dtype --
    moves the output far outside this band.
    """
    config = _config()
    outputs = {}
    for state_dtype in (torch.float32, torch.bfloat16):
        layer = _typed_layer(config, state_dtype)
        torch.manual_seed(1)
        hidden = torch.randn(7, config.hidden_size)
        with torch.no_grad():
            layer.forward_paged(
                hidden[:-1], _metadata([0], cached_seq_len=0), is_decode=False
            )
            outputs[state_dtype] = layer.forward_paged(
                hidden[-1:], _metadata([0], cached_seq_len=6), is_decode=True
            )

    torch.testing.assert_close(
        outputs[torch.bfloat16], outputs[torch.float32], rtol=3e-2, atol=3e-2
    )


def test_a_fresh_sequence_is_masked_at_the_state_dtype():
    """The validity mask must be built per state, never once and reused.

    A mask built at one state's dtype and multiplied into the other promotes
    it silently. CPU torch just promotes, so only the device rejects it -- the
    depthwise conv sees the promoted state through torch.cat and fails with
    "nc_matmul: if one input is tfloat32/float32, both must be". Asserting the
    dtypes survive the mask keeps that from regressing without device time.
    """
    config = _config()
    layer = _typed_layer(config, torch.bfloat16)
    layer.conv_state_cache.normal_()
    layer.recurrent_state_cache.normal_()

    with torch.no_grad():
        layer.forward_paged(
            torch.randn(4, config.hidden_size),
            _metadata([1], cached_seq_len=0),
            is_decode=False,
        )

    assert layer.conv_state_cache.dtype == torch.bfloat16
    assert layer.recurrent_state_cache.dtype == torch.bfloat16
