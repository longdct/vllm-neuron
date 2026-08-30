# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for the Qwen3.5 Gated DeltaNet, run under the NKI simulator.

Gated on ``NKI_SIMULATOR=1`` because the simulator is slow and is not built by
default; shapes are deliberately tiny for the same reason. Run with::

    NKI_SIMULATOR=1 pytest test/vllm_neuron/test_qwen3_5_nki_simulator.py -q --timeout=600

These tests exist because a wrapper that *imports* is not a wrapper that
*runs*: the depthwise conv rejects any call whose ``feature_group_count`` is
not the channel count, and that only shows up once the kernel executes.

The simulator is not a substitute for hardware -- ``docs/model-dev/nki_cpu_simulator.md``
notes CPU float arithmetic differs from a NeuronCore's, and performance is not
representative -- so this establishes numerical semantics only.
"""

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="requires explicit NKI_SIMULATOR=1",
)

nki = pytest.importorskip("nki")

from vllm_neuron.model.qwen3_5 import nki_gdn  # noqa: E402
from vllm_neuron.model.qwen3_5.gated_deltanet import (  # noqa: E402
    chunk_gated_delta_rule,
)

# bf16 online arithmetic vs an fp32 reference; the repo's standing NKI tolerance.
TOL = dict(rtol=0.025, atol=0.025)


def code_of(fn) -> str:
    """Source of ``fn`` with docstrings removed.

    Structure tests assert on source text, and these kernels *document* the
    constructs they avoid in order to explain why. Matching raw source would
    fire on the explanation rather than the code.
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


def _requires_conv_kernel():
    if nki_gdn._wrapped_depthwise_conv1d is None:
        pytest.skip("nkilib depthwise conv unavailable in this build")


def _simulate_conv(extended, weight):
    channels, kernel = weight.shape
    out = nki.simulate(nki_gdn._depthwise_conv1d_jit[nki_gdn._LNC])(
        extended.unsqueeze(2).contiguous(),
        weight.reshape(channels, 1, 1, kernel).contiguous(),
        padding=((0, 0), (0, 0)),
        feature_group_count=channels,
    )
    return torch.as_tensor(out).squeeze(2)


@pytest.mark.parametrize(
    "channels,kernel,tokens",
    [
        (8, 4, 6),
        (16, 4, 12),
        # The 0.8B runs conv_dim = 2*16*128 + 16*128 = 6144 channels over a
        # 2048 bucket. Both axes used to stop two orders of magnitude short of
        # that, so "the conv kernel is covered" meant covered at toy size only.
        (6144, 4, 12),
        (16, 4, 512),
    ],
)
def test_depthwise_conv_matches_the_torch_reference(channels, kernel, tokens):
    _requires_conv_kernel()
    torch.manual_seed(0)

    x = torch.randn(1, channels, tokens, dtype=torch.float32)
    state = torch.randn(1, channels, kernel - 1, dtype=torch.float32)
    weight = torch.randn(channels, kernel, dtype=torch.float32)

    extended = torch.cat([state, x], dim=-1)
    expected = nki_gdn._torch_causal_conv1d(extended, weight, activation=None)

    torch.testing.assert_close(_simulate_conv(extended, weight), expected, **TOL)


def test_feature_group_count_must_equal_the_channel_count():
    """Guard the trap: the kernel is depthwise by validation, not by default.

    Omitting feature_group_count does not silently do a dense convolution -- it
    raises -- but it raises only when a kernel runs, which on a normal CPU test
    run never happens.
    """
    _requires_conv_kernel()
    torch.manual_seed(0)
    channels, kernel, tokens = 8, 4, 6

    extended = torch.randn(1, channels, tokens + kernel - 1, dtype=torch.float32)
    weight = torch.randn(channels, kernel, dtype=torch.float32)

    with pytest.raises(Exception, match="depthwise|feature_group_count"):
        nki.simulate(nki_gdn._depthwise_conv1d_jit[nki_gdn._LNC])(
            extended.unsqueeze(2).contiguous(),
            weight.reshape(channels, 1, 1, kernel).contiguous(),
            padding=((0, 0), (0, 0)),
        )


def test_zero_state_reproduces_a_fresh_sequence():
    """A zero conv state is exactly the reference's left zero-padding."""
    _requires_conv_kernel()
    torch.manual_seed(1)
    channels, kernel, tokens = 8, 4, 6

    x = torch.randn(1, channels, tokens, dtype=torch.float32)
    weight = torch.randn(channels, kernel, dtype=torch.float32)
    zero = torch.zeros(1, channels, kernel - 1, dtype=torch.float32)

    simulated = _simulate_conv(torch.cat([zero, x], dim=-1), weight)

    padded = torch.nn.functional.conv1d(
        x, weight.unsqueeze(1), padding=kernel - 1, groups=channels
    )[..., :tokens]

    torch.testing.assert_close(simulated, padded, **TOL)


def test_state_carry_across_a_split_is_seamless():
    """Splitting a sequence and carrying the conv state must match one shot."""
    _requires_conv_kernel()
    torch.manual_seed(2)
    channels, kernel, tokens = 8, 4, 10

    x = torch.randn(1, channels, tokens, dtype=torch.float32)
    weight = torch.randn(channels, kernel, dtype=torch.float32)
    zero = torch.zeros(1, channels, kernel - 1, dtype=torch.float32)

    whole = _simulate_conv(torch.cat([zero, x], dim=-1), weight)

    first_ext = torch.cat([zero, x[..., :6]], dim=-1)
    first = _simulate_conv(first_ext, weight)
    carried = first_ext[..., -(kernel - 1) :]
    second = _simulate_conv(torch.cat([carried, x[..., 6:]], dim=-1), weight)

    torch.testing.assert_close(torch.cat([first, second], dim=-1), whole, **TOL)


# ---------------------------------------------------------------------------
# Chunk scan
# ---------------------------------------------------------------------------


def _requires_scan_kernel():
    if nki_gdn._wrapped_gdn_chunk_scan is None:
        pytest.skip("GDN chunk-scan kernel unavailable in this build")


def _inputs(batch, heads, chunks, chunk, k_dim, v_dim, seed=0):
    torch.manual_seed(seed)
    tokens = chunks * chunk
    return (
        torch.randn(batch, tokens, heads, k_dim, dtype=torch.float32),
        torch.randn(batch, tokens, heads, k_dim, dtype=torch.float32),
        torch.randn(batch, tokens, heads, v_dim, dtype=torch.float32),
        # g is a log-decay: negative, small magnitude.
        -torch.rand(batch, tokens, heads, dtype=torch.float32) * 0.5,
        torch.rand(batch, tokens, heads, dtype=torch.float32),
    )


def _simulate_scan(prep, state_in, lnc=None):
    """Mirror the production launch: fixed grid, rows padded to fit it.

    An explicit ``lnc`` overrides the grid so a test can still compare the two
    row-shardings against each other. That comparison stays meaningful here
    because the simulator does not go through neuronx-cc, and so is not subject
    to the grid=1 codegen failure that rules the option out on device.
    """
    rows = prep["batch"] * prep["heads"]
    if lnc is None:
        lnc = nki_gdn._SCAN_LNC
    inputs = [
        prep["q_g_T"],
        prep["k_cumdecay_T"],
        prep["attn_T"],
        prep["k_decay"],
        prep["v_base"],
        prep["g_last_rep"],
        state_in,
    ]
    # A grid of 1 divides any row count, so only the fixed grid needs padding.
    if lnc != 1:
        inputs, _ = nki_gdn.pad_rows_for_lnc(inputs, rows)
    out, state_out = nki.simulate(nki_gdn._gdn_chunk_scan_kernel[lnc])(*inputs)
    return torch.as_tensor(out)[:rows], torch.as_tensor(state_out)[:rows]


def _scan_via_simulator(q, k, v, g, beta, chunk, initial_state=None, lnc=None):
    """Mirror chunk_gated_delta_rule_nki, routing the kernel through simulate()."""
    prep = nki_gdn._prepare_chunk_scan(q, k, v, g, beta, chunk, True)
    rows = prep["batch"] * prep["heads"]
    k_dim, v_dim = prep["k_dim"], prep["v_dim"]

    if initial_state is None:
        state_in = torch.zeros(rows, k_dim, v_dim, dtype=torch.float32)
    else:
        state_in = initial_state.reshape(rows, k_dim, v_dim).float().contiguous()

    out, state_out = _simulate_scan(prep, state_in, lnc=lnc)
    batch, heads, seq_len = prep["batch"], prep["heads"], prep["seq_len"]
    out = out.reshape(batch, heads, -1, v_dim)[:, :, :seq_len].transpose(1, 2)
    return out.contiguous(), state_out.reshape(batch, heads, k_dim, v_dim)


@pytest.mark.parametrize(
    "batch,heads,chunks,chunk,dim",
    [
        (1, 1, 3, 16, 16),      # smallest useful case
        (2, 3, 3, 16, 16),      # batch and heads folded into one launch
        (1, 2, 3, 64, 128),     # the shipped geometry: chunk 64, k=v=128
        # A 2048 bucket at chunk_size 64 is 32 chunks. The scan builds its
        # chunk index with register arithmetic in the SBUF domain, which is
        # exactly the kind of thing that is right for three iterations and
        # wrong for thirty-two -- so the real count is pinned, not assumed.
        (1, 2, 32, 16, 16),
    ],
)
def test_chunk_scan_matches_the_torch_oracle(batch, heads, chunks, chunk, dim):
    """The kernel scan must agree with the independent torch implementation."""
    _requires_scan_kernel()
    q, k, v, g, beta = _inputs(batch, heads, chunks, chunk, dim, dim)

    expected, expected_state = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=chunk, initial_state=None
    )
    actual, actual_state = _scan_via_simulator(q, k, v, g, beta, chunk)

    torch.testing.assert_close(actual, expected, **TOL)
    torch.testing.assert_close(actual_state, expected_state, **TOL)


def test_chunk_scan_carries_a_nonzero_initial_state():
    """Continuing a sequence is the chunked-prefill case."""
    _requires_scan_kernel()
    batch, heads, chunks, chunk, dim = 1, 2, 3, 16, 16
    q, k, v, g, beta = _inputs(batch, heads, chunks, chunk, dim, dim, seed=1)
    torch.manual_seed(7)
    state = torch.randn(batch, heads, dim, dim, dtype=torch.float32) * 0.1

    expected, expected_state = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=chunk, initial_state=state
    )
    actual, actual_state = _scan_via_simulator(
        q, k, v, g, beta, chunk, initial_state=state
    )

    torch.testing.assert_close(actual, expected, **TOL)
    torch.testing.assert_close(actual_state, expected_state, **TOL)


def test_heads_do_not_leak_into_each_other():
    """Each row carries its own state; a shared SBUF tile would blend them."""
    _requires_scan_kernel()
    chunks, chunk, dim = 3, 16, 16
    q, k, v, g, beta = _inputs(1, 2, chunks, chunk, dim, dim, seed=3)

    both, _ = _scan_via_simulator(q, k, v, g, beta, chunk)
    head0, _ = _scan_via_simulator(
        q[:, :, :1], k[:, :, :1], v[:, :, :1], g[:, :, :1], beta[:, :, :1], chunk
    )

    torch.testing.assert_close(both[:, :, :1], head0, **TOL)


def test_scan_kernel_uses_fori_loop_not_an_unrolled_range():
    """Structural: the loop must be emitted once, not unrolled per chunk.

    This is the invariant that separates a compile measured in seconds from the
    DeepSeek-V4 MLA kernel's 2h52m with no NEFF, and it is not visible in any
    numerical result -- hence a source-level assertion.
    """
    _requires_scan_kernel()
    source = code_of(nki_gdn._gdn_chunk_scan_kernel)

    assert source.count("fori_loop") == 2      # one per row, one per chunk
    assert "affine_range" not in source
    assert "static_range" not in source
    # Register-offset addressing, never tensor[i] with a register index.
    assert "scalar_offset=" in source
    # State is carried in SBUF across iterations, per the fori_loop contract.
    assert "buffer=nl.sbuf" in source
    # Outputs only in shared HBM.
    assert source.count("buffer=nl.shared_hbm") == 2


def test_scan_kernel_allocates_nothing_shaped_by_the_sequence():
    """No [chunks, ...] SBUF tile: allocations must be per-chunk, not per-bucket.

    ``[Q, history, ...]`` allocations are what the DeepSeek-V4 Q512 investigation
    identified as the in-kernel half of the explosion, and its acceptance
    criteria call for a structural test that rejects them.
    """
    _requires_scan_kernel()
    source = code_of(nki_gdn._gdn_chunk_scan_kernel)

    for line in source.splitlines():
        if "buffer=nl.sbuf" in line or "buffer=nl.psum" in line:
            assert "chunks" not in line, line
            assert "rows" not in line, line


def test_the_scan_grid_is_a_constant_never_derived_from_rows():
    """A grid of 1 must be unreachable, whatever the geometry.

    Under LNC=2 a logical core is two physical cores and codegen is checked per
    core, so a one-program launch leaves core 1 a stub and neuronx-cc rejects
    the module with NCC_IXGM002. The grid was previously ``2 if rows % 2 == 0
    else 1``, which made it a function of the per-rank head count -- odd counts
    (3 at TP=8, 1 at TP=32) silently produced an uncompilable kernel. Pinning
    the constant is what keeps that from coming back.
    """
    assert nki_gdn._SCAN_LNC == 2

    source = code_of(nki_gdn.chunk_gated_delta_rule_nki)
    assert "_wrapped_gdn_chunk_scan[_SCAN_LNC]" in source, source


@pytest.mark.parametrize("rows", [1, 2, 3, 6, 96])
def test_odd_rows_are_padded_to_the_grid_with_an_inert_row(rows):
    """Padding, not a smaller grid, is how an odd row count is absorbed."""
    tensors = [torch.randn(rows, 4, 4) for _ in range(3)]
    padded, padded_rows = nki_gdn.pad_rows_for_lnc(tensors, rows)

    assert padded_rows % nki_gdn._SCAN_LNC == 0
    assert padded_rows == rows + (rows % 2)
    for original, out in zip(tensors, padded):
        assert out.shape[0] == padded_rows
        # The real rows must survive untouched...
        torch.testing.assert_close(out[:rows], original)
        # ...and anything appended must be inert.
        assert out[rows:].abs().sum() == 0


def test_padding_does_not_change_the_scan_result():
    """The inert row must not perturb the rows that carry real work.

    Rows are independent, so this should hold by construction -- which is
    exactly why it is worth asserting: a kernel that ever let one row read
    another would show up here and nowhere else in this file. 3 rows is the
    TP=8 geometry (1 batch x 3 value heads per rank), and is odd, so this is
    the case that used to drop to a grid of 1.
    """
    _requires_scan_kernel()
    q, k, v, g, beta = _inputs(1, 3, 3, 16, 16, 16)

    expected, expected_state = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=16, initial_state=None
    )
    actual, actual_state = _scan_via_simulator(q, k, v, g, beta, 16)

    torch.testing.assert_close(actual, expected, **TOL)
    torch.testing.assert_close(actual_state, expected_state, **TOL)


def test_lnc_two_and_lnc_one_agree():
    """Row sharding must not change the answer, only who computes it.

    Rows are independent -- each carries its own recurrent state -- so splitting
    them across the two programs needs no communication. If a program ever read
    another's state this is where it would show.
    """
    _requires_scan_kernel()
    chunks, chunk, dim = 3, 16, 16
    q, k, v, g, beta = _inputs(2, 3, chunks, chunk, dim, dim, seed=11)

    one, one_state = _scan_via_simulator(q, k, v, g, beta, chunk, lnc=1)
    two, two_state = _scan_via_simulator(q, k, v, g, beta, chunk, lnc=2)

    torch.testing.assert_close(two, one, **TOL)
    torch.testing.assert_close(two_state, one_state, **TOL)


def test_lnc_two_covers_every_row():
    """A dropped or double-counted row would leave a slot at its initial value."""
    _requires_scan_kernel()
    chunks, chunk, dim = 2, 16, 16
    q, k, v, g, beta = _inputs(2, 4, chunks, chunk, dim, dim, seed=12)

    out, state = _scan_via_simulator(q, k, v, g, beta, chunk, lnc=2)

    # Every (batch, head) row must have been written by some program.
    assert out.shape[0] == 2 and out.shape[2] == 4
    for b in range(2):
        for h in range(4):
            assert torch.any(state[b, h] != 0), (b, h)


def test_scan_kernel_shards_rows_across_lnc_programs():
    """Structural: the loop bounds must carry the program base.

    ``row_start + r`` inside the body would be ``int + VirtualRegister``, which
    the tracer rejects; folding the base into the bounds is the documented fix.
    """
    _requires_scan_kernel()
    source = code_of(nki_gdn._gdn_chunk_scan_kernel)

    assert "num_programs" in source
    assert "program_id" in source
    assert "fori_loop(row_start, row_start + rows_per_program" in source
