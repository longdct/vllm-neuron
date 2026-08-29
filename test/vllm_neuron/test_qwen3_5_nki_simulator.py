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

# bf16 online arithmetic vs an fp32 reference; the repo's standing NKI tolerance.
TOL = dict(rtol=0.025, atol=0.025)


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


@pytest.mark.parametrize("channels,kernel,tokens", [(8, 4, 6), (16, 4, 12)])
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
