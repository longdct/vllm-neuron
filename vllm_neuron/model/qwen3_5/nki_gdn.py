# SPDX-License-Identifier: Apache-2.0
"""
NKI kernels for the Qwen3.5 Gated DeltaNet.

The torch forms in ``gated_deltanet.py`` are the oracle; these are the device
path. Both live behind one dispatcher each, so a kernel that is unavailable
(older wheel, CPU-only environment) degrades to the reference rather than
failing to import -- the pattern in
``functional/attention/swa_fused.py``.

The causal convolution reuses ``nkilib``'s depthwise conv rather than being
hand-written: it ships with its own torch reference, which doubles as this
module's fallback and its test oracle.

Note on the conv and history. A zero conv state is *exactly* the reference's
left zero-padding, so there is only one code path: always concatenate the
state -- zeros for a fresh sequence -- and convolve with no padding. That
removes a branch on "is this a new sequence?", which on a traced graph would be
a data-dependent shape.
"""

import logging

import torch
import torch.nn.functional as F

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kernel availability
# ---------------------------------------------------------------------------

_wrapped_depthwise_conv1d = None
try:  # pragma: no cover - depends on the installed nkilib
    import nki
    from torch_neuronx.nki_hop import wrap_nki

    from nkilib.experimental.conv.depthwise_conv1d import (
        depthwise_conv1d_implicit_gemm,
    )

    _depthwise_conv1d_jit = nki.jit()(depthwise_conv1d_implicit_gemm)
    _wrapped_depthwise_conv1d = wrap_nki(_depthwise_conv1d_jit)
except Exception as exc:  # noqa: BLE001
    logger.debug(
        "nkilib depthwise_conv1d unavailable, Gated DeltaNet will use torch: %s", exc
    )


#: LNC grid. The depthwise conv shards on the channel dimension, and every
#: Qwen3.5 conv width is a multiple of 2, so LNC2 is always splittable.
_LNC = 2


def _can_use_conv_kernel(x: torch.Tensor, channels: int) -> bool:
    if _wrapped_depthwise_conv1d is None:
        return False
    if not can_run_kernel(x):
        return False
    # The kernel shards channels across LNC; an odd width cannot split.
    return channels % _LNC == 0


# ---------------------------------------------------------------------------
# Causal depthwise convolution
# ---------------------------------------------------------------------------


def _torch_causal_conv1d(extended, weight, activation):
    """Reference: convolve ``[state | tokens]`` with no padding."""
    channels = extended.shape[1]
    out = F.conv1d(
        extended.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=None,
        padding=0,
        groups=channels,
    )
    if activation == "silu":
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported conv activation {activation!r}")
    return out


def causal_conv1d_with_state(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv over ``[state | tokens]``.

    Args:
        hidden_states: ``[B, C, T]`` new tokens.
        conv_state: ``[B, C, kernel - 1]`` trailing inputs from earlier steps.
            All zeros represents a fresh sequence, which is exactly the
            reference's left zero-padding -- so this is not a special case.
        weight: ``[C, kernel]`` depthwise filter.
        bias: must be None; Qwen3.5's conv1d carries no bias.

    Returns:
        ``(output, new_conv_state)`` with output covering the new tokens only.
    """
    if bias is not None:
        raise ValueError("Qwen3.5's Gated DeltaNet conv1d has no bias")

    kernel = weight.shape[-1]
    seq_len = hidden_states.shape[-1]
    channels = hidden_states.shape[1]

    extended = torch.cat([conv_state, hidden_states], dim=-1)

    if _can_use_conv_kernel(hidden_states, channels):
        # nkilib wants [N, C, 1, W] and [C, 1, 1, S]; the state supplies the
        # left context, so no padding is requested.
        #
        # feature_group_count MUST equal the channel count. The kernel is
        # "depthwise" by validation, not by its default: leaving it at 1 raises
        # NCC_INKI016 ("feature_group_count must equal C"), which only surfaces
        # once a kernel actually runs -- so it is asserted by a simulator test
        # rather than trusted.
        out = _wrapped_depthwise_conv1d[_LNC](
            extended.unsqueeze(2).contiguous(),
            weight.reshape(channels, 1, 1, kernel).contiguous(),
            padding=((0, 0), (0, 0)),
            feature_group_count=channels,
        ).squeeze(2)
        if activation == "silu":
            out = F.silu(out)
        elif activation is not None:
            raise ValueError(f"Unsupported conv activation {activation!r}")
    else:
        out = _torch_causal_conv1d(extended, weight, activation)

    out = out[..., -seq_len:]
    new_state = extended[..., -(kernel - 1) :] if kernel > 1 else conv_state
    return out.to(hidden_states.dtype), new_state


def causal_conv1d(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Fresh-sequence convolution: the stateful form with a zero state."""
    kernel = weight.shape[-1]
    state = torch.zeros(
        hidden_states.shape[0],
        hidden_states.shape[1],
        kernel - 1,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    out, _ = causal_conv1d_with_state(hidden_states, state, weight, bias, activation)
    return out
