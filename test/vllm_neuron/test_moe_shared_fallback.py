# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nki")

from vllm_neuron.functional.moe.moe_block_tkg import moe_block_tkg
from vllm_neuron.functional.moe import moe_blockwise


def test_indexed_flatten_kernel_rejects_one_token_decode_geometry(monkeypatch):
    monkeypatch.setattr(moe_blockwise, "can_run_kernel", lambda _: True)
    assert not moe_blockwise._can_use_indexed_flatten_kernel(
        T=1, tensor=torch.zeros(1, 32), f_len=0
    )


def test_indexed_flatten_kernel_accepts_first_valid_token_geometry(monkeypatch):
    monkeypatch.setattr(moe_blockwise, "can_run_kernel", lambda _: True)
    assert moe_blockwise._can_use_indexed_flatten_kernel(
        T=16, tensor=torch.zeros(16, 32), f_len=1
    )


@pytest.mark.parametrize("input_shape", [(1, 8), (2, 3, 8)])
def test_shared_expert_matches_explicit_torch_for_decode_and_prefill(input_shape):
    generator = torch.Generator().manual_seed(31 + len(input_shape))
    hidden, intermediate, experts = 8, 5, 2
    inp = torch.randn(*input_shape, generator=generator)
    gamma = torch.randn(1, hidden, generator=generator)
    router = torch.randn(hidden, experts, generator=generator)
    gate_up = torch.randn(experts, hidden, 2, intermediate, generator=generator)
    down = torch.randn(experts, intermediate, hidden, generator=generator)
    shared_gate = torch.randn(hidden, intermediate, generator=generator)
    shared_up = torch.randn(hidden, intermediate, generator=generator)
    shared_down = torch.randn(intermediate, hidden, generator=generator)
    gate_bias = torch.randn(intermediate, generator=generator)
    up_bias = torch.randn(intermediate, generator=generator)
    down_bias = torch.randn(hidden, generator=generator)

    common = dict(
        inp=inp,
        gamma=gamma,
        router_weights=router,
        expert_gate_up_weights=gate_up,
        expert_down_weights=down,
        top_k=1,
        skip_router_logits=True,
    )
    routed_only = moe_block_tkg(**common)
    actual = moe_block_tkg(
        **common,
        shared_expert_gate_w=shared_gate,
        shared_expert_up_w=shared_up,
        shared_expert_down_w=shared_down,
        shared_expert_gate_bias=gate_bias,
        shared_expert_up_bias=up_bias,
        shared_expert_down_bias=down_bias,
    )

    flattened = inp.reshape(-1, hidden).float()
    variance = flattened.square().mean(dim=-1, keepdim=True)
    normalized = flattened * torch.rsqrt(variance + 1e-6) * gamma
    expected_shared = (
        torch.nn.functional.silu(normalized @ shared_gate + gate_bias)
        * (normalized @ shared_up + up_bias)
    ) @ shared_down + down_bias
    torch.testing.assert_close(actual, routed_only + expected_shared)


def test_partial_shared_expert_weights_are_rejected():
    inp = torch.zeros(1, 8)
    with pytest.raises(ValueError, match="must be provided together"):
        moe_block_tkg(
            inp=inp,
            gamma=torch.ones(1, 8),
            router_weights=torch.zeros(8, 1),
            expert_gate_up_weights=torch.zeros(1, 8, 2, 4),
            expert_down_weights=torch.zeros(1, 4, 8),
            shared_expert_gate_w=torch.zeros(8, 4),
        )
