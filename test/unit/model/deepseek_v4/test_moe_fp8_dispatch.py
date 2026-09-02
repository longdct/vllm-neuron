# SPDX-License-Identifier: Apache-2.0
"""FP8 experts must reach `shard_on_i`, never `shard_on_block`.

`shard_on_block` accepts `gate_up_proj_scale`/`down_proj_scale` and silently
ignores them: it compiles, runs, and returns finite output roughly 1e10x too
large because it uses the raw e4m3 elements. Measured on device in
`tools/deepseek_v4/probe_fp8_moe.py`, three ways -- four
(quantization_type x scale-order) combinations were bit-identical, doubling
both scales changed nothing, and replacing them with all-ones changed nothing.

That is the worst kind of bug to reintroduce, because nothing raises. So the
dispatch is pinned here, on CPU, where it costs nothing to check.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nkilib")

from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import QuantizationType

from vllm_neuron.model.deepseek_v4.model import DeepseekV4MoE
from vllm_neuron.model.deepseek_v4.parallel import DeepseekV4ParallelTopology


class _Group:
    size = 1
    rank_in_group = 0

    def all_reduce(self, value):
        return value


def _fp8_moe(monkeypatch, *, experts=8, hidden=512, intermediate=256):
    """A minimally-populated MoE in FP8 mode, without constructing the model."""
    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.kind = "routed_moe"
    moe.topk = 2
    moe.num_experts = experts
    moe.num_local_experts = experts
    moe.local_start, moe.local_end = 0, experts
    moe.ep_degree = 1
    moe.expert_tp_degree = 1
    moe.expert_tp_rank = 0
    moe.routed_scaling_factor = 1.0
    moe.topology = DeepseekV4ParallelTopology(tp_degree=1, ep_degree=1)
    moe.tp_group = _Group()
    moe.expert_weight_dtype = torch.float8_e4m3fn
    moe.expert_fp8 = True
    moe.routed_gate_up = torch.nn.Parameter(
        torch.zeros(experts, hidden, 2, intermediate, dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    moe.routed_down = torch.nn.Parameter(
        torch.zeros(experts, intermediate, hidden, dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    moe.routed_gate_up_scale = torch.nn.Parameter(
        torch.ones(experts, 1, 2 * intermediate), requires_grad=False
    )
    moe.routed_down_scale = torch.nn.Parameter(
        torch.ones(experts, 1, hidden), requires_grad=False
    )
    return moe


def _capture_moe_cte_call(monkeypatch, moe, tokens):
    """Run `_forward_nki` far enough to record the kernel arguments."""
    captured = {}

    def fake_moe_cte(**kwargs):
        captured.update(kwargs)
        return torch.zeros(
            kwargs["hidden_states"].shape[0],
            kwargs["hidden_states"].shape[-1],
            dtype=torch.bfloat16,
        )

    def fake_mapping(**kwargs):
        blocks = 4
        return (
            torch.zeros(1),
            torch.zeros(blocks * kwargs["block_size"], dtype=torch.int32),
            torch.zeros(blocks, dtype=torch.int32),
            torch.zeros(blocks + 2),
        )

    import vllm_neuron.functional as NF

    # `_forward_nki` resolves the MoE communication domain before dispatching;
    # neither group exists in a unit test, and neither affects which kernel is
    # chosen, which is all this file is about.
    monkeypatch.setattr(
        "vllm.distributed.get_tp_group", lambda: _Group(), raising=False
    )
    monkeypatch.setattr(
        "vllm_neuron.parallel.neuron_parallel_state.get_neuron_ep_degree",
        lambda: 1,
        raising=False,
    )
    monkeypatch.setattr(NF, "moe_cte", fake_moe_cte, raising=False)
    monkeypatch.setattr(NF, "build_blockwise_mapping", fake_mapping, raising=False)
    monkeypatch.setattr(
        "vllm_neuron.model.deepseek_v4.model.can_run_kernel", lambda t: True
    )

    hidden = torch.zeros(tokens, moe.routed_gate_up.shape[1], dtype=torch.bfloat16)
    affinities = torch.zeros(tokens, moe.num_experts)
    moe._forward_nki(hidden, affinities)
    return captured


class TestFp8Dispatch:
    def test_fp8_experts_use_shard_on_i_not_shard_on_block(self, monkeypatch):
        moe = _fp8_moe(monkeypatch)
        captured = _capture_moe_cte_call(monkeypatch, moe, tokens=8)
        assert captured["implementation"] is MoECTEImplementation.shard_on_i, (
            "shard_on_block ignores FP8 scales silently -- see the module "
            "docstring"
        )

    def test_fp8_passes_both_scales_and_names_the_quantization(self, monkeypatch):
        moe = _fp8_moe(monkeypatch)
        captured = _capture_moe_cte_call(monkeypatch, moe, tokens=8)
        assert captured["gate_up_proj_scale"] is moe.routed_gate_up_scale
        assert captured["down_proj_scale"] is moe.routed_down_scale
        assert captured["quantization_type"] is QuantizationType.ROW

    def test_fp8_block_size_satisfies_the_shard_on_i_constraint(self, monkeypatch):
        """`shard_on_i` asserts block_size % 256 == 0; decode's default is 128."""
        moe = _fp8_moe(monkeypatch)
        for tokens in (1, 8, 128, 512):
            captured = _capture_moe_cte_call(monkeypatch, moe, tokens=tokens)
            assert captured["block_size"] % 256 == 0, tokens

    def test_scale_shapes_match_the_kernel_contract(self, monkeypatch):
        """[E, 1, 2*I] and [E, 1, H]; the flat axis is shard-major."""
        moe = _fp8_moe(monkeypatch, experts=8, hidden=512, intermediate=256)
        assert tuple(moe.routed_gate_up_scale.shape) == (8, 1, 512)
        assert tuple(moe.routed_down_scale.shape) == (8, 1, 512)

    def test_fp8_rejects_weights_that_are_not_e4m3(self, monkeypatch):
        moe = _fp8_moe(monkeypatch)
        moe.routed_gate_up = torch.nn.Parameter(
            moe.routed_gate_up.float().to(torch.bfloat16), requires_grad=False
        )
        with pytest.raises(RuntimeError, match="float8_e4m3fn"):
            _capture_moe_cte_call(monkeypatch, moe, tokens=8)


class TestBf16Unchanged:
    def test_bf16_still_uses_shard_on_block_and_passes_no_scales(self, monkeypatch):
        moe = _fp8_moe(monkeypatch)
        moe.expert_fp8 = False
        moe.expert_weight_dtype = torch.bfloat16
        moe.routed_gate_up = torch.nn.Parameter(
            torch.zeros(8, 512, 2, 256, dtype=torch.bfloat16)
        )
        moe.routed_down = torch.nn.Parameter(
            torch.zeros(8, 256, 512, dtype=torch.bfloat16)
        )
        captured = _capture_moe_cte_call(monkeypatch, moe, tokens=8)
        assert captured["implementation"] is MoECTEImplementation.shard_on_block
        assert captured["gate_up_proj_scale"] is None
        assert captured["down_proj_scale"] is None
        assert captured["quantization_type"] is QuantizationType.NONE
        # The BF16 decode block must not grow: that would be a silent
        # performance regression on the path FP8 is not involved in.
        assert captured["block_size"] == 128


class TestDummyScaleInitialisation:
    """`hc_scale` is a learned mHC weight, not a quantization scale.

    Pinning it to 1.0 amplifies the residual streams into non-finite logits;
    on-device argmax then returns an out-of-range token id and the request
    dies far away in `_validate_token_ids` with no mention of initialization.
    The same name collision is documented at `HyperConnection` for the
    checkpoint loader, so it is worth a test in both places.
    """

    def test_only_quantization_scales_are_pinned_to_one(self):
        from vllm_neuron.model.deepseek_v4.model import (
            _QUANTIZATION_SCALE_PARAMETERS,
        )

        assert "routed_gate_up_scale" in _QUANTIZATION_SCALE_PARAMETERS
        assert "routed_down_scale" in _QUANTIZATION_SCALE_PARAMETERS
        assert "hc_scale" not in _QUANTIZATION_SCALE_PARAMETERS
        assert "hc_head_scale" not in _QUANTIZATION_SCALE_PARAMETERS

    def test_dummy_init_leaves_hc_scale_random_and_fixes_expert_scales(self):
        from vllm_neuron.model.deepseek_v4.model import (
            _initialize_dummy_parameters,
        )

        module = torch.nn.Module()
        module.hc_scale = torch.nn.Parameter(torch.ones(3))
        module.routed_down_scale = torch.nn.Parameter(torch.zeros(4, 1, 8))
        module.weight = torch.nn.Parameter(torch.zeros(4, 4))

        _initialize_dummy_parameters(module)

        assert torch.equal(module.routed_down_scale, torch.ones(4, 1, 8))
        # hc_scale went through the ordinary dummy draw, so it is no longer
        # its constructed value and is small like any other weight.
        assert not torch.equal(module.hc_scale, torch.ones(3))
        assert float(module.hc_scale.abs().max()) <= 1e-3
