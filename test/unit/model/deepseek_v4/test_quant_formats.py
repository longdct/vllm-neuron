# SPDX-License-Identifier: Apache-2.0
"""The MXFP4 -> FP8 widening must be exact, not merely close.

TRN2 cannot execute FP4, so every routed expert weight has to be widened to
FP8. The whole staging of the FP8 work rests on that widening being lossless:
it is why milestone 4 can assert *token equality* against the BF16 oracle
instead of picking a tolerance, and it is what separates the expert conversion
from the attention conversion, which is lossy in the tail.

So these tests assert bitwise equality, and where possible they do it against
the real official shards rather than synthetic data.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.quant_formats import (
    FP4_BLOCK,
    FP4_TABLE,
    MAX_EXACT_CHANNEL_SPREAD,
    dequantize_fp4_blockwise,
    dequantize_fp8_per_channel,
    requantize_fp4_to_fp8,
    unpack_fp4,
)

OFFICIAL_SHARDS = Path("/home/ubuntu/dsv4-official-shards")
#: Shards for the only fully present layers (0, 2, 3); see the plan.
LAYER_SHARDS = {
    0: "model-00002-of-00046.safetensors",
    2: "model-00004-of-00046.safetensors",
    3: "model-00005-of-00046.safetensors",
}


def _read_tensors(path: Path, names: list[str]) -> dict[str, torch.Tensor]:
    """Read through safetensors, exactly as the real loader does.

    This matters: an E8M0 scale comes back as ``torch.float8_e8m0fnu``, whose
    ``.float()`` is the power of two itself, not the biased exponent byte.
    A hand-rolled uint8 reader silently yields values 127x too large and makes
    the exactness assertions test the wrong thing.
    """
    from safetensors import safe_open

    with safe_open(str(path), "pt") as handle:
        return {name: handle.get_tensor(name) for name in names}


def _synthetic_mxfp4(rows: int, cols: int, spread: int, seed: int = 0):
    """Build an MXFP4 weight whose per-channel exponent spread is exactly ``spread``."""
    generator = torch.Generator().manual_seed(seed)
    low = torch.randint(0, 16, (rows, cols // 2), generator=generator)
    high = torch.randint(0, 16, (rows, cols // 2), generator=generator)
    packed = (low | (high << 4)).to(torch.uint8).view(torch.int8)
    blocks = cols // FP4_BLOCK
    # Decoded powers of two, matching what safetensors yields for E8M0.
    exponent = torch.zeros((rows, blocks), dtype=torch.int32)
    if blocks > 1 and spread > 0:
        exponent[:, 1:] = -spread
    return packed, torch.exp2(exponent.float())


class TestUnpackAgainstOfficialTable:
    def test_every_fp4_code_round_trips_through_the_packing(self):
        """Low nibble first: reversing it produces plausible-looking garbage."""
        low = torch.arange(16, dtype=torch.uint8)
        high = torch.arange(16, dtype=torch.uint8).flip(0)
        packed = (low | (high << 4)).view(torch.int8).reshape(1, 16)
        values = unpack_fp4(packed)
        assert values.shape == (1, 32)
        assert torch.equal(values[0, 0::2], FP4_TABLE[low.long()])
        assert torch.equal(values[0, 1::2], FP4_TABLE[high.long()])


class TestRequantiseExactness:
    @pytest.mark.parametrize("spread", [0, 1, 3, MAX_EXACT_CHANNEL_SPREAD])
    def test_widening_is_bitwise_exact_within_the_representable_spread(self, spread):
        packed, exponent = _synthetic_mxfp4(64, 256, spread)
        reference = dequantize_fp4_blockwise(packed, exponent)

        elements, channel_scale = requantize_fp4_to_fp8(packed, exponent)
        restored = dequantize_fp8_per_channel(elements, channel_scale)

        assert elements.dtype is torch.float8_e4m3fn
        assert channel_scale.shape == (64,)
        assert torch.equal(restored, reference), (
            f"spread={spread} lost {int((restored != reference).sum())} of "
            f"{reference.numel()} values"
        )

    def test_an_unrepresentable_spread_is_refused_rather_than_degraded(self):
        packed, exponent = _synthetic_mxfp4(32, 128, MAX_EXACT_CHANNEL_SPREAD + 1)
        with pytest.raises(ValueError, match="exponent spread"):
            requantize_fp4_to_fp8(packed, exponent)

    def test_the_refusal_can_be_overridden_and_is_then_lossy(self):
        """The escape hatch exists, but must not pretend to be exact."""
        packed, exponent = _synthetic_mxfp4(32, 128, MAX_EXACT_CHANNEL_SPREAD + 4)
        reference = dequantize_fp4_blockwise(packed, exponent)
        elements, channel_scale = requantize_fp4_to_fp8(
            packed, exponent, strict=False
        )
        restored = dequantize_fp8_per_channel(elements, channel_scale)
        assert not torch.equal(restored, reference)

    def test_every_element_stays_inside_the_trn2_fp8_range(self):
        """Values above 240 have no e4m3 representation under UNSAFE_FP8FNCAST."""
        packed, exponent = _synthetic_mxfp4(64, 256, 2)
        elements, _ = requantize_fp4_to_fp8(packed, exponent)
        magnitude = elements.float().abs()
        assert torch.isfinite(magnitude).all()
        assert float(magnitude.max()) <= 240.0

    def test_a_scale_that_does_not_match_the_weight_is_rejected(self):
        packed, exponent = _synthetic_mxfp4(32, 128, 0)
        with pytest.raises(ValueError, match="does not match"):
            requantize_fp4_to_fp8(packed, exponent[:, :1])


@pytest.mark.skipif(
    not OFFICIAL_SHARDS.exists(), reason="official DeepSeek-V4 shards not on disk"
)
class TestAgainstTheOfficialCheckpoint:
    """The synthetic cases pin the contract; these pin reality."""

    @pytest.mark.parametrize("layer", sorted(LAYER_SHARDS))
    @pytest.mark.parametrize("expert", [0, 255])
    @pytest.mark.parametrize("projection", ["w1", "w2", "w3"])
    def test_real_expert_weights_widen_bitwise_exactly(
        self, layer, expert, projection
    ):
        shard = OFFICIAL_SHARDS / LAYER_SHARDS[layer]
        if not shard.exists():
            pytest.skip(f"{shard.name} not downloaded")
        prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}"
        tensors = _read_tensors(shard, [f"{prefix}.weight", f"{prefix}.scale"])
        weight = tensors[f"{prefix}.weight"]
        scale = tensors[f"{prefix}.scale"]

        reference = dequantize_fp4_blockwise(weight, scale)
        elements, channel_scale = requantize_fp4_to_fp8(weight, scale)
        restored = dequantize_fp8_per_channel(elements, channel_scale)

        assert torch.equal(restored, reference), (
            f"layer {layer} expert {expert} {projection}: "
            f"{int((restored != reference).sum())} of {reference.numel()} "
            "values changed"
        )

    def test_the_measured_spread_stays_well_inside_the_exact_regime(self):
        """If this starts failing, the exactness claim needs re-deriving."""
        shard = OFFICIAL_SHARDS / LAYER_SHARDS[0]
        if not shard.exists():
            pytest.skip(f"{shard.name} not downloaded")
        worst = 0
        for projection in ("w1", "w2", "w3"):
            name = f"layers.0.ffn.experts.0.{projection}.scale"
            scale = _read_tensors(shard, [name])[name]
            exponent = torch.log2(scale.float()).round().to(torch.int32)
            worst = max(
                worst, int((exponent.amax(dim=1)[:, None] - exponent).amax())
            )
        assert worst <= MAX_EXACT_CHANNEL_SPREAD, worst
        # Recorded so a regression is visible as a number, not just a pass.
        assert worst <= 3, f"spread grew to {worst}; plan measured 3"
