# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-rank memory accounting model (plan P7a).

The properties worth pinning are the ones that would make the budget misleading
rather than merely wrong: that parameter accounting is exact, that streaming
conversion is modelled as cheaper than non-streaming by the right quantity, that
an unknown dtype is fatal instead of defaulted, and that a budget whose range
straddles the capacity reports "undecided" rather than "fits".
"""

import json
import struct

import pytest


def header_bytes(entries: dict) -> bytes:
    """Build a safetensors header buffer with no tensor payload."""
    blob = json.dumps(entries).encode()
    return struct.pack("<Q", len(blob)) + blob


def meta(mb, name: str, dtype: str, shape: tuple[int, ...]):
    return mb.TensorMeta(name=name, dtype=dtype, shape=shape)


class TestHeaderParsing:
    def test_parses_dtype_and_shape_without_payload(self, memory_budget):
        data = header_bytes(
            {
                "model.layers.0.mlp.experts.w1": {
                    "dtype": "F8_E4M3",
                    "shape": [256, 2048, 4096],
                    "data_offsets": [0, 1],
                },
                "model.layers.0.input_layernorm.weight": {
                    "dtype": "BF16",
                    "shape": [4096],
                    "data_offsets": [1, 2],
                },
            }
        )
        tensors = memory_budget.parse_safetensors_header(data)
        assert {t.name for t in tensors} == {
            "model.layers.0.mlp.experts.w1",
            "model.layers.0.input_layernorm.weight",
        }
        experts = next(t for t in tensors if "experts" in t.name)
        assert experts.dtype == "F8_E4M3"
        assert experts.shape == (256, 2048, 4096)
        assert experts.numel == 256 * 2048 * 4096

    def test_metadata_key_is_skipped(self, memory_budget):
        data = header_bytes(
            {
                "__metadata__": {"format": "pt"},
                "w": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]},
            }
        )
        assert [t.name for t in memory_budget.parse_safetensors_header(data)] == ["w"]

    def test_truncated_buffer_raises(self, memory_budget):
        data = header_bytes({"w": {"dtype": "BF16", "shape": [2]}})
        with pytest.raises(ValueError, match="only"):
            memory_budget.parse_safetensors_header(data[:-4])

    def test_reads_header_from_disk_without_reading_payload(
        self, memory_budget, tmp_path
    ):
        """A real file with a large fake payload: only the header is touched."""
        path = tmp_path / "model.safetensors"
        head = header_bytes({"w": {"dtype": "BF16", "shape": [8, 8]}})
        path.write_bytes(head + b"\x00" * (4 << 20))
        tensors = memory_budget.read_safetensors_header(path)
        assert tensors[0].shape == (8, 8)

    def test_absurd_header_length_is_rejected(self, memory_budget, tmp_path):
        path = tmp_path / "bad.safetensors"
        path.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
        with pytest.raises(ValueError, match="sanity limit"):
            memory_budget.read_safetensors_header(path)


class TestDtypeWidths:
    @pytest.mark.parametrize(
        "dtype,expected",
        [("F32", 4000), ("BF16", 2000), ("F8_E4M3", 1000), ("F4", 500)],
    )
    def test_widths_scale_storage(self, memory_budget, dtype, expected):
        assert meta(memory_budget, "w", dtype, (1000,)).nbytes == expected

    def test_sub_byte_dtype_rounds_up_to_whole_bytes(self, memory_budget):
        """An odd element count of FP4 still occupies whole bytes."""
        assert meta(memory_budget, "w", "F4", (3,)).nbytes == 2

    def test_unknown_dtype_is_fatal(self, memory_budget):
        """Guessing a width would rescale the entire budget silently."""
        with pytest.raises(memory_budget.UnknownDtypeError, match="MX9"):
            meta(memory_budget, "w", "MX9", (10,)).nbytes


class TestSharding:
    def test_tensor_parallel_divides(self, memory_budget):
        layout = memory_budget.ParallelLayout(tp_size=8)
        t = meta(memory_budget, "w", "BF16", (4096, 4096))
        got = memory_budget.sharded_bytes(t, layout, memory_budget.ShardRule.TENSOR_PARALLEL)
        assert got == 4096 * 4096 * 2 // 8

    def test_replicated_is_undivided(self, memory_budget):
        layout = memory_budget.ParallelLayout(tp_size=8)
        t = meta(memory_budget, "w", "BF16", (4096,))
        got = memory_budget.sharded_bytes(t, layout, memory_budget.ShardRule.REPLICATED)
        assert got == 4096 * 2

    def test_indivisible_shape_rounds_up_rather_than_averaging(self, memory_budget):
        """Imbalance must be counted: the fullest rank sets the requirement."""
        layout = memory_budget.ParallelLayout(tp_size=3)
        t = meta(memory_budget, "w", "BF16", (10,))
        got = memory_budget.sharded_bytes(t, layout, memory_budget.ShardRule.TENSOR_PARALLEL)
        assert got == 4 * 2  # ceil(10/3) == 4, not 10/3

    def test_alignment_padding_is_counted(self, memory_budget):
        layout = memory_budget.ParallelLayout(tp_size=2, alignment_elements=128)
        t = meta(memory_budget, "w", "BF16", (300,))
        got = memory_budget.sharded_bytes(t, layout, memory_budget.ShardRule.TENSOR_PARALLEL)
        assert got == 256 * 2  # ceil(150/128)*128 == 256

    def test_invalid_layout_rejected(self, memory_budget):
        with pytest.raises(ValueError, match="tp_size"):
            memory_budget.ParallelLayout(tp_size=0)

    def test_default_classifier_routes_experts_and_norms(self, memory_budget):
        rule = memory_budget.default_shard_rule
        assert rule("model.layers.3.mlp.experts.w1") is memory_budget.ShardRule.EXPERT_PARALLEL
        assert rule("model.layers.3.input_layernorm.weight") is memory_budget.ShardRule.REPLICATED
        assert rule("model.layers.3.self_attn.wq_b.weight") is memory_budget.ShardRule.TENSOR_PARALLEL


class TestConversionPeak:
    """The quantity that decides whether BF16 loading is viable."""

    def tensors(self, mb):
        return [
            meta(mb, f"model.layers.{i}.mlp.experts.w1", "F8_E4M3", (64, 512, 512))
            for i in range(4)
        ]

    def test_streaming_peak_is_destination_plus_one_source_shard(self, memory_budget):
        layout = memory_budget.ParallelLayout(ep_size=1)
        tensors = self.tensors(memory_budget)
        source_each = tensors[0].nbytes
        peak = memory_budget.conversion_peak_bytes(
            tensors, layout, destination_width=2.0, streaming=True
        )
        assert peak == (source_each * 2) * 4 + source_each

    def test_non_streaming_holds_everything_at_once(self, memory_budget):
        layout = memory_budget.ParallelLayout(ep_size=1)
        tensors = self.tensors(memory_budget)
        source_total = sum(t.nbytes for t in tensors)
        peak = memory_budget.conversion_peak_bytes(
            tensors, layout, destination_width=2.0, streaming=False
        )
        assert peak == source_total + source_total * 2

    def test_streaming_saves_the_whole_source_but_one_shard(self, memory_budget):
        """The plan's justification for streaming, stated as an assertion."""
        layout = memory_budget.ParallelLayout(ep_size=1)
        tensors = self.tensors(memory_budget)
        streamed = memory_budget.conversion_peak_bytes(
            tensors, layout, destination_width=2.0, streaming=True
        )
        bulk = memory_budget.conversion_peak_bytes(
            tensors, layout, destination_width=2.0, streaming=False
        )
        source_total = sum(t.nbytes for t in tensors)
        assert bulk - streamed == source_total - tensors[0].nbytes

    def test_no_conversion_is_just_resident_source(self, memory_budget):
        layout = memory_budget.ParallelLayout()
        tensors = self.tensors(memory_budget)
        peak = memory_budget.conversion_peak_bytes(
            tensors, layout, destination_width=None
        )
        assert peak == sum(t.nbytes for t in tensors)

    def test_expert_parallelism_reduces_peak(self, memory_budget):
        tensors = self.tensors(memory_budget)
        one = memory_budget.conversion_peak_bytes(
            tensors, memory_budget.ParallelLayout(ep_size=1), destination_width=2.0
        )
        eight = memory_budget.conversion_peak_bytes(
            tensors, memory_budget.ParallelLayout(ep_size=8), destination_width=2.0
        )
        assert eight < one


class TestBudget:
    def tensors(self, mb):
        return [
            meta(mb, "model.layers.0.mlp.experts.w1", "F8_E4M3", (128, 512, 512)),
            meta(mb, "model.layers.0.input_layernorm.weight", "BF16", (512,)),
        ]

    def test_exact_lines_are_marked_exact(self, memory_budget):
        budget = memory_budget.build_weight_budget(
            self.tensors(memory_budget), memory_budget.ParallelLayout(tp_size=2, ep_size=2)
        )
        assert all(c.kind is memory_budget.ComponentKind.EXACT for c in budget.components)
        assert budget.exact_bytes == budget.total_low == budget.total_high

    def test_estimated_lines_widen_the_range_but_not_the_exact_portion(
        self, memory_budget
    ):
        estimate = memory_budget.Estimate(
            low=2 * memory_budget.GIB,
            high=6 * memory_budget.GIB,
            basis="compiler arena, modelled",
        )
        budget = memory_budget.build_weight_budget(
            self.tensors(memory_budget),
            memory_budget.ParallelLayout(),
            extra=[memory_budget.Component.estimated("compiler", estimate)],
        )
        assert budget.total_high - budget.total_low == 4 * memory_budget.GIB
        assert budget.exact_bytes < budget.total_low

    def test_estimate_requires_a_basis(self, memory_budget):
        """An unexplained range is not a model, it is a number with error bars."""
        with pytest.raises(ValueError, match="basis"):
            memory_budget.Estimate(low=1, high=2, basis="  ")

    def test_estimate_rejects_inverted_range(self, memory_budget):
        with pytest.raises(ValueError, match="exceeds"):
            memory_budget.Estimate(low=10, high=1, basis="x")

    def test_fits_is_undecided_when_the_range_straddles_capacity(self, memory_budget):
        """Straddling means the estimate decides -- so measure (P7b), don't guess."""
        budget = memory_budget.build_weight_budget(
            self.tensors(memory_budget),
            memory_budget.ParallelLayout(),
            extra=[
                memory_budget.Component.estimated(
                    "runtime",
                    memory_budget.Estimate(low=0, high=100 * memory_budget.GIB, basis="modelled"),
                )
            ],
        )
        assert budget.fits_in(budget.total_low + 1) is None
        assert budget.fits_in(budget.total_high) is True
        assert budget.fits_in(budget.total_low - 1) is False

    def test_budget_has_no_scalar_total(self, memory_budget):
        """A single total would silently pick a point inside the estimate."""
        budget = memory_budget.build_weight_budget(
            self.tensors(memory_budget), memory_budget.ParallelLayout()
        )
        assert not hasattr(budget, "total")

    def test_render_marks_exact_and_estimated_lines(self, memory_budget):
        budget = memory_budget.build_weight_budget(
            self.tensors(memory_budget),
            memory_budget.ParallelLayout(),
            extra=[
                memory_budget.Component.estimated(
                    "activations",
                    memory_budget.Estimate(low=1, high=2, basis="modelled"),
                )
            ],
        )
        rendered = budget.render()
        assert "= weights (resident)" in rendered
        assert "~ activations" in rendered
        assert "TOTAL per rank" in rendered

    def test_empty_metadata_rejected(self, memory_budget):
        with pytest.raises(ValueError, match="no tensor metadata"):
            memory_budget.build_weight_budget([], memory_budget.ParallelLayout())


class TestFlashScaleSanity:
    """Order-of-magnitude check against the figures the plan quotes."""

    def test_bf16_expert_payload_is_about_554_gib(self, memory_budget):
        """~277B expert params at BF16. Confirms the accounting, not the model."""
        expert_params = 277_000_000_000
        t = meta(memory_budget, "experts.w", "BF16", (expert_params,))
        assert 500 < t.nbytes / memory_budget.GIB < 560

    def test_fp8_source_to_bf16_doubles_resident_weights(self, memory_budget):
        t = [meta(memory_budget, "experts.w", "F8_E4M3", (1_000_000_000,))]
        layout = memory_budget.ParallelLayout()
        as_stored = memory_budget.build_weight_budget(t, layout)
        as_bf16 = memory_budget.build_weight_budget(t, layout, destination_width=2.0)
        resident_stored = as_stored.components[0].low
        resident_bf16 = as_bf16.components[0].low
        assert resident_bf16 == 2 * resident_stored
