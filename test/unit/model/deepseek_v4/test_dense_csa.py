# SPDX-License-Identifier: Apache-2.0
"""Tests for the dense-CSA equivalence bound (plan P5).

The plan asks for tests immediately below, at, and above every boundary, and for
the admission rule to use maximum possible total length rather than current
length. Both are covered here. The arithmetic is exercised against explicit
geometries rather than checkpoint values -- the module refuses to supply
compressor constants, and so does this suite.
"""

import pytest


def geom(mod, *, ratio=4, stride=4, width=4, offset=0, reserved=0):
    return mod.CompressorGeometry(
        compress_ratio=ratio,
        stride=stride,
        kernel_width=width,
        initial_offset=offset,
        reserved_entries=reserved,
    )


class TestEligibleEntries:
    def test_no_entries_before_the_first_window_begins(self, dense_csa):
        g = geom(dense_csa, offset=8)
        assert dense_csa.eligible_entries(0, g) == 0
        assert dense_csa.eligible_entries(8, g) == 0
        assert dense_csa.eligible_entries(9, g) == 1

    def test_started_mode_counts_a_partial_window(self, dense_csa):
        """Conservative: a window that has begun filling is counted."""
        g = geom(dense_csa, stride=4, width=4)
        started = dense_csa.CountingMode.STARTED
        assert dense_csa.eligible_entries(1, g, mode=started) == 1
        assert dense_csa.eligible_entries(4, g, mode=started) == 1
        assert dense_csa.eligible_entries(5, g, mode=started) == 2

    def test_complete_mode_excludes_a_partial_window(self, dense_csa):
        g = geom(dense_csa, stride=4, width=4)
        complete = dense_csa.CountingMode.COMPLETE
        assert dense_csa.eligible_entries(3, g, mode=complete) == 0
        assert dense_csa.eligible_entries(4, g, mode=complete) == 1
        assert dense_csa.eligible_entries(7, g, mode=complete) == 1
        assert dense_csa.eligible_entries(8, g, mode=complete) == 2

    def test_started_never_undercounts_complete(self, dense_csa):
        """The safety property the conservative default rests on."""
        g = geom(dense_csa, stride=4, width=16, offset=3)
        for length in range(0, 200):
            started = dense_csa.eligible_entries(length, g, mode=dense_csa.CountingMode.STARTED)
            complete = dense_csa.eligible_entries(length, g, mode=dense_csa.CountingMode.COMPLETE)
            assert started >= complete

    def test_monotonic_in_length(self, dense_csa):
        """Monotonicity is what makes the bound a single threshold."""
        g = geom(dense_csa, stride=128, width=128, offset=5, reserved=2)
        for mode in dense_csa.CountingMode:
            previous = -1
            for length in range(0, 600):
                current = dense_csa.eligible_entries(length, g, mode=mode)
                assert current >= previous
                previous = current

    def test_reserved_entries_are_always_eligible(self, dense_csa):
        g = geom(dense_csa, reserved=3)
        assert dense_csa.eligible_entries(0, g) == 3

    def test_sliding_window_layers_have_no_compressed_entries(self, dense_csa):
        g = geom(dense_csa, ratio=0, stride=0, width=0)
        assert dense_csa.eligible_entries(100_000, g) == 0

    def test_negative_length_rejected(self, dense_csa):
        with pytest.raises(ValueError, match="non-negative"):
            dense_csa.eligible_entries(-1, geom(dense_csa))


class TestBoundIsTheExactThreshold:
    """Below, at, and above -- the boundary tests the plan calls for."""

    @pytest.mark.parametrize("mode_name", ["STARTED", "COMPLETE"])
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(stride=4, width=4, offset=0, reserved=0),
            dict(stride=128, width=128, offset=0, reserved=0),
            dict(stride=4, width=16, offset=7, reserved=3),
            dict(stride=1, width=1, offset=0, reserved=0),
        ],
    )
    @pytest.mark.parametrize("index_topk", [1, 2, 512])
    def test_bound_is_the_largest_length_still_within_topk(
        self, dense_csa, kwargs, index_topk, mode_name
    ):
        mode = dense_csa.CountingMode[mode_name]
        g = geom(dense_csa, **kwargs)
        if g.reserved_entries > index_topk:
            # No safe range exists; covered by test_reserved_entries_over_topk.
            pytest.skip("reserved entries alone exceed index_topk")
        limit = dense_csa.max_dense_csa_tokens(g, index_topk, mode=mode)
        assert limit is not None

        # At the bound: still exact.
        assert dense_csa.eligible_entries(limit, g, mode=mode) <= index_topk
        # One past it: no longer exact.
        assert dense_csa.eligible_entries(limit + 1, g, mode=mode) > index_topk
        # Everywhere below it: exact.
        for length in range(max(0, limit - 3), limit + 1):
            assert dense_csa.eligible_entries(length, g, mode=mode) <= index_topk

    def test_conservative_mode_never_gives_a_longer_bound(self, dense_csa):
        g = geom(dense_csa, stride=4, width=16, offset=3, reserved=1)
        started = dense_csa.max_dense_csa_tokens(g, 64, mode=dense_csa.CountingMode.STARTED)
        complete = dense_csa.max_dense_csa_tokens(g, 64, mode=dense_csa.CountingMode.COMPLETE)
        assert started <= complete

    def test_sliding_window_layer_is_unbounded(self, dense_csa):
        g = geom(dense_csa, ratio=0, stride=0, width=0)
        assert dense_csa.max_dense_csa_tokens(g, 512) is None

    def test_reserved_entries_exhausting_topk_forbids_any_window(self, dense_csa):
        g = geom(dense_csa, stride=4, width=4, offset=6, reserved=8)
        limit = dense_csa.max_dense_csa_tokens(g, 8)
        assert limit == 6
        assert dense_csa.eligible_entries(limit, g) == 8
        assert dense_csa.eligible_entries(limit + 1, g) == 9

    def test_reserved_entries_over_topk_has_no_safe_range(self, dense_csa):
        """Not a short bound -- over budget at every length, including zero.

        Returning a positive limit here would admit requests whose eligible set
        already exceeds index_topk on an empty sequence.
        """
        g = geom(dense_csa, stride=4, width=16, offset=7, reserved=3)
        assert dense_csa.eligible_entries(0, g) > 2
        with pytest.raises(dense_csa.DenseCsaUnsupportedError, match="at any"):
            dense_csa.max_dense_csa_tokens(g, 2)

    def test_reserved_entries_exactly_at_topk_is_a_real_bound(self, dense_csa):
        """The adjacent case must still produce a usable range."""
        g = geom(dense_csa, stride=4, width=16, offset=7, reserved=3)
        limit = dense_csa.max_dense_csa_tokens(g, 3)
        assert limit == 7
        assert dense_csa.eligible_entries(limit, g) == 3
        assert dense_csa.eligible_entries(limit + 1, g) == 4

    def test_invalid_topk_rejected(self, dense_csa):
        with pytest.raises(ValueError, match="index_topk"):
            dense_csa.max_dense_csa_tokens(geom(dense_csa), 0)


class TestGeometryValidation:
    def test_compressed_layer_requires_positive_stride_and_width(self, dense_csa):
        with pytest.raises(ValueError, match="stride"):
            geom(dense_csa, ratio=4, stride=0)
        with pytest.raises(ValueError, match="kernel_width"):
            geom(dense_csa, ratio=4, width=0)

    def test_negative_fields_rejected(self, dense_csa):
        with pytest.raises(ValueError, match="initial_offset"):
            geom(dense_csa, offset=-1)
        with pytest.raises(ValueError, match="reserved_entries"):
            geom(dense_csa, reserved=-1)

    def test_geometry_has_no_default_constants(self, dense_csa):
        """A defaulted kernel width would be a guess masquerading as a bound."""
        with pytest.raises(TypeError):
            dense_csa.CompressorGeometry(compress_ratio=4)


class TestModelBound:
    def test_tightest_layer_binds(self, dense_csa):
        geometries = {
            0: geom(dense_csa, ratio=0, stride=0, width=0),          # unbounded
            1: geom(dense_csa, ratio=128, stride=128, width=128),    # loose
            2: geom(dense_csa, ratio=4, stride=4, width=4),          # tight
        }
        bound = dense_csa.model_bound(geometries, 512)
        assert bound.binding_layer == 2
        assert bound.max_total_tokens == dense_csa.max_dense_csa_tokens(
            geometries[2], 512
        )

    def test_c4_layer_binds_before_c128(self, dense_csa):
        """Finer compression accumulates entries faster, so it binds first."""
        c4 = geom(dense_csa, ratio=4, stride=4, width=4)
        c128 = geom(dense_csa, ratio=128, stride=128, width=128)
        assert dense_csa.max_dense_csa_tokens(c4, 512) < dense_csa.max_dense_csa_tokens(
            c128, 512
        )

    def test_all_sliding_window_model_is_unbounded(self, dense_csa):
        geometries = {i: geom(dense_csa, ratio=0, stride=0, width=0) for i in range(3)}
        bound = dense_csa.model_bound(geometries, 512)
        assert bound.max_total_tokens is None
        assert bound.binding_layer is None
        assert bound.permits(10**9)

    def test_empty_geometry_map_rejected(self, dense_csa):
        with pytest.raises(ValueError, match="no layer geometries"):
            dense_csa.model_bound({}, 512)


class TestAdmission:
    """The rule must use maximum possible total, not current length."""

    def bound(self, dense_csa, limit=100):
        return dense_csa.DenseCsaBound(
            max_total_tokens=limit,
            binding_layer=2,
            mode=dense_csa.CountingMode.STARTED,
        )

    def test_request_within_the_bound_is_admitted(self, dense_csa):
        assert dense_csa.check_admission(40, 60, self.bound(dense_csa)) == 100

    def test_request_one_token_over_is_rejected(self, dense_csa):
        with pytest.raises(dense_csa.DenseCsaUnsupportedError, match="101"):
            dense_csa.check_admission(41, 60, self.bound(dense_csa))

    def test_short_prompt_with_large_output_budget_is_rejected(self, dense_csa):
        """The case a current-length check would wrongly admit."""
        with pytest.raises(dense_csa.DenseCsaUnsupportedError, match="may reach"):
            dense_csa.check_admission(10, 4096, self.bound(dense_csa))

    def test_uncapped_output_is_rejected_when_a_bound_exists(self, dense_csa):
        with pytest.raises(dense_csa.DenseCsaUnsupportedError, match="no output-length cap"):
            dense_csa.check_admission(10, None, self.bound(dense_csa))

    def test_uncapped_output_is_fine_when_unbounded(self, dense_csa):
        unbounded = dense_csa.DenseCsaBound(
            max_total_tokens=None,
            binding_layer=None,
            mode=dense_csa.CountingMode.STARTED,
        )
        assert dense_csa.check_admission(10, None, unbounded) == 10

    def test_error_names_the_binding_layer_and_the_limit(self, dense_csa):
        with pytest.raises(dense_csa.DenseCsaUnsupportedError) as excinfo:
            dense_csa.check_admission(200, 0, self.bound(dense_csa))
        message = str(excinfo.value)
        assert "layer 2" in message and "100" in message

    def test_prompt_alone_at_the_bound_is_admitted(self, dense_csa):
        assert dense_csa.check_admission(100, 0, self.bound(dense_csa)) == 100

    def test_negative_inputs_rejected(self, dense_csa):
        with pytest.raises(ValueError, match="prompt_tokens"):
            dense_csa.check_admission(-1, 0, self.bound(dense_csa))
        with pytest.raises(ValueError, match="max_output_tokens"):
            dense_csa.check_admission(0, -1, self.bound(dense_csa))


class TestNaiveEstimateIsNotAssumed:
    def test_2048_is_not_hardcoded_anywhere(self, dense_csa):
        """The plan forbids shipping ~2048 before it is derived.

        The bound must fall out of the geometry, so a geometry that does not
        produce 2048 must not produce it anyway.
        """
        g = geom(dense_csa, ratio=4, stride=4, width=4)
        limit = dense_csa.max_dense_csa_tokens(g, 512, mode=dense_csa.CountingMode.STARTED)
        # stride 4 x topk 512 = 2048 under STARTED with zero offset -- but that
        # equality is a consequence of these inputs, not a constant.
        assert limit == 2048
        shifted = geom(dense_csa, ratio=4, stride=4, width=4, offset=9, reserved=2)
        assert dense_csa.max_dense_csa_tokens(shifted, 512) == 9 + 4 * 510
