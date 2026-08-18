# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the lifecycle-phased patch registry.

These cover the semantics the rest of the patch infrastructure relies on:
apply-once, phase isolation, duplicate rejection, error wrapping, and retry
after failure.
"""

import pytest


def test_registers_and_applies(registry):
    calls = []

    @registry.register(registry.Phase.PLATFORM_CONFIG, "p:one")
    def _one():
        calls.append("one")

    registry.apply_phase(registry.Phase.PLATFORM_CONFIG)
    assert calls == ["one"]
    assert "p:one" in registry.applied_patches()


def test_applies_at_most_once(registry):
    calls = []
    registry.register(registry.Phase.PLATFORM_CONFIG, "p:once")(
        lambda: calls.append(1)
    )

    registry.apply_phase(registry.Phase.PLATFORM_CONFIG)
    registry.apply_phase(registry.Phase.PLATFORM_CONFIG)
    assert calls == [1]


def test_phases_are_isolated(registry):
    calls = []
    registry.register(registry.Phase.PLATFORM_CONFIG, "p:cfg")(
        lambda: calls.append("cfg")
    )
    registry.register(registry.Phase.DISTRIBUTED_INIT, "p:dist")(
        lambda: calls.append("dist")
    )

    registry.apply_phase(registry.Phase.PLATFORM_CONFIG)
    assert calls == ["cfg"]
    assert registry.applied_patches() == frozenset({"p:cfg"})

    registry.apply_phase(registry.Phase.DISTRIBUTED_INIT)
    assert calls == ["cfg", "dist"]


def test_applying_an_empty_phase_is_a_noop(registry):
    registry.apply_phase(registry.Phase.WORKER_STARTUP)
    assert registry.applied_patches() == frozenset()


def test_preserves_registration_order_within_a_phase(registry):
    calls = []
    for i in range(3):
        registry.register(registry.Phase.IMPORT, f"p:{i}")(
            lambda i=i: calls.append(i)
        )
    registry.apply_phase(registry.Phase.IMPORT)
    assert calls == [0, 1, 2]


class TestDuplicateRejection:
    def test_rejects_duplicate_name_in_same_phase(self, registry):
        registry.register(registry.Phase.PLATFORM_CONFIG, "p:dup")(lambda: None)
        with pytest.raises(ValueError, match="duplicate patch registration"):
            registry.register(registry.Phase.PLATFORM_CONFIG, "p:dup")(lambda: None)

    def test_rejects_duplicate_name_across_phases(self, registry):
        """Names are the identity used for reporting, so they must be global.

        Regression test: an earlier version checked only within the selected
        phase, so a name reused in another phase was accepted and then silently
        skipped, because the applied-set was keyed by name alone.
        """
        registry.register(registry.Phase.PLATFORM_CONFIG, "p:dup")(lambda: None)
        with pytest.raises(ValueError, match="duplicate patch registration"):
            registry.register(registry.Phase.DISTRIBUTED_INIT, "p:dup")(lambda: None)


class TestErrorHandling:
    def test_wraps_unexpected_exceptions_and_names_the_patch(self, registry):
        registry.register(registry.Phase.PLATFORM_CONFIG, "p:boom")(
            lambda: (_ for _ in ()).throw(KeyError("inner"))
        )
        with pytest.raises(registry.PatchError, match="p:boom"):
            registry.apply_phase(registry.Phase.PLATFORM_CONFIG)

    def test_preserves_the_original_exception_as_cause(self, registry):
        original = KeyError("inner")

        def boom():
            raise original

        registry.register(registry.Phase.PLATFORM_CONFIG, "p:boom")(boom)
        with pytest.raises(registry.PatchError) as caught:
            registry.apply_phase(registry.Phase.PLATFORM_CONFIG)
        assert caught.value.__cause__ is original

    def test_propagates_patch_error_unwrapped(self, registry):
        """A guard failure is already actionable; re-wrapping would bury it."""

        def boom():
            raise registry.PatchError("guard says no")

        registry.register(registry.Phase.PLATFORM_CONFIG, "p:guard")(boom)
        with pytest.raises(registry.PatchError, match="guard says no"):
            registry.apply_phase(registry.Phase.PLATFORM_CONFIG)

    def test_failed_patch_is_retried_not_silently_skipped(self, registry):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("transient")

        registry.register(registry.Phase.PLATFORM_CONFIG, "p:flaky")(flaky)

        with pytest.raises(registry.PatchError):
            registry.apply_phase(registry.Phase.PLATFORM_CONFIG)
        assert registry.applied_patches() == frozenset()

        registry.apply_phase(registry.Phase.PLATFORM_CONFIG)
        assert attempts == [1, 1]
        assert "p:flaky" in registry.applied_patches()

    def test_a_failure_stops_later_patches_in_the_phase(self, registry):
        calls = []
        registry.register(registry.Phase.PLATFORM_CONFIG, "p:first")(
            lambda: (_ for _ in ()).throw(RuntimeError("x"))
        )
        registry.register(registry.Phase.PLATFORM_CONFIG, "p:second")(
            lambda: calls.append("second")
        )
        with pytest.raises(registry.PatchError):
            registry.apply_phase(registry.Phase.PLATFORM_CONFIG)
        assert calls == []


class TestReporting:
    def test_mark_applied_records_unphased_patches(self, registry):
        registry.mark_applied("node_topology:in_the_same_node_as")
        assert "node_topology:in_the_same_node_as" in registry.applied_patches()

    def test_mark_applied_is_idempotent(self, registry):
        registry.mark_applied("x")
        registry.mark_applied("x")
        assert registry.applied_patches() == frozenset({"x"})

    def test_registered_patches_for_a_phase(self, registry):
        registry.register(registry.Phase.IMPORT, "p:a")(lambda: None)
        registry.register(registry.Phase.PLATFORM_CONFIG, "p:b")(lambda: None)
        assert registry.registered_patches(registry.Phase.IMPORT) == ("p:a",)
        assert registry.registered_patches() == ("p:a", "p:b")

    def test_registered_is_not_applied(self, registry):
        registry.register(registry.Phase.IMPORT, "p:a")(lambda: None)
        assert registry.registered_patches() == ("p:a",)
        assert registry.applied_patches() == frozenset()


def test_logger_is_optional_and_used_when_given(registry):
    class Recorder:
        def __init__(self):
            self.records = []

        def debug(self, msg, *args):
            self.records.append(msg % args)

    recorder = Recorder()
    registry.register(registry.Phase.IMPORT, "p:logged")(lambda: None)
    registry.apply_phase(registry.Phase.IMPORT, logger=recorder)
    assert len(recorder.records) == 1
    assert "p:logged" in recorder.records[0]
