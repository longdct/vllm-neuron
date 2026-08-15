# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the scheduler-override decision.

Pure logic, no vLLM required. That these paths are still what upstream actually
resolves to is proved separately by ``TestSchedulerDefaults`` in
``test/vllm_neuron/test_upstream_compat.py``.
"""

import pytest


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_unset_scheduler_is_replaced(scheduler_selection, async_scheduling):
    """vLLM's real default is None, resolved to a class only at use time."""
    expected = (
        scheduler_selection.NEURON_ASYNC_SCHEDULER_CLS
        if async_scheduling
        else scheduler_selection.NEURON_SCHEDULER_CLS
    )
    result = scheduler_selection.resolve_neuron_scheduler_cls(None, async_scheduling)
    assert result == expected


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_explicit_upstream_default_is_replaced(scheduler_selection, async_scheduling):
    """Spelling out the default must not opt you out of the Neuron scheduler."""
    for path in scheduler_selection.UPSTREAM_DEFAULT_SCHEDULER_PATHS:
        result = scheduler_selection.resolve_neuron_scheduler_cls(
            path, async_scheduling
        )
        assert result is not None
        assert result.startswith("vllm_neuron.")


def test_async_flag_selects_the_async_neuron_scheduler(scheduler_selection):
    sync = scheduler_selection.resolve_neuron_scheduler_cls(None, False)
    asyn = scheduler_selection.resolve_neuron_scheduler_cls(None, True)
    assert sync == scheduler_selection.NEURON_SCHEDULER_CLS
    assert asyn == scheduler_selection.NEURON_ASYNC_SCHEDULER_CLS
    assert sync != asyn


def test_custom_scheduler_is_preserved(scheduler_selection):
    assert (
        scheduler_selection.resolve_neuron_scheduler_cls("my.pkg.MyScheduler", False)
        is None
    )


def test_the_neuron_scheduler_itself_is_preserved(scheduler_selection):
    """Re-entry must not thrash the setting -- check_and_update_config can rerun."""
    for path in (
        scheduler_selection.NEURON_SCHEDULER_CLS,
        scheduler_selection.NEURON_ASYNC_SCHEDULER_CLS,
    ):
        assert scheduler_selection.resolve_neuron_scheduler_cls(path, False) is None


def test_a_class_object_is_treated_as_custom(scheduler_selection):
    """scheduler_cls is ``str | type | None``; a class is an explicit choice."""

    class MyScheduler:
        ...

    assert (
        scheduler_selection.resolve_neuron_scheduler_cls(MyScheduler, False) is None
    )


def test_neuron_paths_are_importable_targets(scheduler_selection):
    """Guards against a typo in the dotted paths, which would fail only at runtime."""
    for path in (
        scheduler_selection.NEURON_SCHEDULER_CLS,
        scheduler_selection.NEURON_ASYNC_SCHEDULER_CLS,
    ):
        module, _, name = path.rpartition(".")
        assert module == "vllm_neuron.vllm.core.scheduler"
        assert name in ("NeuronScheduler", "NeuronAsyncScheduler")
