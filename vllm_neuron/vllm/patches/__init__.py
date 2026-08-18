# SPDX-License-Identifier: Apache-2.0
"""Patch registry for vllm-neuron.

This package owns the *definitions and guards* for patches against upstream
vLLM. It deliberately does not own every patch *body*: several patches must run
at a lifecycle phase this module cannot reach. ``apply_port_hold_patch()`` is
applied at ``vllm_neuron`` import time so it survives spawn-mode re-imports (the
EngineCore subprocess never calls ``check_and_update_config``), while the
all2all backend registration must run inside ``check_and_update_config`` because
it mutates config validation. Collapsing them into one hook would break both.

So patches are registered against an explicit :class:`Phase` and applied by the
code that owns that phase. Every registered entry runs at most once per process
and is expected to guard its assumptions with
:mod:`vllm_neuron.vllm.patches.guards`, so upstream drift surfaces as a startup
error rather than as the plugin silently running unpatched vLLM behaviour.

The registry semantics live in :mod:`vllm_neuron.vllm.patches.registry`, which
is kept importable without vllm/torch so they can be unit-tested directly.
"""

from vllm.logger import init_logger

from vllm_neuron.vllm.patches.guards import PatchError
from vllm_neuron.vllm.patches.registry import (
    Phase,
    applied_patches,
    mark_applied,
    register,
    registered_patches,
)
from vllm_neuron.vllm.patches.registry import apply_phase as _apply_phase

logger = init_logger(__name__)

__all__ = [
    "PatchError",
    "Phase",
    "applied_patches",
    "apply_patches",
    "apply_phase",
    "mark_applied",
    "register",
    "registered_patches",
]


def apply_phase(phase: Phase) -> None:
    """Apply every patch registered for *phase*, at most once per process."""
    _apply_phase(phase, logger=logger)


def apply_patches() -> None:
    """Apply the platform-config phase.

    Retained as the name called from ``NeuronPlatform.check_and_update_config``.
    """
    apply_phase(Phase.PLATFORM_CONFIG)


# Registers the phase entries as an import side effect. Imported last so the
# names above are bound before the decorators run.
from vllm_neuron.vllm.patches import import_patches as _import_patches  # noqa: F401
from vllm_neuron.vllm.patches import tripwires as _tripwires  # noqa: F401
