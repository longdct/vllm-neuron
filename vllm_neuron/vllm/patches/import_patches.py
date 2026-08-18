# SPDX-License-Identifier: Apache-2.0
"""Import-phase patch registrations.

``port_hold`` and ``pin_memory`` must be applied at ``vllm_neuron`` import time
so they survive the spawn-mode re-import in worker subprocesses -- the
EngineCore subprocess never calls ``check_and_update_config``. Upstream applies
both by calling their ``apply_*`` functions directly from ``vllm_neuron``'s
``__init__``; registering them here instead routes them through the same
apply-once bookkeeping as every other patch, so ``applied_patches()`` reports
the whole picture rather than silently omitting the two earliest ones.

The patch bodies stay in their own modules; this only declares *when* they run.
"""

from vllm_neuron.vllm.patches.registry import Phase, register


@register(Phase.IMPORT, "port_hold")
def _port_hold() -> None:
    from vllm_neuron.vllm.patches.port_hold_patch import apply_port_hold_patch

    apply_port_hold_patch()


@register(Phase.IMPORT, "pin_memory")
def _pin_memory() -> None:
    # vLLM's pooling path pins host memory via a cached PIN_MEMORY constant;
    # force it off to match NeuronPlatform.is_pin_memory_available()
    # (privateuse1 has no pinned-memory hooks in CPU mode).
    from vllm_neuron.vllm.patches.pin_memory_patch import apply_pin_memory_patch

    apply_pin_memory_patch()
