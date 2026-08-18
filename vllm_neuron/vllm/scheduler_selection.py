# SPDX-License-Identifier: Apache-2.0
"""Which scheduler class the Neuron platform installs, and when.

Kept free of vllm/torch imports so the decision is unit-testable on a bare
interpreter. Whether these paths are still what upstream actually resolves to is
a separate question, answered by the compatibility tests in
``test/vllm_neuron/test_upstream_compat.py``.
"""

#: Upstream scheduler class paths treated as "the default, safe to replace".
#:
#: vLLM's own default is ``SchedulerConfig.scheduler_cls = None``, resolved at use
#: time by ``SchedulerConfig.get_scheduler_cls()`` to ``Scheduler`` or
#: ``AsyncScheduler``. These strings therefore only appear when a caller passes
#: one explicitly -- which must still count as "unset", or the plugin would
#: refuse to install the Neuron scheduler for anyone who spelled out the default.
UPSTREAM_DEFAULT_SCHEDULER_PATHS = (
    "vllm.v1.core.sched.scheduler.Scheduler",
    "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
)

NEURON_SCHEDULER_CLS = "vllm_neuron.vllm.core.scheduler.NeuronScheduler"
NEURON_ASYNC_SCHEDULER_CLS = "vllm_neuron.vllm.core.scheduler.NeuronAsyncScheduler"


def resolve_neuron_scheduler_cls(
    current_scheduler_cls: object, async_scheduling: bool
) -> str | None:
    """The Neuron scheduler path to install, or ``None`` to leave *current* alone.

    A genuinely custom scheduler is preserved: someone who supplied their own
    class did not ask for the Neuron one. Anything else -- unset, or explicitly
    one of the upstream defaults -- is replaced.
    """
    if (
        current_scheduler_cls is not None
        and current_scheduler_cls not in UPSTREAM_DEFAULT_SCHEDULER_PATHS
    ):
        return None
    return NEURON_ASYNC_SCHEDULER_CLS if async_scheduling else NEURON_SCHEDULER_CLS
