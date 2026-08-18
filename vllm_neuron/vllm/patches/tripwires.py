# SPDX-License-Identifier: Apache-2.0
"""Startup assertions for patches whose failure mode is silence.

Each entry guards a patch that, when upstream drifts, keeps running and simply
stops having any effect -- leaving the plugin executing upstream vLLM behaviour
on Neuron. The patch bodies stay where they are; what lives here is the
assumption each one makes, checked early and loudly.

Scope note: these run at engine startup, so they must be cheap and must not
construct engine objects. Assertions that need to *build* config objects or
observe real log output live in ``test/vllm_neuron/test_upstream_compat.py``
instead, which runs against the pinned vLLM in CI. Where a startup guard can
only prove a necessary-but-insufficient condition, it says so and names the
compatibility test that proves the rest.
"""

from vllm_neuron.vllm.patches.guards import (
    require_attr,
    require_dataclass_field_default,
    require_importable,
    require_params,
    require_same_object,
)
from vllm_neuron.vllm.patches.registry import Phase, register
from vllm_neuron.vllm.scheduler_selection import UPSTREAM_DEFAULT_SCHEDULER_PATHS

@register(Phase.PLATFORM_CONFIG, "tripwire:scheduler_default_detection")
def _check_scheduler_default_detection() -> None:
    """The override must still be able to recognise "user chose no scheduler".

    Two conditions, both necessary:

    * ``SchedulerConfig.scheduler_cls`` still defaults to ``None`` -- the value
      ``check_and_update_config`` treats as "unset, safe to override". If
      upstream starts defaulting it to a concrete path, that test stops firing
      and Neuron silently runs the upstream scheduler.
    * The explicit paths callers may pass still resolve to real classes.

    Neither proves the resolved defaults are *these* classes; upstream could keep
    the aliases importable and change what ``get_scheduler_cls()`` returns. That
    is proved behaviourally by
    ``test_neuron_scheduler_replaces_actual_upstream_defaults``.
    """
    from vllm.config.scheduler import SchedulerConfig

    require_dataclass_field_default(
        SchedulerConfig, "scheduler_cls", None, patch="scheduler_cls_override"
    )
    for path in UPSTREAM_DEFAULT_SCHEDULER_PATHS:
        require_importable(path, patch="scheduler_cls_override")


@register(Phase.PLATFORM_CONFIG, "tripwire:termination_timeout_targets")
def _check_termination_timeout_targets() -> None:
    """Both hardcoded SIGTERM->SIGKILL paths must still be replaceable.

    Checks existence, calling convention, and the re-export relationship:

    * ``v1.utils.shutdown(procs, timeout=None)`` -- the replacement is called
      with both, so a changed convention would fail at shutdown, long after the
      patch installed cleanly.
    * ``v1.engine.utils.shutdown`` must still *be* the same object, because it is
      a ``from ... import`` re-export captured by ``weakref.finalize``. If
      upstream wraps it, replacing only one binding half-applies the patch.
    * ``MultiprocExecutor._ensure_worker_termination(worker_procs)``.
    """
    import vllm.v1.engine.utils as engine_utils
    import vllm.v1.utils as v1_utils
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    shutdown = require_attr(v1_utils, "shutdown", patch="termination_timeout")
    require_params(shutdown, "procs", "timeout", patch="termination_timeout")
    require_same_object(
        shutdown,
        require_attr(engine_utils, "shutdown", patch="termination_timeout"),
        what="vllm.v1.utils.shutdown and vllm.v1.engine.utils.shutdown",
        patch="termination_timeout",
    )

    ensure_termination = require_attr(
        MultiprocExecutor, "_ensure_worker_termination", patch="termination_timeout"
    )
    require_params(ensure_termination, "worker_procs", patch="termination_timeout")


@register(Phase.DISTRIBUTED_INIT, "tripwire:in_the_same_node_as")
def _check_in_the_same_node_as() -> None:
    """The symbol the node-topology patch replaces must still exist.

    Its replacement is installed by
    :func:`vllm_neuron.vllm.patches.node_topology.install_in_the_same_node_as`,
    which needs a runtime ``ranks_per_node`` and so cannot be a zero-arg entry.
    """
    import vllm.distributed.parallel_state as parallel_state

    fn = require_attr(parallel_state, "in_the_same_node_as", patch="node_topology")
    require_params(fn, "pg", patch="node_topology")
