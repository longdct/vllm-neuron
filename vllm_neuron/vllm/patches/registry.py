# SPDX-License-Identifier: Apache-2.0
"""Lifecycle-phased registry for vllm-neuron's upstream patches.

Patches cannot all be applied from one hook: several must run at a lifecycle
phase that a single entry point cannot reach (see the package docstring). So
each is registered against an explicit :class:`Phase` and applied by whichever
component owns that phase.

Kept free of vllm/torch imports -- like ``guards`` -- so the registry semantics
(apply-once, phase isolation, duplicate rejection, error wrapping) are testable
on a bare interpreter.
"""

from collections.abc import Callable
from enum import Enum

from vllm_neuron.vllm.patches.guards import PatchError


class Phase(str, Enum):
    """When a patch must be applied, relative to engine startup."""

    #: Process/module import, before any vLLM config exists. Survives the
    #: spawn-mode re-import in worker subprocesses.
    IMPORT = "import"
    #: ``NeuronPlatform.check_and_update_config`` -- config objects exist and are
    #: still mutable.
    PLATFORM_CONFIG = "platform_config"
    #: Around ``init_distributed_environment`` / process-group creation.
    DISTRIBUTED_INIT = "distributed_init"
    #: ``NeuronModelRunner`` construction, after the KV cache config is known.
    MODEL_RUNNER_INIT = "model_runner_init"
    #: ``NeuronWorker`` startup and shutdown paths.
    WORKER_STARTUP = "worker_startup"


_REGISTRY: dict[Phase, list[tuple[str, Callable[[], None]]]] = {}
_APPLIED: set[tuple[Phase, str]] = set()
#: Patches applied outside a phase (parameterized installs that guard
#: themselves), recorded so ``applied_patches`` reports the whole picture.
_APPLIED_UNPHASED: set[str] = set()


def _registered_names() -> set[str]:
    return {name for entries in _REGISTRY.values() for name, _ in entries}


def register(phase: Phase, name: str) -> Callable[[Callable[[], None]], Callable]:
    """Register a zero-arg patch/guard to run at *phase*.

    ``name`` must be unique across every phase, not merely within one: it is the
    identity used in logs and in the applied-once bookkeeping, and a name reused
    across phases makes those reports ambiguous.
    """

    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        if name in _registered_names():
            raise ValueError(f"duplicate patch registration: {name!r}")
        _REGISTRY.setdefault(phase, []).append((name, fn))
        return fn

    return decorator


def apply_phase(phase: Phase, *, logger: object | None = None) -> None:
    """Apply every patch registered for *phase*, at most once per process.

    A :class:`PatchError` propagates: a patch that no longer matches upstream is
    a correctness problem, not something to warn about and continue past. Other
    exceptions are wrapped so the failing patch is named.

    A patch that raises is *not* marked applied, so a caller that recovers and
    retries the phase will attempt it again rather than silently skip it.
    """
    for name, fn in _REGISTRY.get(phase, []):
        key = (phase, name)
        if key in _APPLIED:
            continue
        try:
            fn()
        except PatchError:
            raise
        except Exception as exc:
            raise PatchError(f"patch {name!r} failed to apply: {exc}") from exc
        _APPLIED.add(key)
        if logger is not None:
            logger.debug(  # type: ignore[attr-defined]
                "Applied vllm-neuron patch %r (phase=%s)", name, phase.value
            )


def mark_applied(name: str) -> None:
    """Record a parameterized patch that guards and installs itself.

    Some patches cannot be zero-arg registry entries because they need runtime
    values (for example the node-topology patch needs ``ranks_per_node``). They
    still report themselves here so startup assertions can see them.
    """
    _APPLIED_UNPHASED.add(name)


def applied_patches() -> frozenset[str]:
    """Names applied so far -- for startup assertions and tests."""
    return frozenset({name for _, name in _APPLIED} | _APPLIED_UNPHASED)


def registered_patches(phase: Phase | None = None) -> tuple[str, ...]:
    """Registered names, for the whole process or a single phase."""
    if phase is None:
        return tuple(sorted(_registered_names()))
    return tuple(name for name, _ in _REGISTRY.get(phase, []))


def _reset_for_tests() -> None:
    """Clear registry and applied state. Test-support only."""
    _REGISTRY.clear()
    _APPLIED.clear()
    _APPLIED_UNPHASED.clear()
