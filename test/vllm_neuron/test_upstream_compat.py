# SPDX-License-Identifier: Apache-2.0
"""Compatibility tests against the *installed* vLLM.

The startup tripwires in ``vllm_neuron/vllm/patches/tripwires.py`` must be cheap
and must not construct engine objects, so several of them can only prove a
necessary-but-insufficient condition. The tests here prove the rest, by
inspecting real upstream source and by exercising real config objects.

They are the tests that must pass before the vLLM pin moves (Workstream A2/A4).
Run them against 0.21 first to establish the baseline, then against 0.26.

All of these skip cleanly when vLLM is not installed, so the CPU-only unit suite
stays runnable on a bare interpreter.
"""

import ast
import inspect

import pytest

vllm = pytest.importorskip("vllm", reason="compatibility tests require vLLM")


def _dotted(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


class TestSchedulerDefaults:
    """The Neuron scheduler must actually replace whatever upstream defaults to.

    The startup tripwire only proves the two documented paths are importable and
    that ``scheduler_cls`` still defaults to ``None``. Upstream could keep those
    aliases importable while changing what ``get_scheduler_cls()`` returns, and
    the plugin would silently preserve the upstream scheduler on Neuron.
    """

    @pytest.mark.parametrize("async_scheduling", [False, True])
    def test_upstream_defaults_are_the_paths_we_treat_as_default(
        self, async_scheduling
    ):
        from vllm.config.scheduler import SchedulerConfig

        from vllm_neuron.vllm.scheduler_selection import (
            UPSTREAM_DEFAULT_SCHEDULER_PATHS,
        )

        config = SchedulerConfig(
            scheduler_cls=None, async_scheduling=async_scheduling
        )
        resolved = _dotted(config.get_scheduler_cls())
        assert resolved in UPSTREAM_DEFAULT_SCHEDULER_PATHS, (
            f"vLLM now resolves its default scheduler to {resolved}, which the "
            f"Neuron override does not recognise as a default; it would refuse "
            f"to install the Neuron scheduler"
        )

    def test_sync_and_async_defaults_are_distinct(self):
        from vllm.config.scheduler import SchedulerConfig

        sync = _dotted(SchedulerConfig(async_scheduling=False).get_scheduler_cls())
        asyn = _dotted(SchedulerConfig(async_scheduling=True).get_scheduler_cls())
        assert sync != asyn

    @pytest.mark.parametrize("async_scheduling", [False, True])
    def test_neuron_scheduler_replaces_actual_upstream_defaults(
        self, async_scheduling
    ):
        """Feed the resolver exactly what upstream resolves to, not a constant."""
        from vllm.config.scheduler import SchedulerConfig

        from vllm_neuron.vllm.scheduler_selection import (
            NEURON_ASYNC_SCHEDULER_CLS,
            NEURON_SCHEDULER_CLS,
            resolve_neuron_scheduler_cls,
        )

        upstream_default = _dotted(
            SchedulerConfig(async_scheduling=async_scheduling).get_scheduler_cls()
        )
        expected = (
            NEURON_ASYNC_SCHEDULER_CLS if async_scheduling else NEURON_SCHEDULER_CLS
        )
        # Both the unset case and the explicitly-spelled-out default case.
        assert resolve_neuron_scheduler_cls(None, async_scheduling) == expected
        assert resolve_neuron_scheduler_cls(upstream_default, async_scheduling) == (
            expected
        )

    def test_a_genuinely_custom_scheduler_is_preserved(self):
        from vllm_neuron.vllm.scheduler_selection import resolve_neuron_scheduler_cls

        assert resolve_neuron_scheduler_cls("my.pkg.MyScheduler", False) is None


class TestAll2AllBackendRegistration:
    """The 'neuron' all2all backend registration must actually take effect.

    Editing ``__pydantic_core_schema__`` and rebuilding the validator is not
    guaranteed to change validation -- on pydantic 2.13 a rebuilt
    ``SchemaValidator`` was observed still rejecting a value added this way. If
    that reproduces against the real ``ParallelConfig``, expert parallelism on
    Neuron is broken regardless of the vLLM version.
    """

    def test_all2all_backend_registration_takes_effect(self):
        from vllm.config import ParallelConfig

        from vllm_neuron.vllm.platform import NeuronPlatform

        NeuronPlatform._register_neuron_all2all_backend()

        config = ParallelConfig(all2all_backend="neuron")
        assert config.all2all_backend == "neuron"

    def test_registration_is_idempotent(self):
        from vllm.config import ParallelConfig

        from vllm_neuron.vllm.platform import NeuronPlatform

        NeuronPlatform._register_neuron_all2all_backend()
        NeuronPlatform._register_neuron_all2all_backend()

        expected = ParallelConfig.__pydantic_core_schema__
        from vllm_neuron.vllm.patches.guards import find_literal_field_schema

        literal = find_literal_field_schema(expected, "all2all_backend")
        assert literal is not None
        assert literal["expected"].count("neuron") == 1

    def test_still_rejects_an_unknown_backend(self):
        """The patch must widen the Literal, not disable validation."""
        import pydantic
        from vllm.config import ParallelConfig

        from vllm_neuron.vllm.platform import NeuronPlatform

        NeuronPlatform._register_neuron_all2all_backend()

        with pytest.raises(pydantic.ValidationError):
            ParallelConfig(all2all_backend="definitely_not_a_backend")


class TestModelRegistryOverwriteMessage:
    """The overwrite-warning filter must still match upstream's real wording.

    Checking a locally-defined regex against a locally-defined sample proves only
    internal consistency -- both can sit unchanged while upstream rewrites its
    message. So read the format string out of the installed upstream source.

    This filter is cosmetic, which is why it is a compatibility-test failure
    rather than a startup failure: a stale filter means noisy logs, not wrong
    output.
    """

    @staticmethod
    def _upstream_format_strings() -> list[str]:
        """String constants in upstream's registry mentioning re-registration.

        Implicit concatenation is folded by the parser, so a message split across
        source lines arrives here as one constant.
        """
        import vllm.model_executor.models.registry as upstream

        tree = ast.parse(inspect.getsource(upstream))
        return [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "already registered" in node.value
        ]

    def test_upstream_still_emits_an_already_registered_message(self):
        found = self._upstream_format_strings()
        assert found, (
            "no 'already registered' message found in "
            "vllm.model_executor.models.registry; the overwrite-suppression "
            "filter is now dead code"
        )

    def test_filter_pattern_matches_the_real_upstream_message(self):
        from vllm_neuron.vllm.worker.neuron_worker import (
            _SuppressModelRegistryOverwrite,
        )

        pattern = _SuppressModelRegistryOverwrite._PATTERN
        rendered = [
            self._render(fmt) for fmt in self._upstream_format_strings()
        ]
        assert any(pattern.search(text) for text in rendered), (
            f"filter pattern {pattern.pattern!r} matches none of the upstream "
            f"messages {rendered!r}"
        )

    @staticmethod
    def _render(fmt: str) -> str:
        """Fill %s placeholders with representative registration arguments."""
        placeholders = fmt.count("%s")
        args = ("LlamaForCausalLM", "<class 'vllm_neuron.model.llama3.X'>")
        return fmt % args[:placeholders] if placeholders else fmt

    def test_filter_suppresses_a_matching_record(self):
        import logging

        from vllm_neuron.vllm.worker.neuron_worker import (
            _SuppressModelRegistryOverwrite,
        )

        fmts = self._upstream_format_strings()
        record = logging.LogRecord(
            name="vllm.model_executor.models.registry",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=fmts[0],
            args=("LlamaForCausalLM", "<class 'X'>")[: fmts[0].count("%s")],
            exc_info=None,
        )
        assert _SuppressModelRegistryOverwrite().filter(record) is False

    def test_filter_lets_unrelated_records_through(self):
        import logging

        from vllm_neuron.vllm.worker.neuron_worker import (
            _SuppressModelRegistryOverwrite,
        )

        record = logging.LogRecord(
            name="vllm.model_executor.models.registry",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="Something else entirely went wrong",
            args=(),
            exc_info=None,
        )
        assert _SuppressModelRegistryOverwrite().filter(record) is True


class TestTripwiresRunClean:
    """Every platform-config tripwire must pass against the installed vLLM."""

    def test_platform_config_phase_applies(self):
        from vllm_neuron.vllm.patches import Phase, applied_patches, apply_phase

        apply_phase(Phase.PLATFORM_CONFIG)
        applied = applied_patches()
        assert "tripwire:scheduler_default_detection" in applied
        assert "tripwire:parallel_config_all2all_literal" in applied
        assert "tripwire:termination_timeout_targets" in applied

    def test_distributed_init_phase_applies(self):
        from vllm_neuron.vllm.patches import Phase, applied_patches, apply_phase

        apply_phase(Phase.DISTRIBUTED_INIT)
        assert "tripwire:in_the_same_node_as" in applied_patches()


class TestNodeTopologyPatch:
    def test_install_replaces_the_upstream_symbol(self):
        import vllm.distributed.parallel_state as parallel_state

        from vllm_neuron.vllm.patches.node_topology import (
            install_in_the_same_node_as,
        )

        original = parallel_state.in_the_same_node_as
        try:
            install_in_the_same_node_as(8)
            assert parallel_state.in_the_same_node_as is not original
            assert parallel_state.in_the_same_node_as._neuron_node_topology_patched
        finally:
            parallel_state.in_the_same_node_as = original

    def test_install_is_idempotent(self):
        import vllm.distributed.parallel_state as parallel_state

        from vllm_neuron.vllm.patches.node_topology import (
            install_in_the_same_node_as,
        )

        original = parallel_state.in_the_same_node_as
        try:
            install_in_the_same_node_as(8)
            first = parallel_state.in_the_same_node_as
            install_in_the_same_node_as(8)
            assert parallel_state.in_the_same_node_as is first
        finally:
            parallel_state.in_the_same_node_as = original
