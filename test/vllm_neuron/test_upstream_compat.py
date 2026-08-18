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
import textwrap

import pytest

vllm = pytest.importorskip("vllm", reason="compatibility tests require vLLM")


def _dotted(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _scheduler_config(**kwargs):
    """Build a ``SchedulerConfig``, supplying fields upstream requires but we don't care about.

    vLLM 0.26 made ``SchedulerConfig`` a pydantic model with ``max_model_len``
    and ``is_encoder_decoder`` as required fields; on 0.21 both were optional.
    Neither influences ``get_scheduler_cls()``, so they are pinned to inert
    values here rather than parametrized -- these tests are about which
    scheduler class upstream resolves to, nothing else.

    Deliberately not defensive about *further* required fields appearing: if a
    later release adds one, this raises and the pin-move review has to look at
    it, which is the point of the file.
    """
    from vllm.config.scheduler import SchedulerConfig

    return SchedulerConfig(max_model_len=128, is_encoder_decoder=False, **kwargs)


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
        from vllm_neuron.vllm.scheduler_selection import (
            UPSTREAM_DEFAULT_SCHEDULER_PATHS,
        )

        config = _scheduler_config(
            scheduler_cls=None, async_scheduling=async_scheduling
        )
        resolved = _dotted(config.get_scheduler_cls())
        assert resolved in UPSTREAM_DEFAULT_SCHEDULER_PATHS, (
            f"vLLM now resolves its default scheduler to {resolved}, which the "
            f"Neuron override does not recognise as a default; it would refuse "
            f"to install the Neuron scheduler"
        )

    def test_sync_and_async_defaults_are_distinct(self):
        sync = _dotted(_scheduler_config(async_scheduling=False).get_scheduler_cls())
        asyn = _dotted(_scheduler_config(async_scheduling=True).get_scheduler_cls())
        assert sync != asyn

    @pytest.mark.parametrize("async_scheduling", [False, True])
    def test_neuron_scheduler_replaces_actual_upstream_defaults(
        self, async_scheduling
    ):
        """Feed the resolver exactly what upstream resolves to, not a constant."""
        from vllm_neuron.vllm.scheduler_selection import (
            NEURON_ASYNC_SCHEDULER_CLS,
            NEURON_SCHEDULER_CLS,
            resolve_neuron_scheduler_cls,
        )

        upstream_default = _dotted(
            _scheduler_config(async_scheduling=async_scheduling).get_scheduler_cls()
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


class TestAll2AllBackendSelection:
    """The 'neuron' all2all backend must be selectable, without patching vLLM.

    This class previously asserted that ``NeuronPlatform`` widened vLLM's
    ``ParallelConfig.all2all_backend`` Literal by editing
    ``__pydantic_core_schema__`` and rebuilding the validator -- fragile, and on
    pydantic 2.13 a rebuilt ``SchemaValidator`` was observed still rejecting the
    added value. The 0.24 plugin designs that away: the backend is a plugin-side
    ``NeuronConfig`` field read by ``NeuronCommunicator``, so vLLM's own schema is
    never touched. What is worth pinning is that the plugin-side path still works
    and still rejects nonsense.
    """

    def test_neuron_config_accepts_the_neuron_backend(self):
        from vllm_neuron.model.neuron_config import NeuronConfig

        assert NeuronConfig.from_dict({"all2all_backend": "neuron"}).all2all_backend == (
            "neuron"
        )

    def test_the_communicator_reads_it_back(self):
        from vllm_neuron.parallel.neuron_communicator import NeuronCommunicator

        assert hasattr(NeuronCommunicator, "_read_neuron_all2all_backend")

    def test_vllm_parallel_config_is_left_unpatched(self):
        """Scope guard: nothing may reintroduce the pydantic schema surgery.

        If a future port re-adds it, this fails and points at the 0.24 mechanism
        instead.
        """
        from vllm.config import ParallelConfig

        from vllm_neuron.vllm.platform import NeuronPlatform

        assert not hasattr(NeuronPlatform, "_register_neuron_all2all_backend")
        schema = ParallelConfig.__pydantic_core_schema__
        from vllm_neuron.vllm.patches.guards import find_literal_field_schema

        literal = find_literal_field_schema(schema, "all2all_backend")
        if literal is not None:
            assert "neuron" not in literal["expected"]


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


class TestValidatedVersionClaim:
    """``VALIDATED_VLLM_VERSION`` must describe the tree the tests actually ran against.

    The constant is what every ``PatchError`` message cites when it tells an
    operator which version the guards were written for. Left to drift it becomes
    an active lie -- the patches would claim validation against a release nobody
    ran them on. These two tests are the automated form of the manual gate:
    every tripwire executes clean, and the version named is the version here.
    """

    def test_every_registered_tripwire_passes_against_installed_vllm(self):
        """Run the real tripwire bodies, not just check they are registered.

        The phase-application tests above prove the registry *invokes* each
        entry; this proves each entry's assumption still holds against the
        installed upstream. That is the condition for bumping the constant, so
        it is asserted rather than left to a one-off manual run.
        """
        import vllm_neuron.vllm.patches.tripwires  # noqa: F401  (registers entries)
        from vllm_neuron.vllm.patches.registry import _REGISTRY

        failures = []
        checked = 0
        for phase, entries in _REGISTRY.items():
            for name, fn in entries:
                if not name.startswith("tripwire:"):
                    continue
                checked += 1
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
                    failures.append(f"{phase.value}/{name}: {type(exc).__name__}: {exc}")

        assert checked, "no tripwires were registered; the import above is not working"
        assert not failures, "tripwires failed against installed vLLM:\n" + "\n".join(
            failures
        )

    def test_validated_version_matches_installed(self):
        """Fail when vLLM moves past the version the guards were validated on.

        Deliberately exact rather than a floor: the whole point of the patch
        registry is that upstream drift is silent, so "newer than validated" is
        precisely the state that needs a human to re-run the gate and re-read
        the diff -- not something to wave through with a >= comparison.
        """
        from vllm_neuron.vllm.patches.guards import VALIDATED_VLLM_VERSION

        installed = vllm.__version__
        assert installed == VALIDATED_VLLM_VERSION, (
            f"vLLM {installed} is installed but the patches are validated against "
            f"{VALIDATED_VLLM_VERSION}. Re-run this file and the startup tripwires "
            f"against {installed}, review the upstream diff for the patched "
            f"symbols, then bump VALIDATED_VLLM_VERSION in patches/guards.py."
        )


def _instance_attrs(cls) -> set[str]:
    """Attribute names ``cls.__init__`` assigns to ``self``.

    Instance attributes are invisible to ``hasattr`` on the class, and building a
    real ``Request`` or ``CachedRequestState`` here would drag in sampling
    params, tokenizers and a model config -- far more coupling than a name check
    warrants. Reading the assignments out of the source is enough to catch the
    rename/removal cases these guards exist for.
    """
    # Dedent: a method's source carries its class indentation, which is a
    # syntax error on its own.
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


class TestPortedUpstreamSurfaces:
    """Field-level guards for the two line-for-line upstream ports.

    ``NeuronModelRunner._update_states`` and
    ``NeuronAsyncScheduler._update_after_schedule`` are copies of upstream
    bodies with Neuron deltas applied. Upstream moved roughly 1900/7900 lines in
    ``GPUModelRunner`` and 1050/2900 in ``Scheduler`` between 0.21 and 0.26, and
    a copied body drifts *silently*: it keeps running against a struct that has
    quietly lost or renamed a field only when that field is reached at runtime.

    These assert the surfaces each port reads. They are the automated form of
    "re-diff field by field" -- cheaper to run than the diff, and they fail on
    the next release rather than after the next incident.
    """

    def test_cached_request_data_fields_the_runner_reads(self):
        """Fields ``_update_states`` indexes per cached request."""
        import dataclasses

        from vllm.v1.core.sched.output import CachedRequestData

        present = {f.name for f in dataclasses.fields(CachedRequestData)}
        required = {
            "req_ids",
            "resumed_req_ids",
            "new_block_ids",
            "num_computed_tokens",
            "num_output_tokens",
        }
        assert required <= present, (
            f"CachedRequestData lost {sorted(required - present)}; "
            f"NeuronModelRunner._update_states reads these per request"
        )

    def test_cached_request_state_fields_the_runner_constructs(self):
        """Keywords the runner passes to ``CachedRequestState``.

        A *removed* field raises immediately at construction, so the real risk
        this covers is the opposite: upstream adding a field the runner must
        start populating. That cannot be asserted generically, so this pins the
        set the port was written against -- a diff here is the prompt to check
        whether a new field needs a Neuron value.
        """
        import dataclasses

        from vllm.v1.worker.gpu_input_batch import CachedRequestState

        present = {f.name for f in dataclasses.fields(CachedRequestState)}
        constructed = {
            "req_id",
            "prompt_token_ids",
            "prompt_embeds",
            "mm_features",
            "sampling_params",
            "pooling_params",
            "generator",
            "block_ids",
            "num_computed_tokens",
            "output_token_ids",
            "lora_request",
        }
        assert constructed <= present, (
            f"CachedRequestState lost {sorted(constructed - present)}, which "
            f"NeuronModelRunner._update_states passes by keyword"
        )
        # Read outside the constructor, on the spec-decode path.
        assert "prev_num_draft_len" in present

    def test_scheduler_output_fields_the_scheduler_port_reads(self):
        import dataclasses

        from vllm.v1.core.sched.output import SchedulerOutput

        present = {f.name for f in dataclasses.fields(SchedulerOutput)}
        required = {
            "num_scheduled_tokens",
            "scheduled_new_reqs",
            "scheduled_cached_reqs",
            "scheduled_spec_decode_tokens",
            "pending_structured_output_tokens",
        }
        assert required <= present, (
            f"SchedulerOutput lost {sorted(required - present)}; "
            f"NeuronAsyncScheduler._update_after_schedule reads these"
        )

    def test_request_attributes_the_scheduler_port_mutates(self):
        from vllm.v1.request import Request

        present = _instance_attrs(Request) | set(dir(Request))
        required = {
            "is_prefill_chunk",
            "num_output_placeholders",
            "use_structured_output",
            "spec_token_ids",
            "num_computed_tokens",
        }
        assert required <= present, (
            f"Request lost {sorted(required - present)}; "
            f"NeuronAsyncScheduler._update_after_schedule reads or mutates these"
        )

    def test_base_scheduler_hook_is_still_callable_unbound(self):
        """The port calls ``Scheduler._update_after_schedule(self, ...)`` directly.

        It deliberately reaches past ``AsyncScheduler``'s override to skip
        unconditional spec-placeholder injection, then re-applies the parts it
        wants. That only works while the base method exists with this signature
        *and* ``AsyncScheduler`` still overrides it -- if upstream removed the
        override, the bypass would silently become a no-op refactor and the
        Neuron gating would be applied on top of behaviour that already matched.
        """
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler

        params = inspect.signature(Scheduler._update_after_schedule).parameters
        assert list(params) == ["self", "scheduler_output"]
        assert "_update_after_schedule" in AsyncScheduler.__dict__, (
            "AsyncScheduler no longer overrides _update_after_schedule; the "
            "Neuron override's bypass of it is now meaningless and its spec "
            "placeholder gating must be re-derived"
        )


class TestDisaggregatedInferenceIsSupported:
    """DI works on vLLM 0.24, and the connectors must actually import.

    This class previously asserted the opposite. The vLLM 0.26 line split the
    monolithic NIXL connector and dropped ``NixlConnectorWorker``, so the plugin
    rejected DI loudly rather than failing deep inside a worker. vLLM 0.24 still
    exports ``NixlConnectorWorker`` -- it simply lives in ``...v1.nixl.worker``
    rather than ``...v1.nixl.connector`` -- so the rejection is gone and the
    import itself is the assertion worth keeping: it is what regresses if the
    module split lands in a future pin.
    """

    def test_the_nixl_connector_imports(self):
        import vllm_neuron.vllm.kv_connector.neuron_nixl_connector as connector  # noqa: F401

        assert connector.NeuronNixlConnector is not None

    def test_the_nixl_connector_worker_base_is_importable(self):
        """The symbol whose disappearance forced the 0.26 rejection."""
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
            NixlConnectorWorker,
        )

        assert NixlConnectorWorker is not None

    def test_the_decode_bench_connector_still_imports(self):
        """Scope guard: DI must not be blanket-disabled."""
        import vllm_neuron.vllm.kv_connector.neuron_decode_bench_connector  # noqa: F401
