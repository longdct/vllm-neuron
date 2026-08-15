# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the patch guards.

``vllm_neuron.vllm.patches.guards`` is deliberately free of vllm/torch imports so
these run on a bare interpreter. The ``guards`` fixture in ``conftest.py`` loads
it from its file path, because importing the ``patches`` package would pull in
``vllm.logger``.
"""

import re

import pytest


def _literal(*expected):
    return {"type": "literal", "expected": list(expected)}


def _dataclass_args_schema(fields):
    """Approximate pydantic-core's dataclass shape: fields as a list of entries."""
    return {
        "type": "dataclass",
        "schema": {
            "type": "dataclass-args",
            "fields": [
                {"type": "dataclass-field", "name": name, "schema": schema}
                for name, schema in fields
            ],
        },
    }


class TestFindLiteralFieldSchema:
    def test_finds_field_in_dataclass_args_list(self, guards):
        schema = _dataclass_args_schema(
            [
                ("tensor_parallel_size", {"type": "int"}),
                ("all2all_backend", _literal("naive", "pplx")),
            ]
        )
        found = guards.find_literal_field_schema(schema, "all2all_backend")
        assert found is not None
        assert found["expected"] == ["naive", "pplx"]

    def test_finds_field_in_model_fields_mapping(self, guards):
        """pydantic has also represented fields as a name -> schema mapping."""
        schema = {
            "type": "model",
            "schema": {
                "type": "model-fields",
                "fields": {
                    "all2all_backend": {
                        "type": "model-field",
                        "schema": _literal("naive"),
                    }
                },
            },
        }
        found = guards.find_literal_field_schema(schema, "all2all_backend")
        assert found is not None
        assert found["expected"] == ["naive"]

    def test_survives_extra_nesting(self, guards):
        """The point of searching by name: fixed-depth indexing breaks here."""
        inner = _dataclass_args_schema([("all2all_backend", _literal("naive"))])
        wrapped = {"type": "definitions", "schema": {"type": "nullable", "schema": inner}}
        assert guards.find_literal_field_schema(wrapped, "all2all_backend") is not None

    def test_descends_through_intermediate_schema_links(self, guards):
        schema = _dataclass_args_schema(
            [
                (
                    "all2all_backend",
                    {"type": "default", "schema": _literal("naive", "deepep")},
                )
            ]
        )
        found = guards.find_literal_field_schema(schema, "all2all_backend")
        assert found is not None
        assert found["expected"] == ["naive", "deepep"]

    def test_mutating_the_result_edits_the_original(self, guards):
        """The caller registers 'neuron' by mutating the returned node in place."""
        schema = _dataclass_args_schema([("all2all_backend", _literal("naive"))])
        found = guards.find_literal_field_schema(schema, "all2all_backend")
        found["expected"] = [*found["expected"], "neuron"]
        field = schema["schema"]["fields"][0]
        assert field["schema"]["expected"] == ["naive", "neuron"]

    def test_missing_field_returns_none(self, guards):
        schema = _dataclass_args_schema([("tensor_parallel_size", {"type": "int"})])
        assert guards.find_literal_field_schema(schema, "all2all_backend") is None

    def test_non_literal_field_returns_none(self, guards):
        schema = _dataclass_args_schema([("all2all_backend", {"type": "str"})])
        assert guards.find_literal_field_schema(schema, "all2all_backend") is None

    def test_terminates_on_self_referential_schema(self, guards):
        """Real pydantic schemas contain definition-refs and can cycle."""
        schema = _dataclass_args_schema([("all2all_backend", _literal("naive"))])
        schema["self"] = schema
        assert guards.find_literal_field_schema(schema, "all2all_backend") is not None
        assert guards.find_literal_field_schema(schema, "nope") is None


class TestRequireParams:
    def test_accepts_present_params(self, guards):
        def fn(request, num_new_tokens, *, num_external_computed_tokens=0):
            ...

        guards.require_params(fn, "num_external_computed_tokens", patch="p")

    def test_rejects_removed_param(self, guards):
        def fn(request, num_new_tokens):
            ...

        with pytest.raises(guards.PatchError, match="num_external_computed_tokens"):
            guards.require_params(fn, "num_external_computed_tokens", patch="p")

    def test_tolerates_added_upstream_params(self, guards):
        """0.26 added reserved_blocks/has_scheduled_reqs; wrappers forward **kwargs."""

        def fn(request, num_new_tokens, num_external_computed_tokens=0, *, added=1):
            ...

        guards.require_params(fn, "num_external_computed_tokens", patch="p")


class TestRequireAttr:
    def test_returns_attribute(self, guards):
        assert guards.require_attr(re, "compile", patch="p") is re.compile

    def test_raises_on_missing(self, guards):
        with pytest.raises(guards.PatchError, match="gone_upstream"):
            guards.require_attr(re, "gone_upstream", patch="p")


class TestRequireImportable:
    def test_imports_dotted_path(self, guards):
        assert guards.require_importable("re.Pattern", patch="p") is re.Pattern

    def test_raises_on_missing_attr(self, guards):
        with pytest.raises(guards.PatchError, match="no attribute"):
            guards.require_importable("re.NotAThing", patch="p")

    def test_raises_on_missing_module(self, guards):
        with pytest.raises(guards.PatchError, match="cannot import"):
            guards.require_importable("vllm_neuron_nope.Thing", patch="p")

    def test_raises_on_bare_name(self, guards):
        with pytest.raises(guards.PatchError, match="not a dotted path"):
            guards.require_importable("re", patch="p")


class TestRequirePatternMatches:
    #: The rendered form of upstream's format string, identical in 0.21 and 0.26.
    UPSTREAM_SAMPLE = (
        "Model architecture LlamaForCausalLM is already registered, and will be "
        "overwritten by the new model class <class 'x.Y'>."
    )

    def test_current_filter_pattern_matches_upstream_message(self, guards):
        pattern = re.compile(
            r"Model architecture \w+ is already registered.*will be overwritten"
        )
        guards.require_pattern_matches(pattern, self.UPSTREAM_SAMPLE, patch="p")

    def test_raises_when_upstream_rewords(self, guards):
        pattern = re.compile(r"will be clobbered")
        with pytest.raises(guards.PatchError, match="no longer matches"):
            guards.require_pattern_matches(pattern, self.UPSTREAM_SAMPLE, patch="p")


class TestRequireSameObject:
    """Guards symbols that upstream re-exports and the patch replaces twice."""

    def test_accepts_identical_bindings(self, guards):
        fn = re.compile
        guards.require_same_object(fn, fn, what="two bindings", patch="p")

    def test_rejects_a_wrapped_re_export(self, guards):
        def original():
            ...

        def wrapper():
            return original()

        with pytest.raises(guards.PatchError, match="no longer the same object"):
            guards.require_same_object(
                original, wrapper, what="two bindings", patch="p"
            )


class TestRequireDataclassFieldDefault:
    def test_accepts_expected_default(self, guards):
        import dataclasses

        @dataclasses.dataclass
        class Cfg:
            scheduler_cls: str | None = None

        guards.require_dataclass_field_default(Cfg, "scheduler_cls", None, patch="p")

    def test_rejects_changed_default(self, guards):
        """Upstream switching the default away from None breaks 'is unset' logic."""
        import dataclasses

        @dataclasses.dataclass
        class Cfg:
            scheduler_cls: str | None = "vllm.v1.core.sched.scheduler.Scheduler"

        with pytest.raises(guards.PatchError, match="now defaults to"):
            guards.require_dataclass_field_default(
                Cfg, "scheduler_cls", None, patch="p"
            )

    def test_resolves_default_factory(self, guards):
        import dataclasses

        @dataclasses.dataclass
        class Cfg:
            items: list = dataclasses.field(default_factory=list)

        guards.require_dataclass_field_default(Cfg, "items", [], patch="p")

    def test_rejects_missing_field(self, guards):
        import dataclasses

        @dataclasses.dataclass
        class Cfg:
            other: int = 1

        with pytest.raises(guards.PatchError, match="has no field"):
            guards.require_dataclass_field_default(Cfg, "scheduler_cls", None, patch="p")

    def test_rejects_non_dataclass(self, guards):
        with pytest.raises(guards.PatchError, match="not a dataclass"):
            guards.require_dataclass_field_default(object, "x", None, patch="p")


class TestLiteralRejectionLocs:
    """Tells 'the Literal patch did not take' apart from an unrelated field error."""

    class _Exc:
        def __init__(self, entries):
            self._entries = entries

        def errors(self):
            return self._entries

    def test_detects_rejection_of_the_named_field(self, guards):
        exc = self._Exc(
            [{"type": "literal_error", "loc": ("all2all_backend",), "msg": "x"}]
        )
        assert guards.literal_rejection_locs(exc, "all2all_backend") is True

    def test_ignores_literal_error_on_a_different_field(self, guards):
        exc = self._Exc([{"type": "literal_error", "loc": ("distributed_backend",)}])
        assert guards.literal_rejection_locs(exc, "all2all_backend") is False

    def test_ignores_non_literal_error_on_the_same_field(self, guards):
        """A missing/typed error means something else objected, not our patch."""
        exc = self._Exc([{"type": "missing", "loc": ("all2all_backend",)}])
        assert guards.literal_rejection_locs(exc, "all2all_backend") is False

    def test_handles_exception_without_errors_api(self, guards):
        assert guards.literal_rejection_locs(ValueError("nope"), "f") is False

    def test_handles_errors_that_raise(self, guards):
        class Boom:
            def errors(self):
                raise RuntimeError("boom")

        assert guards.literal_rejection_locs(Boom(), "f") is False

    def test_against_a_real_pydantic_validation_error(self, guards):
        pydantic = pytest.importorskip("pydantic")
        from typing import Literal

        @pydantic.dataclasses.dataclass
        class Cfg:
            all2all_backend: Literal["naive"] = "naive"

        with pytest.raises(pydantic.ValidationError) as caught:
            Cfg(all2all_backend="neuron")
        assert guards.literal_rejection_locs(caught.value, "all2all_backend") is True
        assert guards.literal_rejection_locs(caught.value, "other_field") is False


def test_patch_error_names_the_validated_version(guards):
    with pytest.raises(guards.PatchError, match=guards.VALIDATED_VLLM_VERSION):
        guards.require_attr(re, "gone_upstream", patch="p")
