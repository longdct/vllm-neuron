# SPDX-License-Identifier: Apache-2.0
"""Guards that turn silent vLLM API drift into loud startup failures.

Every monkey-patch in this package targets an upstream symbol that vLLM is free
to rename, move, or re-shape between releases. When that happens the failure
mode is almost never an exception -- the patch simply stops matching and the
plugin runs upstream behaviour on Neuron. Past examples: a renamed scheduler
class path means the GPU scheduler is used unpatched; a reworked log message
means the registry-overwrite filter stops suppressing; a changed pydantic model
nesting means the ``neuron`` all2all backend is never registered.

The helpers here make each patch state its assumption up front and fail at
startup with an actionable message. The prototype is the existing tripwire in
``vllm_neuron/vllm/core/scheduler.py`` (``_install_swa_di_eviction_fix``), which
asserts that ``allocate_slots`` still accepts ``num_external_computed_tokens``.

Usage::

    from vllm_neuron.vllm.patches.guards import require_params, PatchError

    require_params(
        KVCacheManager.allocate_slots,
        "num_external_computed_tokens",
        patch="swa_di_eviction_fix",
    )
"""

import inspect
import re
from typing import Any

# Deliberately free of vllm/torch imports: the schema and signature helpers below
# are pure functions, and keeping this module importable on a bare interpreter
# means they can be unit-tested without a Neuron or CUDA stack installed.

# Bumped by hand when the plugin is validated against a new vLLM release, so the
# error text can name the version the guards were written against.
#
# This is a claim, not a formality: it may only move once every tripwire in
# ``tripwires.py`` executes clean against an installed tree of that version *and*
# ``test/vllm_neuron/test_upstream_compat.py`` passes against it. Both were run
# against vllm 0.26.0 before this was set. ``test_validated_version_matches_installed``
# keeps the claim honest by failing when the installed version drifts past it.
VALIDATED_VLLM_VERSION = "0.26.0"


class PatchError(RuntimeError):
    """Raised when an upstream vLLM symbol no longer matches a patch's needs."""


def _fail(patch: str, detail: str) -> None:
    raise PatchError(
        f"vllm-neuron patch {patch!r} no longer matches upstream vLLM: {detail}. "
        f"These patches were validated against vLLM {VALIDATED_VLLM_VERSION}; "
        f"re-validate the patch against the installed version."
    )


def require_attr(owner: Any, name: str, *, patch: str) -> Any:
    """Return ``owner.name``, failing loudly if it is gone.

    Use before replacing a symbol, so a rename is reported at startup instead of
    silently installing a patch onto an object nothing reads any more.
    """
    if not hasattr(owner, name):
        owner_name = getattr(owner, "__name__", repr(owner))
        _fail(patch, f"{owner_name} has no attribute {name!r}")
    return getattr(owner, name)


def require_params(func: Any, *names: str, patch: str) -> None:
    """Assert every parameter in *names* is accepted by *func*.

    Guards patches that pass or intercept a specific keyword. Extra upstream
    parameters are fine -- wrappers should forward ``*args, **kwargs`` -- so this
    only checks for removal, not addition.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError) as exc:  # builtins / C functions
        _fail(patch, f"cannot introspect {func!r}: {exc}")
    missing = [n for n in names if n not in params]
    if missing:
        _fail(
            patch,
            f"{getattr(func, '__qualname__', func)!r} no longer accepts "
            f"{', '.join(repr(m) for m in missing)}",
        )


def require_same_object(first: Any, second: Any, *, what: str, patch: str) -> None:
    """Assert two bindings are still the *same* object.

    Guards patches that must replace a symbol in more than one module because
    upstream re-exports it. If the re-export is replaced by a wrapper, patching
    only one binding leaves the other live and the patch half-applies.
    """
    if first is not second:
        _fail(patch, f"{what} are no longer the same object ({first!r} vs {second!r})")


def require_dataclass_field_default(
    cls: Any, field_name: str, expected: Any, *, patch: str
) -> None:
    """Assert a dataclass field still defaults to *expected*.

    For patches whose logic keys on a specific default -- e.g. treating
    ``scheduler_cls is None`` as "the user did not choose a scheduler". If
    upstream changes the default, that test silently stops identifying the
    default case.
    """
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        _fail(patch, f"{getattr(cls, '__name__', cls)!r} is not a dataclass")
    for f in dataclasses.fields(cls):
        if f.name != field_name:
            continue
        default = f.default
        if default is dataclasses.MISSING and f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        if default != expected:
            _fail(
                patch,
                f"{getattr(cls, '__name__', cls)}.{field_name} now defaults to "
                f"{default!r}, expected {expected!r}",
            )
        return
    _fail(patch, f"{getattr(cls, '__name__', cls)!r} has no field {field_name!r}")


def require_importable(path: str, *, patch: str) -> Any:
    """Import and return the object at a dotted ``module.attr`` path.

    For patches keyed on a *string* class path rather than an imported symbol --
    those degrade silently, since a stale string simply never compares equal.
    """
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        _fail(patch, f"{path!r} is not a dotted path")
    try:
        module = __import__(module_path, fromlist=[attr])
    except ImportError as exc:
        _fail(patch, f"cannot import {module_path!r} ({exc})")
    if not hasattr(module, attr):
        _fail(patch, f"{module_path!r} has no attribute {attr!r}")
    return getattr(module, attr)


def require_pattern_matches(
    pattern: re.Pattern[str], sample: str, *, patch: str
) -> None:
    """Assert a regex still matches a representative upstream message.

    Log-message filters are the most invisible patches in the package: when
    upstream rewords a message the filter keeps running and matches nothing.
    Pass a sample string copied from the upstream source.
    """
    if not pattern.search(sample):
        _fail(
            patch,
            f"pattern {pattern.pattern!r} no longer matches the upstream "
            f"message {sample!r}",
        )


def literal_rejection_locs(exc: Any, field_name: str) -> bool:
    """Whether *exc* is a pydantic error rejecting *field_name* as a bad literal.

    Used to tell "the Literal patch did not take effect" apart from "an unrelated
    required field was missing" when a registration verifies itself by trying a
    value it just added. Accepts anything exposing pydantic's ``errors()``.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return False
    try:
        entries = errors()
    except Exception:
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "literal_error":
            continue
        if any(str(part) == field_name for part in entry.get("loc", ())):
            return True
    return False


def _descend_to_literal(node: Any) -> dict | None:
    """Walk ``schema`` links down to the ``literal`` node, if there is one."""
    seen: set[int] = set()
    while isinstance(node, dict) and id(node) not in seen:
        seen.add(id(node))
        if node.get("type") == "literal" and isinstance(node.get("expected"), list):
            return node
        node = node.get("schema")
    return None


def find_literal_field_schema(core_schema: Any, field_name: str) -> dict | None:
    """Find the ``literal`` sub-schema for *field_name* in a pydantic schema.

    Replaces fixed-depth indexing such as
    ``schema["schema"]["schema"]["schema"]["fields"]``, which either raises or --
    worse -- silently selects the wrong node whenever pydantic changes how it
    nests model schemas. Searches by field name instead of by position, so it
    survives re-nesting. Returns ``None`` if the field is absent, letting the
    caller decide whether that is fatal.
    """
    stack: list[Any] = [core_schema]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, dict):
            fields = node.get("fields")
            # pydantic-core has used both a name->schema mapping and a list of
            # {"name": ...} entries for model fields; accept either.
            if isinstance(fields, dict) and field_name in fields:
                found = _descend_to_literal(fields[field_name])
                if found is not None:
                    return found
            elif isinstance(fields, list):
                for entry in fields:
                    if isinstance(entry, dict) and entry.get("name") == field_name:
                        found = _descend_to_literal(entry)
                        if found is not None:
                            return found
            stack.extend(v for v in node.values() if isinstance(v, dict | list))
        elif isinstance(node, list):
            stack.extend(v for v in node if isinstance(v, dict | list))
    return None
