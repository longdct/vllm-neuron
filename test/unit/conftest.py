# SPDX-License-Identifier: Apache-2.0
"""Loader for the vllm/torch-free modules of ``vllm_neuron``.

``guards``, ``registry``, ``scheduler_selection`` and the DeepSeek-V4 config
normalizer are deliberately dependency-free so their logic can be tested on a
bare interpreter -- which is the only tier available on a machine without Linux
or Neuron hardware. Importing them normally would not work: the
``vllm_neuron`` package ``__init__`` pulls in torch and Neuron device setup, and
``vllm_neuron.vllm.patches.__init__`` imports ``vllm.logger``.

So they are loaded straight from their file paths, with just enough of the
package tree stubbed into ``sys.modules`` that their absolute intra-package
imports (``from vllm_neuron.vllm.patches.guards import PatchError``) resolve to
the modules loaded here rather than triggering the real packages.
"""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHES_DIR = REPO_ROOT / "vllm_neuron" / "vllm" / "patches"

_PACKAGE_PREFIX = "vllm_neuron.vllm.patches"


_STUB_PACKAGES = ("vllm_neuron", "vllm_neuron.vllm", _PACKAGE_PREFIX)

#: Modules loaded here, cached so repeated fixture use returns one instance.
_LOADED: dict[str, ModuleType] = {}


@contextmanager
def _temporary_package_stubs(extra: dict[str, ModuleType]):
    """Expose stub packages and already-loaded siblings, then put ``sys.modules`` back.

    The stubs exist only so a module's absolute intra-package imports (``from
    vllm_neuron.vllm.patches.guards import PatchError``) resolve while it is being
    executed. They must not outlive that: leaving them behind makes
    ``vllm_neuron.vllm.patches`` look importable but empty, which breaks any
    later suite that imports the real package -- and does so only when both
    suites run in one session, which is exactly how CI runs them.

    Safe to drop afterwards because the executed module holds direct references
    to what it imported; nothing here re-reads ``sys.modules`` at call time.
    """
    saved = {name: sys.modules.get(name) for name in (*_STUB_PACKAGES, *extra)}
    try:
        for name in _STUB_PACKAGES:
            if sys.modules.get(name) is None:
                stub = ModuleType(name)
                stub.__path__ = []  # mark as a package
                sys.modules[name] = stub
        sys.modules.update(extra)
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def load_neuron_module(dotted: str, relative_path: str) -> ModuleType:
    """Load a dependency-free module under ``vllm_neuron`` by file path."""
    if dotted in _LOADED:
        return _LOADED[dotted]

    spec = importlib.util.spec_from_file_location(dotted, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Siblings already loaded must be visible under their real names, plus the
    # module itself so a self-referential import resolves.
    with _temporary_package_stubs({**_LOADED, dotted: module}):
        spec.loader.exec_module(module)
    _LOADED[dotted] = module
    return module


def load_patch_module(module_name: str) -> ModuleType:
    """Load ``vllm_neuron/vllm/patches/<module_name>.py`` in isolation."""
    return load_neuron_module(
        f"{_PACKAGE_PREFIX}.{module_name}",
        str((PATCHES_DIR / f"{module_name}.py").relative_to(REPO_ROOT)),
    )


@pytest.fixture(scope="session")
def guards() -> ModuleType:
    return load_patch_module("guards")


@pytest.fixture(scope="session")
def scheduler_selection() -> ModuleType:
    return load_neuron_module(
        "vllm_neuron.vllm.scheduler_selection",
        "vllm_neuron/vllm/scheduler_selection.py",
    )


@pytest.fixture(scope="session")
def deepseek_v4_config() -> ModuleType:
    """The DeepSeek-V4 config normalizer, loaded without torch or vllm."""
    return load_neuron_module(
        "vllm_neuron.model.deepseek_v4.config",
        "vllm_neuron/model/deepseek_v4/config.py",
    )


@pytest.fixture
def registry(guards: ModuleType) -> ModuleType:
    """The registry module, reset between tests.

    Module-level registration state is process-global, so each test gets a clean
    slate rather than inheriting entries from whichever test ran first.
    """
    module = load_patch_module("registry")
    module._reset_for_tests()
    yield module
    module._reset_for_tests()
