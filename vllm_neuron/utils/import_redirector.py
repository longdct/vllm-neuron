# SPDX-License-Identifier: Apache-2.0
"""Redirect ``torch_neuronx`` imports to ``libtorch_neuronx_lite``.

NKI's ``nki.framework.torch_xla`` module imports from ``torch_neuronx``
(e.g. ``from torch_neuronx.pyhlo import xla_data_pb2``).  In environments
where only ``libtorch_neuronx_lite`` is installed, this redirector
rewrites those imports so NKI works without the full ``torch-neuronx``
package.

Usage::

    from vllm_neuron.utils.import_redirector import install
    install()   # call early, before any torch_neuronx imports
"""

import importlib
import importlib.abc
import importlib.util
import sys

_PREFIX = "torch_neuronx"
_TARGET = "libtorch_neuronx_lite"


_BLOCKED = "libneuronxla"


class _TorchNeuronxToLiteRedirector(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Redirect all torch_neuronx.* imports to libtorch_neuronx_lite.*
    and block libneuronxla imports (legacy torch_neuronx dependency)."""

    def _should_redirect(self, fullname):
        return fullname == _PREFIX or fullname.startswith(_PREFIX + ".")

    def _should_block(self, fullname):
        return fullname == _BLOCKED or fullname.startswith(_BLOCKED + ".")

    def find_module(self, fullname, path=None):
        if self._should_redirect(fullname) or self._should_block(fullname):
            return self
        return None

    def find_spec(self, fullname, path=None, target=None):
        if self._should_redirect(fullname) or self._should_block(fullname):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def load_module(self, fullname):
        if self._should_block(fullname):
            raise ImportError(
                f"Import of {fullname} is blocked by vllm_neuron import redirector"
            )
        if fullname in sys.modules:
            return sys.modules[fullname]
        target = fullname.replace(_PREFIX, _TARGET, 1)
        mod = importlib.import_module(target)
        sys.modules[fullname] = mod
        return mod


_installed = False


def install():
    """Install the redirector on ``sys.meta_path`` (idempotent)."""
    from vllm_neuron import envs

    if not envs.VLLM_NEURON_LIBTORCH_NEURONX_LITE:
        return

    global _installed
    if _installed:
        return

    # Poison libneuronxla in sys.modules BEFORE any import that could
    # transitively load it (including libtorch_neuronx_lite itself).
    for _blocked_name in list(sys.modules):
        if _blocked_name == _BLOCKED or _blocked_name.startswith(_BLOCKED + "."):
            sys.modules.pop(_blocked_name, None)
    sys.modules[_BLOCKED] = None

    try:
        import libtorch_neuronx_lite
    except ImportError:
        sys.modules.pop(_BLOCKED, None)
        return

    sys.meta_path.insert(0, _TorchNeuronxToLiteRedirector())
    sys.modules.setdefault(_PREFIX, libtorch_neuronx_lite)
    _installed = True
