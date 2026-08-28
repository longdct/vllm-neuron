# SPDX-License-Identifier: Apache-2.0
"""NKI kernel compilation bridge.

Provides compile_nki() which compiles NKI kernels via CompileKernel
and returns the fields the HOP infrastructure needs.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch

from nki.language.buffers import shared_hbm
from nki.language.tensor import NkiTensor
from nki.framework.compiled import CompileKernel

from vllm_neuron import envs

from ..compile.platform import get_platform_target
from .nki_dtype import nki_dtype_to_torch, torch_to_nki_dtype

logger = logging.getLogger(__name__)


@dataclass
class NKICompileResult:
    """Compilation output consumed by the HOP dispatch implementations."""

    dumped_config: str
    return_types: tuple[tuple[torch.dtype, tuple[int, ...]], ...]
    operand_output_aliases: dict[int, int]


def compile_nki(
    func: Callable,
    args: dict[str, Any],
    grid: tuple[int, ...],
) -> NKICompileResult:
    """Compile an NKI kernel via CompileKernel.

    Results are persisted to the vLLM Neuron compile cache so that
    warm starts skip the expensive BIR compilation step.

    Args:
        func: Raw NKI kernel function.
        args: Parameter name → value (torch.Tensor or scalar).
        grid: LNC grid, e.g. (2,).

    Returns:
        NKICompileResult with everything the HOP needs.
    """
    from .nki_cache import compile_with_cache, create_nki_cache_key

    cache_key = create_nki_cache_key(func, args, grid)

    def _do_compile() -> NKICompileResult:
        lnc = grid[0] if grid else 1
        kernel = CompileKernel(func=func, lnc=lnc, target=get_platform_target())
        inputs = {name: _convert_input(v, name) for name, v in args.items()}

        compile_opts = kernel._compile_opts()

        if envs.VLLM_NEURON_KERNEL_DEVICE_DUMP:
            # Runtime env vars (NEURON_RT_DEBUG_OUTPUT_DIR, NEURON_RT_DEBUG_SAVE_BINARY)
            # are set in executor.py worker_process().
            # Default output dir: /tmp/vllm_neuron_kernel_device_dumps
            from nki.compiler.frontend import TracerFrontend

            compile_opts = replace(compile_opts, enable_device_dump=True)
            frontend = TracerFrontend(enable_backend_opt=kernel._enable_backend_opt)
        else:
            frontend = kernel._frontend_cls(
                enable_backend_opt=kernel._enable_backend_opt,
            )

        nir = kernel._cached_compile_to_bir(
            frontend=frontend,
            inputs=inputs,
            compile_opts=compile_opts,
        )

        config = nir.build_config()

        return NKICompileResult(
            dumped_config=config.backend_config_b64.decode("ascii"),
            return_types=tuple(
                (nki_dtype_to_torch(s.dtype), tuple(s.shape))
                for s in config.output_specs
            ),
            operand_output_aliases=config.operand_output_aliases,
        )

    return compile_with_cache(cache_key, _do_compile)


def _convert_input(x: Any, name: str) -> Any:
    if isinstance(x, torch.Tensor):
        # convert to NkiTensor
        return NkiTensor(
            name=name,
            shape=x.shape,
            dtype=torch_to_nki_dtype(x.dtype),
            storage=None,
            buffer=shared_hbm,
        )

    return x
