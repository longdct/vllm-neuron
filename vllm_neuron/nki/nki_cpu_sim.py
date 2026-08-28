# SPDX-License-Identifier: Apache-2.0
"""NKI CPU simulator adapter for vllm-neuron.

Provides torch↔numpy conversion and a single entry point for running NKI
kernels through the built-in ``nki.simulator.simulate_kernel`` API.

Public API:
    simulate_nki_kernel(func, lnc, kwargs) → torch.Tensor | tuple[torch.Tensor, ...]

Example:
    >>> from vllm_neuron.nki.nki_cpu_sim import simulate_nki_kernel
    >>> result = simulate_nki_kernel(my_nki_kernel, 1, {"x": tensor})
"""

import logging
from collections.abc import Callable
from typing import Any

import ml_dtypes
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dtype mappings
# ---------------------------------------------------------------------------

# Torch dtypes with native numpy equivalents.
_TORCH_TO_NUMPY_DTYPE: dict[torch.dtype, np.dtype] = {
    torch.float16: np.dtype(np.float16),
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
    torch.int8: np.dtype(np.int8),
    torch.int16: np.dtype(np.int16),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
    torch.uint8: np.dtype(np.uint8),
    torch.uint16: np.dtype(np.uint16),
    torch.uint32: np.dtype(np.uint32),
    torch.bool: np.dtype(np.bool_),
}

_NUMPY_TO_TORCH_DTYPE: dict[np.dtype, torch.dtype] = {
    v: k for k, v in _TORCH_TO_NUMPY_DTYPE.items()
}

# Torch dtypes that require ml_dtypes for numpy conversion.
_TORCH_TO_ML_DTYPE: dict[torch.dtype, Any] = {
    torch.bfloat16: ml_dtypes.bfloat16,
    torch.float8_e4m3fn: ml_dtypes.float8_e4m3fn,
    torch.float8_e5m2: ml_dtypes.float8_e5m2,
}


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to numpy, preserving dtype identity for NKI.

    Native numpy dtypes go through ``tensor.numpy()`` directly. Others
    (bfloat16, float8_*) are converted via float32 then cast to the
    corresponding ml_dtypes type.

    Args:
        tensor: Input torch tensor.

    Returns:
        Numpy array with matching dtype.

    Example:
        >>> arr = _torch_to_numpy(torch.randn(4, dtype=torch.float32))
        >>> arr.dtype
        dtype('float32')
    """
    if tensor.dtype in _TORCH_TO_NUMPY_DTYPE:
        return tensor.detach().cpu().numpy()
    ml_dtype = _TORCH_TO_ML_DTYPE.get(tensor.dtype)
    if ml_dtype is not None:
        # bfloat16 / float8_* are bit-compatible with their ml_dtypes
        # counterparts. View the raw bytes (as uint8/uint16) and reinterpret
        # so the numpy array shares storage with the torch tensor — in-place
        view_dtype = torch.uint16 if tensor.dtype == torch.bfloat16 else torch.uint8
        return tensor.detach().cpu().view(view_dtype).numpy().view(ml_dtype)
    raise TypeError(f"Unsupported dtype for NKI simulation: {tensor.dtype}")


def _numpy_to_torch(arr: np.ndarray) -> torch.Tensor:
    """Convert a numpy array back to a torch tensor.

    Handles both native numpy dtypes and ml_dtypes (bfloat16, float8_*).
    For ml_dtypes, goes through a float32 intermediate.

    Args:
        arr: Input numpy array.

    Returns:
        Torch tensor with matching dtype.

    Example:
        >>> t = _numpy_to_torch(np.zeros(4, dtype=np.float32))
        >>> t.dtype
        torch.float32
    """
    native = _NUMPY_TO_TORCH_DTYPE.get(arr.dtype)
    if native is not None:
        return torch.from_numpy(np.ascontiguousarray(arr))
    # ml_dtypes: infer torch dtype by name, go through float32
    name = arr.dtype.name
    torch_dtype = getattr(torch, name, None)
    # NKI float8_e4m3 (no "fn") has no torch equivalent; map to float8_e4m3fn
    if torch_dtype is None and name == "float8_e4m3":
        torch_dtype = torch.float8_e4m3fn
    if torch_dtype is None:
        raise TypeError(f"Unsupported numpy dtype: {arr.dtype}")
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).to(
        torch_dtype
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def simulate_nki_kernel(
    func: Callable,
    lnc: int,
    kwargs: dict[str, Any],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Run an NKI kernel through the NKI CPU simulator with torch↔numpy conversion.

    Args:
        func: The raw NKI kernel function.
        lnc: LNC count (1 or 2).
        kwargs: Keyword arguments (torch tensors or scalars).

    Returns:
        Torch tensor or tuple of torch tensors.

    Example:
        >>> result = simulate_nki_kernel(my_kernel, 1, {"x": torch.randn(4)})
    """
    from nki.simulator import simulate_kernel
    from vllm_neuron.compile.platform import get_platform_target

    np_kwargs = {
        k: _torch_to_numpy(v) if isinstance(v, torch.Tensor) else v
        for k, v in kwargs.items()
    }

    # On trn2, we use fp8_e4m3 (since the finite version fp8_e4m3fn is not supported on trn3)
    if get_platform_target() == "trn2":
        for k, v in np_kwargs.items():
            if isinstance(v, np.ndarray) and v.dtype == ml_dtypes.float8_e4m3fn:
                np_kwargs[k] = v.view(ml_dtypes.float8_e4m3)

    result = simulate_kernel(func, (), np_kwargs, lnc=lnc)

    if isinstance(result, np.ndarray):
        return _numpy_to_torch(result)
    if isinstance(result, (tuple, list)):
        return tuple(
            _numpy_to_torch(r) if isinstance(r, np.ndarray) else r for r in result
        )
    return result
