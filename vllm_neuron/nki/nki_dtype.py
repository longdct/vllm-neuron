# SPDX-License-Identifier: Apache-2.0
"""NKI → torch dtype mapping.

Maps NKI dtype strings (from CompilationResult.output_specs) back to
torch.dtype so the HOP can create output tensors with the correct type.
"""

import torch

from nki.dtype import float8_e4m3 as _NKI_FLOAT8_E4M3

from vllm_neuron.compile.platform import get_platform_target


_NKI_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "int64": torch.int64,
    "uint64": torch.uint64,
    "int32": torch.int32,
    "uint32": torch.uint32,
    "int16": torch.int16,
    "uint16": torch.uint16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}
_TORCH_DTYPE_TO_NKI = {
    torch_dtype: nki_dtype for nki_dtype, torch_dtype in _NKI_DTYPE_TO_TORCH.items()
}
_STR_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    f"torch.{name}": dtype for name, dtype in _NKI_DTYPE_TO_TORCH.items()
}


def nki_dtype_to_torch(dtype_str: str) -> torch.dtype:
    """Map an NKI dtype string (from TensorSpec) to torch.dtype."""
    if dtype_str == _NKI_FLOAT8_E4M3 and get_platform_target() == "trn2":
        return torch.float8_e4m3fn
    result = _NKI_DTYPE_TO_TORCH.get(dtype_str)
    if result is None:
        raise ValueError(f"Unsupported NKI dtype for torch: {dtype_str}")
    return result


def torch_to_nki_dtype(torch_dtype: torch.dtype) -> str:
    if torch_dtype == torch.float8_e4m3fn and get_platform_target() == "trn2":
        return _NKI_FLOAT8_E4M3
    result = _TORCH_DTYPE_TO_NKI.get(torch_dtype)
    if result is None:
        raise ValueError(f"Unsupported torch dtype for NKI: {torch_dtype}")
    return result


def str_to_torch_dtype(s: str) -> torch.dtype:
    """Convert a string like 'torch.bfloat16' to the corresponding torch.dtype."""
    dtype = _STR_TO_TORCH_DTYPE.get(s)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype string: {s}")
    return dtype
