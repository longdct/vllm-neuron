#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim: the dequant formats moved into the model package.

They now live in ``vllm_neuron/model/deepseek_v4/quant_formats.py`` so the
load path can import them; this module only kept them reachable from the
``tools/deepseek_v4`` scripts that use a sys.path-relative import.
"""

from vllm_neuron.model.deepseek_v4.quant_formats import (  # noqa: F401
    FP4_BLOCK,
    FP4_TABLE,
    FP8_BLOCK,
    dequantize,
    dequantize_fp4_blockwise,
    dequantize_fp8_blockwise,
    dequantize_fp8_per_channel,
    requantize_fp4_to_fp8,
    requantize_fp8_blockwise_to_per_channel,
    unpack_fp4,
)

__all__ = [
    "FP4_BLOCK",
    "FP4_TABLE",
    "FP8_BLOCK",
    "dequantize",
    "dequantize_fp4_blockwise",
    "dequantize_fp8_blockwise",
    "dequantize_fp8_per_channel",
    "requantize_fp4_to_fp8",
    "requantize_fp8_blockwise_to_per_channel",
    "unpack_fp4",
]
