# SPDX-License-Identifier: Apache-2.0
"""NKI Higher-Order-Operator integration for torch.compile / FX tracing.

Public API:
    wrap_nki(nki_kernel)  → NKIHOPCaller   (supports [grid](**kwargs) syntax)
    can_run_kernel(device) → bool
    NKIKernelWrapper       (type used by FX passes)

The goal is to have NKI kernels show up like this in the FX graph:

%nki_attention = call_function[target=nki_kernel_wrapper](
    kernel_idx=3,           # int
    grid=[2],               # list of ints
    backend_config="eyJm...",  # string (encoded JSON containing kernel info + binary)
    operand_output_aliases={0: 1},  # dict of ints
    ...
)

In order to do this, we provide wrap_nki(nki_kernel), which returns a
NKIHOPCaller. When the user calls that object, it invokes a NKIKernelWrapper
which plugs into Torch's HigherOrderOperator. This dispatches to the correct
implementation based on the current mode (e.g. fake tensor, XLA, etc.)
"""

import inspect
import logging
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dynamo import allow_in_graph
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    unset_fake_temporarily,
)
from torch.fx.experimental.proxy_tensor import (
    ProxyTorchDispatchMode,
    disable_proxy_modes_tracing,
    track_tensor_tree,
)

from vllm_neuron import envs

try:
    from .nki_compile import NKICompileResult, compile_nki
except (ImportError, KeyError):
    NKICompileResult = None
    compile_nki = None

if TYPE_CHECKING:
    from .nki_compile import NKICompileResult
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI

NKIGridType = tuple[int, ...] | list[int]
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel registry — maps idx → (func, arg_names, default_args)
# ---------------------------------------------------------------------------


class NKIRegistry:
    funcs: ClassVar[dict[int, Callable]] = {}
    func_to_id: ClassVar[dict[Callable, int]] = {}
    arg_names: ClassVar[dict[int, list[str]]] = {}
    default_args: ClassVar[dict[int, dict[str, Any]]] = {}
    constant_args: ClassVar[dict[int, dict[str, Any]]] = {}

    def add(
        self, func: Callable, names: list[str], defaults: dict[str, Any] | None = None
    ) -> int:
        if func in self.func_to_id:
            return self.func_to_id[func]
        idx = len(self.funcs)
        self.funcs[idx] = func
        self.func_to_id[func] = idx
        self.arg_names[idx] = names
        self.default_args[idx] = defaults or {}
        return idx

    def get_func(self, idx: int) -> Callable:
        return self.funcs[idx]

    def get_arg_names(self, idx: int) -> list[str]:
        return self.arg_names[idx]

    def get_default_args(self, idx: int) -> dict[str, Any]:
        return self.default_args[idx]

    def add_constant_args(self, args: dict[str, Any], key: int):
        self.constant_args[key] = args

    def get_constant_args(self, key: int) -> dict[str, Any]:
        return self.constant_args[key]

    def reset(self):
        self.funcs.clear()
        self.func_to_id.clear()
        self.arg_names.clear()
        self.default_args.clear()
        self.constant_args.clear()


kernel_registry = NKIRegistry()


# ---------------------------------------------------------------------------
# Arg merging helpers
# ---------------------------------------------------------------------------


def _merge_names_args(names: Sequence[str], args: Sequence[Any]) -> dict[str, Any]:
    return dict(zip(names, args))


def _merge_all_args(
    kernel_idx: int,
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
) -> dict[str, Any]:
    all_names = kernel_registry.get_arg_names(kernel_idx)
    constants = kernel_registry.get_constant_args(constant_args_key)
    tensor_dict = _merge_names_args(arg_names, args)
    return {n: tensor_dict[n] if n in tensor_dict else constants[n] for n in all_names}


def _build_kernel_kwargs(
    kernel_idx: int,
    args: Sequence[Any],
    arg_names: Sequence[str],
) -> dict[str, Any]:
    """Build a complete kwargs dict from args, filling in registered defaults."""
    all_names = kernel_registry.get_arg_names(kernel_idx)
    defaults = kernel_registry.get_default_args(kernel_idx)
    combined = dict(zip(arg_names, args))
    for name in all_names:
        if name not in combined:
            combined[name] = defaults.get(name)
    return combined


# ---------------------------------------------------------------------------
# Compilation helper
# ---------------------------------------------------------------------------


def _get_kernel_config(
    kernel_idx: int,
    grid: NKIGridType,
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
) -> "NKICompileResult":
    func = kernel_registry.get_func(kernel_idx)
    if constant_args_key == -1:
        merged = _merge_names_args(arg_names, args)
    else:
        merged = _merge_all_args(kernel_idx, args, arg_names, constant_args_key)

    # NKI-790: NKI V3 doesn't support FakeTensors yet.
    # Use Meta-tensors instead as a temporary workaround.
    # TODO: revert to FakeTensor once the issue is resolved.
    with unset_fake_temporarily():
        meta_args = {}
        for k, v in merged.items():
            if isinstance(v, FakeTensor):
                meta_args[k] = torch.empty(v.shape, dtype=v.dtype, device="meta")
            else:
                meta_args[k] = v
        grid_tuple = (grid,) if isinstance(grid, int) else tuple(grid)
        return compile_nki(func, meta_args, grid_tuple)


# ---------------------------------------------------------------------------
# HOP definition
# ---------------------------------------------------------------------------


class NKIKernelWrapper(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("nki_kernel_wrapper", cacheable=True)

    def __call__(
        self,
        kernel_idx: int,
        grid: NKIGridType,
        backend_config: str,
        operand_output_aliases: dict[int, int],
        args: Sequence[Any],
        arg_names: Sequence[str],
        constant_args_key: int,
    ) -> Any:
        # In CPU sim mode, skip NKI compilation (not available) and dispatch
        # directly. The CPU dispatch key will run the simulator.
        if envs.VLLM_NEURON_CPU_MODE and os.environ.get("NKI_SIMULATOR") == "1":
            return super().__call__(
                kernel_idx=kernel_idx,
                grid=grid,
                backend_config="",
                operand_output_aliases={},
                args=args,
                arg_names=arg_names,
                constant_args_key=constant_args_key,
            )
        return super().__call__(
            kernel_idx=kernel_idx,
            grid=grid,
            backend_config=backend_config,
            operand_output_aliases=operand_output_aliases,
            args=args,
            arg_names=arg_names,
            constant_args_key=constant_args_key,
        )


nki_kernel_wrapper = NKIKernelWrapper()
nki_kernel_wrapper = allow_in_graph(nki_kernel_wrapper)


# ---------------------------------------------------------------------------
# Meta dispatch — creates output tensors with correct shape/dtype
# ---------------------------------------------------------------------------


@nki_kernel_wrapper.py_impl(torch._C.DispatchKey.Meta)
def _meta_impl(
    *,
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    result = _get_kernel_config(kernel_idx, grid, args, arg_names, constant_args_key)
    if len(result.return_types) == 1:
        dtype, shape = result.return_types[0]
        return torch.empty(shape, dtype=dtype, device="meta")
    return tuple(
        torch.empty(shape, dtype=dtype, device="meta")
        for dtype, shape in result.return_types
    )


# ---------------------------------------------------------------------------
# XLA dispatch — emits HLO custom call
# ---------------------------------------------------------------------------


def _to_pyhlo_type(scribe: Any, dtype: torch.dtype) -> Any:
    """Map torch dtype to pyhlo scribe attribute name."""
    name = (
        str(dtype)
        .removeprefix("torch.")
        .replace("float", "f")
        .replace("uint", "u")
        .replace("int", "s")
        .replace("_", "")
    )
    return getattr(scribe, name)


class NkiKernel:
    """Emits an AwsNeuronNkiKernel HLO custom call."""

    def __init__(self, func: Callable, grid: tuple[int, ...]) -> None:
        self.func = func
        self.grid = grid

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        func = kernel_registry.get_func(kwargs.get("kernel_idx", 0))
        grid = kwargs.get("grid", self.grid)
        arg_names = kwargs.get("arg_names", [])
        constant_args_key = kwargs.get("constant_args_key", -1)
        kernel_idx = kwargs.get("kernel_idx", 0)
        tensor_args = kwargs.get("args", args)

        result = _get_kernel_config(
            kernel_idx, grid, tensor_args, arg_names, constant_args_key
        )

        inputs = [a for a in tensor_args if isinstance(a, torch.Tensor)]

        def opfn(*hlo_args):
            from torch_neuronx.xla_impl.custom_call_targets import AwsNeuronNkiKernel

            scribe = hlo_args[0].scribe
            output_tys = [
                _to_pyhlo_type(scribe, dtype)[shape]
                for dtype, shape in result.return_types
            ]
            if len(output_tys) == 1:
                return output_tys[0].CustomCall(
                    *hlo_args[: len(inputs)],
                    backend_config=str.encode(result.dumped_config),
                    custom_call_target=AwsNeuronNkiKernel,
                )
            return scribe.tuple(*output_tys).CustomCall(
                *hlo_args[: len(inputs)],
                backend_config=str.encode(result.dumped_config),
                custom_call_target=AwsNeuronNkiKernel,
            )

        from torch_neuronx.xla_impl.base import xla_call, AwsNeuronCustomLoweringType

        hlo_call = AwsNeuronCustomLoweringType.hlo_register(
            f"_{func.__name__}__Nki_Kernel_Call_Impl", opfn
        )
        rewritten = xla_call(
            hlo_call, name=f"{func.__name__}_AwsNeuronNkiKernelWrapper"
        )(*inputs)

        return rewritten if len(result.return_types) == 1 else tuple(rewritten)


@nki_kernel_wrapper.py_impl(torch._C.DispatchKey.XLA)
def _xla_impl(
    *,
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
) -> Any:
    func = kernel_registry.get_func(kernel_idx)
    grid_tuple = (grid,) if isinstance(grid, int) else tuple(grid)
    kernel = NkiKernel(func, grid_tuple)
    return kernel(
        kernel_idx=kernel_idx,
        grid=grid,
        args=args,
        arg_names=arg_names,
        constant_args_key=constant_args_key,
    )


# ---------------------------------------------------------------------------
# FakeTensor dispatch
# ---------------------------------------------------------------------------


def _materialize_fake_arg(a: Any) -> Any:
    """Convert FakeTensors and symbolic types to concrete values for the NKI simulator."""
    if isinstance(a, torch.Tensor):
        return torch.ones([int(s) for s in a.shape], dtype=a.dtype)
    if isinstance(a, torch.SymInt):
        return int(a)
    if isinstance(a, torch.SymFloat):
        return float(a)
    if isinstance(a, torch.SymBool):
        return bool(a)
    return a


@nki_kernel_wrapper.py_impl(FakeTensorMode)
def _fake_impl(
    mode: FakeTensorMode,
    *,
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    device = next((a.device for a in args if isinstance(a, torch.Tensor)), None)
    if device is None:
        raise ValueError("No tensor arguments — cannot infer device")

    # In CPU sim mode, the NKI compiler is not available for shape inference.
    # Run the simulator on ones-filled tensors to determine output shape/dtype.
    # Ones are used instead of zeros to avoid edge cases (e.g., division by
    # zero in normalization kernels). This only runs during dynamo tracing
    # (once per kernel signature), not during execution — the CPU dispatch
    # key handles actual execution.
    if envs.VLLM_NEURON_CPU_MODE and os.environ.get("NKI_SIMULATOR") == "1":
        from .nki_cpu_sim import simulate_nki_kernel

        func = kernel_registry.get_func(kernel_idx)
        lnc = grid[0] if grid else 1
        with unset_fake_temporarily():
            real_args = [_materialize_fake_arg(a) for a in args]
            combined = _build_kernel_kwargs(kernel_idx, real_args, arg_names)
            result = simulate_nki_kernel(func, lnc, combined)
        with mode:
            if isinstance(result, torch.Tensor):
                return torch.empty(result.shape, dtype=result.dtype).to(device)
            return tuple(
                torch.empty(r.shape, dtype=r.dtype).to(device)
                if isinstance(r, torch.Tensor)
                else r
                for r in result
            )

    result = _get_kernel_config(kernel_idx, grid, args, arg_names, constant_args_key)
    with mode:
        if len(result.return_types) == 1:
            dtype, shape = result.return_types[0]
            return torch.empty(shape, dtype=dtype).to(device)
        return tuple(
            torch.empty(shape, dtype=dtype).to(device)
            for dtype, shape in result.return_types
        )


# ---------------------------------------------------------------------------
# Proxy dispatch (torch.compile tracing)
# ---------------------------------------------------------------------------


def _split_tensor_constant(
    kernel_idx: int,
    grid: NKIGridType,
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
) -> tuple[Sequence[Any], Sequence[str], int]:
    """Separate tensor args from constants; register constants if needed."""
    if constant_args_key != -1:
        return args, arg_names, constant_args_key

    tensor_args, tensor_names = [], []
    constant_args = {}
    for idx, name in enumerate(arg_names):
        if isinstance(args[idx], torch.Tensor):
            tensor_args.append(args[idx])
            tensor_names.append(name)
        else:
            constant_args[name] = args[idx]

    # Hash constants for caching
    key = hash(tuple(sorted(constant_args.items())))
    kernel_registry.add_constant_args(constant_args, key)
    return tensor_args, tensor_names, key


@nki_kernel_wrapper.py_impl(ProxyTorchDispatchMode)
def _proxy_impl(
    proxy_mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
) -> Any:
    tensor_args, tensor_names, ckey = _split_tensor_constant(
        kernel_idx, grid, args, arg_names, constant_args_key
    )
    result = _get_kernel_config(kernel_idx, grid, args, arg_names, constant_args_key)

    node_args = {
        "kernel_idx": kernel_idx,
        "grid": grid,
        "backend_config": result.dumped_config,
        "operand_output_aliases": result.operand_output_aliases,
        "args": tensor_args,
        "arg_names": tensor_names,
        "constant_args_key": ckey,
    }
    with disable_proxy_modes_tracing():
        out = nki_kernel_wrapper(**node_args)

    proxy_args = pytree.tree_map(
        proxy_mode.tracer.unwrap_proxy,
        node_args,
    )
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        nki_kernel_wrapper,
        (),
        proxy_args,
        name=f"nki_{kernel_registry.get_func(kernel_idx).__name__}_proxy",
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


# ---------------------------------------------------------------------------
# Functionalize dispatch (mutation tracking)
# ---------------------------------------------------------------------------


@nki_kernel_wrapper.py_functionalize_impl
def _functionalize_impl(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
) -> Any:
    unwrapped = [
        ctx.unwrap_tensors(a) if isinstance(a, torch.Tensor) else a for a in args
    ]
    result = _get_kernel_config(
        kernel_idx, grid, unwrapped, arg_names, constant_args_key
    )

    with ctx.redispatch_to_next():
        outputs = nki_kernel_wrapper(
            kernel_idx=kernel_idx,
            grid=grid,
            backend_config=result.dumped_config,
            operand_output_aliases=result.operand_output_aliases,
            args=unwrapped,
            arg_names=arg_names,
            constant_args_key=constant_args_key,
        )

    tuple_outputs = (
        (outputs,) if not isinstance(outputs, (list, tuple)) else tuple(outputs)
    )
    tensor_inputs = [a for a in args if isinstance(a, torch.Tensor)]
    for input_idx, output_idx in result.operand_output_aliases.items():
        inp = tensor_inputs[input_idx]
        ctx.replace(inp, tuple_outputs[output_idx])
        ctx.mark_mutation_hidden_from_autograd(inp)
        ctx.commit_update(inp)
        ctx.sync(inp)

    return ctx.wrap_tensors(outputs)


# ---------------------------------------------------------------------------
# Fallthrough dispatch keys
# ---------------------------------------------------------------------------

nki_kernel_wrapper.fallthrough(DispatchKey.PythonDispatcher)
nki_kernel_wrapper.fallthrough(DispatchKey.PythonTLSSnapshot)
nki_kernel_wrapper.fallthrough(DispatchKey.ADInplaceOrView)
nki_kernel_wrapper.fallthrough(DispatchKey.BackendSelect)
nki_kernel_wrapper.fallthrough(DispatchKey.AutocastCPU)
nki_kernel_wrapper.fallthrough(DispatchKey.AutocastCUDA)
nki_kernel_wrapper.fallthrough(DispatchKey.AutocastPrivateUse1)
nki_kernel_wrapper.fallthrough(DispatchKey.AutogradCUDA)
nki_kernel_wrapper.fallthrough(DispatchKey.AutogradCPU)


# ---------------------------------------------------------------------------
# CPU dispatch — runs NKI kernels through the NKI CPU simulator
# ---------------------------------------------------------------------------


@nki_kernel_wrapper.py_impl(DispatchKey.CPU)
def _cpu_impl(
    *,
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if os.environ.get("NKI_SIMULATOR") != "1":
        raise RuntimeError(
            "NKI kernel dispatched to CPU without the simulator enabled. "
            "Set NKI_SIMULATOR=1 to enable the NKI CPU simulator."
        )
    from .nki_cpu_sim import simulate_nki_kernel

    func = kernel_registry.get_func(kernel_idx)
    lnc = grid[0] if grid else 1
    combined = _build_kernel_kwargs(kernel_idx, args, arg_names)
    return simulate_nki_kernel(func, lnc, combined)


# ---------------------------------------------------------------------------
# Public API: NKIHOPCaller + wrap_nki + can_run_kernel
# ---------------------------------------------------------------------------


class NKIHOPCaller:
    """Callable returned by wrap_nki(). Supports [grid](**kwargs) syntax."""

    def __init__(
        self,
        kernel_idx: int,
        grid: Optional[NKIGridType],
        arg_names: list[str],
        default_args: dict[str, Any],
    ):
        self.kernel_idx = kernel_idx
        self.grid = [] if grid is None else grid
        self.kernel_arg_names = arg_names
        self.kernel_default_args = default_args

    def __getitem__(self, grid: NKIGridType) -> "NKIHOPCaller":
        if isinstance(grid, int):
            grid = (grid,)
        elif not isinstance(grid, tuple):
            grid = tuple(map(int, grid))
        return NKIHOPCaller(
            self.kernel_idx, grid, self.kernel_arg_names, self.kernel_default_args
        )

    def __call__(self, *args, **kwargs):
        combined = {
            **_merge_names_args(self.kernel_arg_names, args),
            **kwargs,
        }
        for name in self.kernel_arg_names:
            if name not in combined:
                combined[name] = self.kernel_default_args[name]
        ordered = tuple(combined[n] for n in self.kernel_arg_names)

        return nki_kernel_wrapper(
            kernel_idx=self.kernel_idx,
            grid=self.grid,
            backend_config="",
            operand_output_aliases={},
            args=ordered,
            arg_names=self.kernel_arg_names,
            constant_args_key=-1,
        )


def can_run_kernel(device: torch.Tensor | str = "") -> bool:
    """Check if NKI kernels can run on the given device."""
    if envs.VLLM_NEURON_DISABLE_NKI_KERNELS:
        return False
    if envs.VLLM_NEURON_CPU_MODE:
        return os.environ.get("NKI_SIMULATOR") == "1"
    device_str = str(device.device) if isinstance(device, torch.Tensor) else device
    return device_str != "cpu"


# This will execute the kernel registration during tracing
# and fold it to the constant kernel idx.
@torch._dynamo.assume_constant_result
def register_kernel_to_torch(
    func: Callable, arg_names: list[str], default_args: dict[str, Any]
) -> int:
    return kernel_registry.add(func, arg_names, default_args)


def wrap_nki(nki_kernel: Any, **kwargs: Any) -> NKIHOPCaller:
    """Wrap an NKI kernel for use with torch.compile / FX tracing.

    Returns an NKIHOPCaller.
    """
    sig = (
        nki_kernel.sign
        if hasattr(nki_kernel, "sign")
        else inspect.signature(nki_kernel)
    )
    arg_names = list(sig.parameters.keys())
    default_args = {
        name: p.default
        for name, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    func = nki_kernel.func if hasattr(nki_kernel, "func") else nki_kernel
    idx = register_kernel_to_torch(func, arg_names, default_args)
    return NKIHOPCaller(idx, [], arg_names, default_args)
