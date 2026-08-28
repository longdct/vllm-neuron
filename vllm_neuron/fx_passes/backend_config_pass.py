# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Any, Dict, Tuple

from .base import FXPass

from vllm_neuron.nki.nki_hop import (
    NKIKernelWrapper,
    _get_kernel_config,
)


def _resolve_arg_value(gm: torch.fx.GraphModule, value: Any) -> Any:
    """Resolve an NKI-kernel arg (FX node or literal) to a concrete value.

    Tensor args carry their fake/meta tensor in ``meta['example_value']`` (set by
    Dynamo) — that is what the kernel-config machinery expects for them. But a
    kernel may also take a NON-tensor compile-time object (e.g. rotational_topk's
    ``RotationalTopkConfig``, an ``nl.NKIObject``). Dynamo represents such an
    object not as a constant but by RECONSTRUCTING it in the graph: a
    ``call_function`` node whose target is the object's class and whose kwargs are
    the (constant) fields, possibly nesting further constructor nodes. Those nodes
    have no ``example_value``. Reproduce the value the same way the FX interpreter
    would at execution time — call the constructor with resolved args/kwargs — so
    the config-arg case stops crashing this pass on a missing ``example_value``.
    """
    if isinstance(value, torch.fx.Node):
        node = value
        if "example_value" in node.meta:
            # Tensor input — use the fake/meta tensor Dynamo attached.
            return node.meta["example_value"]
        if node.op == "get_attr":
            return getattr(gm, node.target)
        if node.op == "call_function" and isinstance(node.target, type):
            # A constructor node (e.g. RotationalTopkConfig(**fields)). Rebuild it
            # exactly as graph execution would, recursing into nested constructors.
            ctor_args = tuple(_resolve_arg_value(gm, a) for a in node.args)
            ctor_kwargs = {k: _resolve_arg_value(gm, v) for k, v in node.kwargs.items()}
            return node.target(*ctor_args, **ctor_kwargs)
        raise KeyError(
            f"NKI kernel arg node {node.name!r} (op={node.op}, "
            f"target={node.target!r}) has no 'example_value' and is not a "
            f"constructor/get_attr node that can be resolved to a value"
        )
    if isinstance(value, (tuple, list)):
        return type(value)(_resolve_arg_value(gm, v) for v in value)
    if isinstance(value, dict):
        return {k: _resolve_arg_value(gm, v) for k, v in value.items()}
    return value


class NkiKernelWriteBackendConfigPass(FXPass):
    """Transforms NKI kernel calls in FX graph to include the backend config string,
    which is used to detect changes to the kernel call for the purposes of caching.

    Example transformation:
        Before: call_function[target=torch.ops.higher_order.nki_kernel_wrapper](args = (), kwargs = {kernel_idx: 0, grid: [], backend_config: 0
        After:  call_function[target=torch.ops.higher_order.nki_kernel_wrapper](args = (), kwargs = {kernel_idx: 0, grid: [], backend_config: eyJrZXJuZWxfd...

    The backend config is unique to a traced kernel and the compile-time arguments it receives
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        """Return the pass name."""
        return "nki_kernel_backend_config_pass"

    def run(
        self, gm: torch.fx.GraphModule, **kwargs
    ) -> Tuple[torch.fx.GraphModule, Dict]:
        """Execute the NkiKernelWriteBackendConfigPass.

        Args:
            gm: The PyTorch FX GraphModule to transform
            **kwargs: Additional arguments (target_device, etc.)

        Returns:
            Tuple containing the transformed GraphModule and metadata

        Raises:
            RuntimeError: If the transformation fails
        """
        try:
            kernel_call_count = 0
            for node in gm.graph.nodes:
                if node.op != "call_function":
                    continue

                if type(node.target) != NKIKernelWrapper:
                    continue

                node_kwargs = dict(node.kwargs.copy())
                kernel_idx = node_kwargs["kernel_idx"]
                grid = node_kwargs["grid"]
                args = tuple(
                    _resolve_arg_value(gm, arg_node) for arg_node in node_kwargs["args"]
                )
                arg_names = node_kwargs["arg_names"]
                constant_args_key = node_kwargs["constant_args_key"]
                result = _get_kernel_config(
                    kernel_idx, grid, args, arg_names, constant_args_key
                )

                node_kwargs["backend_config"] = result.dumped_config
                # operand_output_aliases from NKI uses tensor-only input indices (removes
                # scalar/None/compile-time-object args). We need to remap those to use
                # the index within the full arg list. A kernel's tensor inputs are the
                # arg nodes that carry an 'example_value' (a fake/meta tensor); a
                # non-tensor object arg (e.g. a config) is a node WITHOUT example_value
                # and must not be counted as a tensor input here.
                tensor_positions = [
                    i
                    for i, _arg in enumerate(node_kwargs["args"])
                    if isinstance(_arg, torch.fx.Node) and "example_value" in _arg.meta
                ]
                remapped_aliases = {
                    tensor_positions[input_idx]: output_idx
                    for input_idx, output_idx in result.operand_output_aliases.items()
                }
                node_kwargs["operand_output_aliases"] = remapped_aliases

                node.kwargs = node_kwargs
                kernel_call_count += 1

            # Recompile and return
            gm.recompile()
            metadata = {"kernel_call_count": kernel_call_count}

            return gm, metadata

        except Exception as e:
            raise RuntimeError(
                f"Writing backend config to NKI kernel calls failed: {str(e)}"
            ) from e
