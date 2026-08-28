# SPDX-License-Identifier: Apache-2.0
# Custom collective implementations for Neuron backend (AutogradPrivateUse1)
# These are needed because PyTorch's _c10d_functional ops don't have native
# implementations for the Neuron device, so we register placeholder implementations
# to prevent NotImplementedError during torch.compile with vllm_neuron backend

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.library import impl


@impl("_c10d_functional::all_gather_into_tensor", "AutogradPrivateUse1")
def custom_all_gather_into_tensor_autograd_neuron(input_tensor, group_size, group_name):
    # torch.compiler.is_compiling() returns false here even if it is being called
    # by torch.compile(). We cannot assert this does not get invoked in eager mode.
    # Instead we check if the input tensors is FakeTensor that Dynamo passes.
    if not isinstance(input_tensor, FakeTensor):
        raise NotImplementedError(
            "all_gather_into_tensor is only implemented for compile flow"
        )

    # Mock output shape for fx graph - all_gather concatenates along dim 0
    output_shape = list(input_tensor.shape)
    output_shape[0] *= group_size
    return torch.zeros(
        output_shape, dtype=input_tensor.dtype, device=input_tensor.device
    )


@impl("_c10d_functional::all_reduce", "AutogradPrivateUse1")
def custom_all_reduce_autograd_neuron(input_tensor, reduce_op, group_name):
    # torch.compiler.is_compiling() returns false here even if it is being called
    # by torch.compile(). We cannot assert this does not get invoked in eager mode.
    # Instead we check if the input tensors is FakeTensor that Dynamo passes.
    if not isinstance(input_tensor, FakeTensor):
        raise NotImplementedError("all_reduce is only implemented for compile flow")

    # Mock output shape for fx graph - all_reduce preserves input shape
    return torch.zeros_like(input_tensor)


@impl("_c10d_functional::reduce_scatter_tensor", "AutogradPrivateUse1")
def custom_reduce_scatter_tensor_autograd_neuron(
    input_tensor, reduce_op, group_size, group_name
):
    # torch.compiler.is_compiling() returns false here even if it is being called
    # by torch.compile(). We cannot assert this does not get invoked in eager mode.
    # Instead we check if the input tensors is FakeTensor that Dynamo passes.
    if not isinstance(input_tensor, FakeTensor):
        raise NotImplementedError(
            "reduce_scatter_tensor is only implemented for compile flow"
        )

    # Mock output shape for fx graph - reduce_scatter divides last dimension by group_size
    output_shape = list(input_tensor.shape)
    output_shape[0] //= group_size
    return torch.zeros(
        output_shape, dtype=input_tensor.dtype, device=input_tensor.device
    )


@impl("_c10d_functional::broadcast", "AutogradPrivateUse1")
def custom_broadcast_autograd_neuron(input_tensor, src_rank, group_name):
    # torch.compiler.is_compiling() returns false here even if it is being called
    # by torch.compile(). We cannot assert this does not get invoked in eager mode.
    # Instead we check if the input tensors is FakeTensor that Dynamo passes.
    if not isinstance(input_tensor, FakeTensor):
        raise NotImplementedError("broadcast is only implemented for compile flow")

    # Mock output shape for fx graph - broadcast preserves input shape
    return torch.zeros_like(input_tensor)


@impl("_c10d_functional::all_to_all", "AutogradPrivateUse1")
def custom_all_to_all_autograd_neuron(
    input_tensor, output_split_sizes, input_split_sizes, group_name
):
    # torch.compiler.is_compiling() returns false here even if it is being called
    # by torch.compile(). We cannot assert this does not get invoked in eager mode.
    # Instead we check if the input tensors is FakeTensor that Dynamo passes.
    if not isinstance(input_tensor, FakeTensor):
        raise NotImplementedError("all_to_all is only implemented for compile flow")

    # Mock output shape for fx graph - all_to_all preserves total elements but redistributes them
    return torch.zeros_like(input_tensor)


@impl("_c10d_functional::all_to_all_single", "AutogradPrivateUse1")
def custom_all_to_all_single_autograd_neuron(
    input_tensor, output_split_sizes, input_split_sizes, group_name
):
    if not isinstance(input_tensor, FakeTensor):
        raise NotImplementedError(
            "all_to_all_single is only implemented for compile flow"
        )

    # Mock output shape for fx graph - all_to_all_single preserves shape on dim 0
    return torch.zeros_like(input_tensor)


@impl("_c10d_functional::wait_tensor", "AutogradPrivateUse1")
def custom_wait_tensor_autograd_neuron(tensor):
    # wait_tensor is needed because functional collectives are asynchronous by design.
    # When Dynamo converts dist.all_reduce() to functional collectives, it creates:
    # 1. _c10d_functional.all_reduce() - returns a "future" tensor
    # 2. _c10d_functional.wait_tensor() - waits for the collective to complete
    # This ensures proper synchronization before the tensor is used.

    # torch.compiler.is_compiling() returns false here even if it is being called
    # by torch.compile(). We cannot assert this does not get invoked in eager mode.
    # Instead we check if the input tensors is FakeTensor that Dynamo passes.
    if not isinstance(tensor, FakeTensor):
        raise NotImplementedError("wait_tensor is only implemented for compile flow")

    # For FX graph tracing, wait_tensor is a no-op that returns the input tensor
    return torch.zeros_like(tensor)
