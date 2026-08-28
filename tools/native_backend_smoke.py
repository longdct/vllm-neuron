#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile and execute a minimal TorchNeuron Native full graph."""

import torch
import torch_neuronx
from torch import nn


class SmokeModel(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(value + 1)


def main() -> None:
    model = torch.compile(
        SmokeModel().to("neuron"),
        backend="neuron",
        fullgraph=True,
        options={"model_name": "vllm_native_smoke"},
    )
    value = torch.arange(16, dtype=torch.float32).reshape(4, 4).to("neuron")
    actual = model(value)
    torch_neuronx.synchronize()
    expected = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    print("TorchNeuron Native smoke output matched CPU exactly")


if __name__ == "__main__":
    main()
