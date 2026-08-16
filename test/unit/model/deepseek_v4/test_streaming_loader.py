# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.streaming_loader import (
    dequantize_symmetric,
    stream_into_final_tensors,
)


def test_streams_quantized_shards_directly_into_final_tensors():
    destinations = {
        "layer.0": torch.empty(4, 8, dtype=torch.bfloat16),
        "layer.1": torch.empty(2, 8, dtype=torch.bfloat16),
    }
    yielded = []

    def sources():
        for name, shape in (("layer.0", (4, 8)), ("layer.1", (2, 8))):
            yielded.append(name)
            yield name, torch.arange(torch.tensor(shape).prod()).reshape(shape).to(torch.int8)

    stats = stream_into_final_tensors(
        sources(),
        destinations,
        convert=lambda tensor, dtype: dequantize_symmetric(
            tensor, dtype, scale=0.25
        ),
    )
    assert yielded == ["layer.0", "layer.1"]
    assert stats.tensors_loaded == 2
    assert stats.peak_temporary_bytes == 4 * 8 * (1 + 2)
    torch.testing.assert_close(
        destinations["layer.0"],
        (torch.arange(32).reshape(4, 8).float() * 0.25).to(torch.bfloat16),
    )


def test_missing_duplicate_and_shape_mismatch_fail_loudly():
    destination = {"w": torch.empty(2)}
    with pytest.raises(KeyError, match="were not loaded"):
        stream_into_final_tensors([], destination)
    with pytest.raises(ValueError, match="duplicate"):
        stream_into_final_tensors(
            [("w", torch.ones(2)), ("w", torch.ones(2))], destination
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        stream_into_final_tensors([("w", torch.ones(3))], destination)
