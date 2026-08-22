# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner


def _metadata(table, slots, block_size=4):
    table = torch.tensor(table)
    return {
        "block_table_tensor": table,
        "slot_mapping": torch.tensor(slots),
        "block_size": block_size,
        "max_blocks_per_seq": table.shape[1],
    }


def _runner(**caches):
    return SimpleNamespace(_kv_cache_full_tensors=caches)


def test_validator_allows_minus_one_padding_and_heterogeneous_capacities():
    runner = _runner(mla=torch.zeros(3, 1, 8, 2), kv=torch.zeros(2, 2, 1, 4, 2))
    metadata = {
        "mla": _metadata([[0, 2, -1]], [0, 11, -1]),
        "kv": _metadata([[0, 1, -1]], [0, 7, -1]),
    }
    NeuronModelRunner._validate_attention_metadata(runner, metadata)


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (_metadata([[0, -2]], [0]), "physical block id"),
        (_metadata([[0, 3]], [0]), "physical block id"),
        (_metadata([[0]], [-2]), "slot mapping"),
        (_metadata([[0]], [12]), "slot mapping"),
        (_metadata([[0]], [0], block_size=0), "logical page width"),
    ],
)
def test_validator_rejects_invalid_bounds(metadata, match):
    runner = _runner(mla=torch.zeros(3, 1, 8, 2))
    with pytest.raises(ValueError, match=match):
        NeuronModelRunner._validate_attention_metadata(runner, {"mla": metadata})
