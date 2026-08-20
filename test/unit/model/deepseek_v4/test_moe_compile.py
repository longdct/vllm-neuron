# SPDX-License-Identifier: Apache-2.0
"""Compile-safety regressions for DeepSeek-V4 routing primitives."""

import torch

from vllm_neuron.model.deepseek_v4.moe import hash_topk, routed_topk


def _assert_compile_safe(graphs):
    assert graphs, "expected torch.compile to capture a graph"
    targets = [str(node.target) for graph in graphs for node in graph.graph.nodes]
    forbidden = ("_local_scalar_dense", "_assert_scalar", "nonzero")
    assert not [target for target in targets if any(op in target for op in forbidden)]


def test_hash_topk_fullgraph_has_no_tensor_scalar_guards():
    graphs = []

    def record_graph(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(hash_topk, backend=record_graph, fullgraph=True)
    logits = torch.randn(4, 6)
    input_ids = torch.tensor([0, 1, 2, 3])
    tid2eid = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], dtype=torch.long
    )

    expected = hash_topk(logits, input_ids, tid2eid)
    actual = compiled(logits, input_ids, tid2eid)

    _assert_compile_safe(graphs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_routed_topk_fullgraph_has_no_tensor_scalar_guards():
    graphs = []

    def record_graph(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(routed_topk, backend=record_graph, fullgraph=True)
    logits = torch.randn(4, 6)
    correction_bias = torch.randn(6)

    expected = routed_topk(logits, correction_bias, 2)
    actual = compiled(logits, correction_bias, 2)

    _assert_compile_safe(graphs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_hash_topk_eager_validation_is_preserved():
    logits = torch.randn(1, 2)
    tid2eid = torch.tensor([[0], [2]], dtype=torch.long)

    try:
        hash_topk(logits, torch.tensor([1]), tid2eid)
    except ValueError as error:
        assert "outside the gate logits" in str(error)
    else:
        raise AssertionError("invalid eager expert ids must be rejected")
