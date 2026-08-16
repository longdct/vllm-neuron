# SPDX-License-Identifier: Apache-2.0

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nki")

from vllm_neuron.model.deepseek_v4.nki_mla import simulate_512_mla, torch_reference


pytestmark = pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="P2.b requires explicit NKI_SIMULATOR=1",
)


@pytest.mark.parametrize(
    ("query_length", "context_length", "causal"),
    [(1, 8, False), (8, 8, True)],
)
def test_512d_prefill_and_decode_match_fp32_reference(
    query_length, context_length, causal
):
    generator = torch.Generator().manual_seed(11 + query_length)
    query = torch.randn(
        1, query_length, 512, generator=generator, dtype=torch.bfloat16
    )
    key = torch.randn(
        1, context_length, 512, generator=generator, dtype=torch.bfloat16
    )
    value = torch.randn(
        1, context_length, 512, generator=generator, dtype=torch.bfloat16
    )
    expected = torch_reference(query, key, value, causal=causal)
    actual = simulate_512_mla(query, key, value, causal=causal)
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)
