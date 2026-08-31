# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch

from vllm_neuron.model.deepseek_v4 import nki_mla
from vllm_neuron.model.deepseek_v4.attention import SharedLatentMLAInputs


def test_paged_mla_has_one_bounded_runtime_loop_call_per_scheduler_bucket():
    shared = inspect.getsource(nki_mla._paged_shared_latent_mla_kernel)
    sliding = inspect.getsource(nki_mla._paged_sliding_latent_mla_kernel)

    assert nki_mla._PAGED_KERNEL_QUERY_BUCKETS == frozenset(
        (1, 2, 4, 8, 64, 512, 1024, 2048, 4096, 8192)
    )
    for source in (shared, sliding):
        assert "q_count in (1, 2, 4, 8, 64, 512, 1024, 2048, 4096, 8192)" in source
        assert "program_id * queries_per_program" in source
        assert "nl.fori_loop(0, queries_per_program, process_query)" in source
        assert "2 * _MAX_RUNTIME_LOOP_TRIPS" in source


def test_paged_mla_gathers_the_sliding_run_once_not_per_query():
    """The sliding stream must not reach the per-row vector-indirect gather.

    The backend `unroll` pass expands a `vector_offset` gather into one DMA
    descriptor per row, which is the dominant whole-model compile cost. Each
    query's sliding window is a contiguous run of logical positions overlapping
    its neighbours by all but one, so the run is gathered once and each query
    reads an affine slice of it.
    """
    stream = inspect.getsource(nki_mla._stream_paged_query)
    span = inspect.getsource(nki_mla._build_sliding_span)

    assert "vector_offset=offsets" in stream, "the CSA stream still needs it"
    assert "src=source[row : row + width, :]" in stream
    assert "buffer=nl.private_hbm" in span

    for kernel in (
        nki_mla._paged_shared_latent_mla_kernel,
        nki_mla._paged_sliding_latent_mla_kernel,
    ):
        source = inspect.getsource(kernel)
        assert "_SPAN_GATHER and sliding_contiguous" in source
        assert "_build_sliding_span(" in source
        # Built once per query run, outside the per-query loop.
        assert source.index("_build_sliding_span(") < source.index(
            "for local_q_idx in nl.affine_range"
        )


def test_paged_mla_gathers_a_uniform_compressed_stream_once_per_launch():
    """HCA's compressed stream must not reach the per-row gather either.

    Its suffix is sized from the addressable entry capacity, so every query in a
    launch requests the same logical entries and only validity differs. One
    gather of the last query's row therefore serves the whole run, turning
    `n_q x count` per-row descriptors into `count` plus one affine read each.
    CSA keeps the per-row path: its rows are an arbitrary per-query top-k.
    """
    span = inspect.getsource(nki_mla._build_uniform_span)
    kernel = inspect.getsource(nki_mla._paged_shared_latent_mla_kernel)

    assert "buffer=nl.private_hbm" in span
    # 128 is the vector_offset partition cap, so the count must be chunked.
    assert "nl.static_range(0, count, 128)" in span

    assert "compressed_uniform: bool = False" in kernel
    assert "_build_uniform_span(" in kernel
    # Built once per query run, outside the per-query loop.
    assert kernel.index("_build_uniform_span(") < kernel.index(
        "for local_q_idx in nl.affine_range"
    )
    # The run's last query, not its first: `used` is non-decreasing across a
    # launch, so only the last row is a superset of every other query's.
    assert "first_q + queries_per_program - 1" in kernel
    # A trace-time 0 base, which takes _stream_paged_query's affine branch.
    assert "None if compressed_span is None else 0" in kernel


def test_hca_count_buckets_cover_every_compiled_history_geometry():
    """model.py may only pick counts the kernel and the dispatcher accept."""
    assert nki_mla._HCA_COUNT_BUCKETS == tuple(sorted(nki_mla._HCA_COUNT_BUCKETS))
    # CSA arrives at the same kernel with index_topk entries.
    assert 512 in nki_mla._HCA_COUNT_BUCKETS
    for count in nki_mla._HCA_COUNT_BUCKETS:
        assert 128 + count in nki_mla._HISTORY_LIMITS


def test_paged_mla_does_not_materialize_query_by_history_hbm():
    sources = "\n".join(
        inspect.getsource(function)
        for function in (
            nki_mla._stream_paged_query,
            nki_mla._paged_shared_latent_mla_kernel,
            nki_mla._paged_sliding_latent_mla_kernel,
        )
    )

    assert "(q_count, history, latent_dim)" not in sources
    assert "(q_count, heads, padded_history)" not in sources
    assert "buffer=nl.shared_hbm" not in inspect.getsource(
        nki_mla._stream_paged_query
    )


@pytest.mark.parametrize("query_count", [8, 64, 512, 1024, 2048, 4096, 8192])
def test_paged_dispatch_preserves_microchunk_and_tile_order(monkeypatch, query_count):
    launches = []

    class Wrapped:
        def __getitem__(self, grid):
            assert grid == 2

            def invoke(query, *args):
                start = (query.data_ptr() - base_pointer) // (
                    query.element_size() * query.stride(0)
                )
                launches.append((start, query.shape[0]))
                return query

            return invoke

    monkeypatch.setattr(nki_mla, "can_run_kernel", lambda tensor: True)
    monkeypatch.setattr(nki_mla, "_wrapped_paged_sliding_latent_mla", Wrapped())
    query = torch.zeros(query_count, 1, 1, 512, dtype=torch.bfloat16)
    base_pointer = query.data_ptr()
    slots = torch.zeros(query_count, 128, dtype=torch.int32)
    valid = torch.ones_like(slots, dtype=torch.bool)
    output = nki_mla.paged_shared_latent_mla(
        SharedLatentMLAInputs(
            query=query,
            sliding_cache=torch.zeros(1, 1, 128, 512, dtype=torch.bfloat16),
            sliding_slots=slots,
            sliding_valid=valid,
            compressed_cache=None,
            compressed_slots=None,
            compressed_valid=None,
            sinks=torch.zeros(1),
        )
    )

    assert output.shape == query.shape
    assert launches == [(0, query_count)]


def test_batched_decode_disables_cross_request_span_reuse(monkeypatch):
    captured = {}

    class Wrapped:
        def __getitem__(self, grid):
            assert grid == 2

            def invoke(query, *args):
                captured["sliding_contiguous"] = args[-2]
                captured["compressed_uniform"] = args[-1]
                return query

            return invoke

    monkeypatch.setattr(nki_mla, "can_run_kernel", lambda _: True)
    monkeypatch.setattr(nki_mla, "_wrapped_paged_shared_latent_mla", Wrapped())
    query = torch.zeros(4, 1, 1, 512, dtype=torch.bfloat16)
    sliding_slots = torch.zeros(4, 128, dtype=torch.int32)
    compressed_slots = torch.zeros(4, 32, dtype=torch.int32)
    output = nki_mla.paged_shared_latent_mla(
        SharedLatentMLAInputs(
            query=query,
            sliding_cache=torch.zeros(1, 1, 128, 512, dtype=torch.bfloat16),
            sliding_slots=sliding_slots,
            sliding_valid=torch.ones_like(sliding_slots, dtype=torch.bool),
            compressed_cache=torch.zeros(1, 1, 32, 512, dtype=torch.bfloat16),
            compressed_slots=compressed_slots,
            compressed_valid=torch.ones_like(compressed_slots, dtype=torch.bool),
            sinks=torch.zeros(1),
            sliding_contiguous=False,
            compressed_uniform=False,
        )
    )

    assert output.shape == query.shape
    assert captured == {
        "sliding_contiguous": False,
        "compressed_uniform": False,
    }
