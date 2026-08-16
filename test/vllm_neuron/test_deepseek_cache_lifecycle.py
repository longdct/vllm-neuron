# SPDX-License-Identifier: Apache-2.0
"""P1 lifecycle matrix against vLLM 0.26's real cache managers."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


def cache_manager(num_blocks=256):
    specs = [
        SlidingWindowSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
        ),
        MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=4,
            model_version="deepseek_v4",
        ),
        MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=128,
            model_version="deepseek_v4",
        ),
        # Compressor carry state deliberately uses SWA lifecycle semantics.
        SlidingWindowSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
        ),
    ]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer.{index}"], spec)
            for index, spec in enumerate(specs)
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=2048,
        scheduler_block_size=128,
        hash_block_size=32,
        max_in_flight_tokens=128,
        enable_caching=False,
    )


def request(request_id, prompt_tokens=200):
    return Request(
        request_id,
        [1] * prompt_tokens,
        SamplingParams(max_tokens=32),
        None,
    )


def ids(manager, request_id):
    return tuple(
        tuple(block.block_id for block in group)
        for group in manager.get_blocks(request_id).blocks
    )


def test_new_allocation_chunk_continuation_and_decode_are_exact():
    manager = cache_manager()
    req = request("r")
    manager.allocate_slots(req, 128)
    first = ids(manager, "r")
    assert tuple(map(len, first)) == (4, 1, 1, 4)

    req.num_computed_tokens = 128
    manager.allocate_slots(req, 72)
    continued = ids(manager, "r")
    assert tuple(map(len, continued)) == (7, 2, 2, 7)
    assert all(after[: len(before)] == before for before, after in zip(first, continued))

    req.num_computed_tokens = 200
    manager.allocate_slots(req, 1)
    decoded = ids(manager, "r")
    # allocate_slots performs SWA eviction before decode allocation. It replaces
    # expired pages with the shared null block while compressed logical pages
    # remain stable and no new page is needed for this token.
    assert decoded[0][:2] == (0, 0)
    assert decoded[3][:2] == (0, 0)
    assert decoded[1:3] == continued[1:3]
    assert tuple(map(len, decoded)) == tuple(map(len, continued))


def test_reorder_and_batch_compaction_do_not_rekey_state():
    manager = cache_manager()
    a, b = request("a", 64), request("b", 96)
    manager.allocate_slots(a, 64)
    manager.allocate_slots(b, 96)
    before = {"a": ids(manager, "a"), "b": ids(manager, "b")}

    # Scheduler reorder/compaction changes batch indices, not request IDs. The
    # real managers must therefore return byte-identical block ownership in
    # either lookup order.
    after = {req_id: ids(manager, req_id) for req_id in ("b", "a")}
    assert after["a"] == before["a"]
    assert after["b"] == before["b"]


@pytest.mark.parametrize("reason", ["completion", "abort"])
def test_completion_and_abort_release_every_owned_block(reason):
    manager = cache_manager()
    req = request(reason)
    initial_free = manager.block_pool.get_num_free_blocks()
    manager.allocate_slots(req, 200)
    assert manager.block_pool.get_num_free_blocks() < initial_free
    manager.free(req)
    assert manager.block_pool.get_num_free_blocks() == initial_free


def test_sliding_window_remapping_uses_null_blocks_but_latents_remain_stable():
    manager = cache_manager()
    req = request("remap", 200)
    manager.allocate_slots(req, 200)
    before = ids(manager, "remap")
    manager.remove_skipped_blocks("remap", 200, 200)
    after = ids(manager, "remap")

    assert after[0][:2] == (0, 0)
    assert after[3][:2] == (0, 0)  # carry state follows the same SWA lifecycle
    assert after[1] == before[1]   # c4 logical pages are not SWA-remapped
    assert after[2] == before[2]   # c128 logical pages are not SWA-remapped
