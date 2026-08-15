# SPDX-License-Identifier: Apache-2.0
"""``InputBatch`` per-group parameter derivation, against the installed vLLM.

Covers the P0.3 break: vLLM 0.26 dropped ``pin_memory`` from ``InputBatch``,
made ``max_num_blocks_per_req`` required, and added ``slot_mapping_modes``. The
first two are silent-at-import, loud-at-startup; the third defaults to ``None``
and so would be silently wrong rather than raising.

The heterogeneous multi-group cases are the ones P1 depends on: until groups
with genuinely different block sizes and per-request block counts coexist, the
"heterogeneous cache" support is untested. These construct a real ``InputBatch``
rather than only checking the derived lists, because agreeing with our own
helper proves nothing about whether upstream accepts the result.
"""

import math

import pytest

vllm = pytest.importorskip("vllm", reason="requires vLLM")

import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.block_table import SlotMappingMode

from vllm_neuron.vllm.worker.input_batch_params import (
    InputBatchGroupParams,
    build_input_batch_group_params,
    mla_cache_shape,
)

MAX_MODEL_LEN = 1024


@pytest.fixture(scope="module")
def vllm_config():
    """A minimal config carrying only what the derivation actually reads.

    ``max_num_blocks_per_req`` consults ``parallel_config`` (for decode context
    parallelism) and, for Mamba specs, ``cache_config``. Nothing here loads a
    model, so ``ModelConfig`` is skipped entirely.
    """
    return VllmConfig(
        cache_config=CacheConfig(),
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=MAX_MODEL_LEN, is_encoder_decoder=False
        ),
    )


def _full(block_size: int, **kw) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        **kw,
    )


def _swa(block_size: int, window: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=window,
    )


def _mla(block_size: int, **kw) -> MLAAttentionSpec:
    """A latent MLA spec: one 512-wide head, which is DeepSeek-V4's real shape."""
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        **kw,
    )


def _config(*specs) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[f"layer.{i}"], kv_cache_spec=spec)
            for i, spec in enumerate(specs)
        ],
    )


class TestSingleGroup:
    def test_one_full_attention_group(self, vllm_config):
        params = build_input_batch_group_params(
            _config(_full(16)), vllm_config, MAX_MODEL_LEN
        )
        assert params.block_sizes == [16]
        assert params.kernel_block_sizes == [16]
        assert params.max_num_blocks_per_req == [MAX_MODEL_LEN // 16]
        assert params.slot_mapping_modes == [SlotMappingMode.TOKEN_TO_KV_SLOT]

    def test_block_count_rounds_up(self, vllm_config):
        """A non-dividing length must round up, or the last tokens have no block."""
        params = build_input_batch_group_params(
            _config(_full(16)), vllm_config, max_model_len=1000
        )
        assert params.max_num_blocks_per_req == [63]  # cdiv(1000, 16), not 62


class TestHeterogeneousGroups:
    """Groups with genuinely different geometry -- the P1 prerequisite."""

    def test_block_sizes_and_counts_differ_per_group(self, vllm_config):
        params = build_input_batch_group_params(
            _config(_swa(16, 128), _mla(32), _full(64)),
            vllm_config,
            MAX_MODEL_LEN,
        )
        assert params.block_sizes == [16, 32, 64]
        # Not one value repeated: each group pages at its own granularity.
        assert params.max_num_blocks_per_req == [64, 32, 16]
        assert params.slot_mapping_modes == [SlotMappingMode.TOKEN_TO_KV_SLOT] * 3

    def test_group_order_is_preserved(self, vllm_config):
        """Order is the only thing linking a spec to its index in every list."""
        params = build_input_batch_group_params(
            _config(_full(64), _swa(16, 128)), vllm_config, MAX_MODEL_LEN
        )
        assert params.block_sizes == [64, 16]
        assert params.max_num_blocks_per_req == [16, 64]

    def test_compressed_mla_groups_coexist(self, vllm_config):
        """c4 and c128 compressed latents alongside an uncompressed one.

        ``compress_ratio`` is the 0.26 field DeepSeek-V4's compressed caches use.
        The block table is sized in native token positions, so a compressed group
        does not shrink its row length -- asserted here so a future upstream
        change that starts dividing by ``compress_ratio`` is caught rather than
        silently halving every compressed group's block table.
        """
        params = build_input_batch_group_params(
            _config(_mla(32), _mla(32, compress_ratio=4), _mla(32, compress_ratio=128)),
            vllm_config,
            MAX_MODEL_LEN,
        )
        assert params.block_sizes == [32, 32, 32]
        assert params.max_num_blocks_per_req == [32, 32, 32]


class TestEncoderOnlyGroupsAreSkipped:
    def test_encoder_only_group_contributes_nothing(self, vllm_config):
        """Skipping must drop the entry entirely, not emit a placeholder.

        An encoder-only group has no per-request block table. Emitting a
        placeholder would keep the lists long enough to look right while shifting
        every later group's index by one.
        """
        encoder_only = EncoderOnlyAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
        )
        params = build_input_batch_group_params(
            _config(encoder_only, _full(32)), vllm_config, MAX_MODEL_LEN
        )
        # Only the trailing decoder group survives.
        assert params.block_sizes == [32]
        assert params.max_num_blocks_per_req == [MAX_MODEL_LEN // 32]


class TestEncoderDecoderIsRejected:
    def test_encoder_decoder_fails_before_block_tables_are_undersized(self):
        """Neuron has no max_encoder_len sizing path yet (risk R1)."""
        with pytest.raises(NotImplementedError, match="encoder-decoder"):
            build_input_batch_group_params(
                _config(_full(16)),
                vllm_config=VllmConfig(
                    cache_config=CacheConfig(),
                    device_config=DeviceConfig(device="cpu"),
                    parallel_config=ParallelConfig(),
                    scheduler_config=SchedulerConfig(
                        max_model_len=MAX_MODEL_LEN, is_encoder_decoder=True
                    ),
                ),
                max_model_len=MAX_MODEL_LEN,
                is_encoder_decoder=True,
            )


class TestAlignmentInvariant:
    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            InputBatchGroupParams(
                block_sizes=[16, 32],
                kernel_block_sizes=[16, 32],
                max_num_blocks_per_req=[64],  # one short
                slot_mapping_modes=[SlotMappingMode.TOKEN_TO_KV_SLOT] * 2,
            )

    @pytest.mark.parametrize("ratio", [1, 4, 128])
    def test_mla_is_one_physical_latent_with_compressed_storage(self, ratio):
        spec = _mla(128, compress_ratio=ratio)
        shape = mla_cache_shape(spec, num_blocks=3)
        assert shape == (3, 1, 128 // ratio, 512)
        # No leading K/V dimension: allocation exactly matches spec accounting.
        assert math.prod(shape) * torch.tensor([], dtype=spec.dtype).element_size() == (
            3 * spec.real_page_size_bytes
        )

    def test_compression_ratio_must_divide_native_block(self):
        with pytest.raises(ValueError, match="must be divisible"):
            mla_cache_shape(_mla(32, compress_ratio=128), num_blocks=1)


class TestUpstreamAcceptsTheResult:
    """Feed the derived parameters to the real ``InputBatch``.

    This is the part that actually closes P0.3: our lists agreeing with our own
    helper says nothing about whether upstream's constructor accepts them. If
    0.27 renames or re-shapes any of these, this fails here rather than at
    engine startup on a Trn2 instance.
    """

    @pytest.fixture(autouse=True)
    def _no_pinned_memory(self, monkeypatch):
        """Allocate host tensors unpinned.

        ``PIN_MEMORY`` is resolved from the active platform at import time, and
        the registered Neuron platform reports it as available. Actually pinning
        then calls into device hooks that only exist with the Neuron runtime
        present, so off-hardware it raises. Pinning is a host-transfer
        optimization and changes no value or shape being asserted here.
        """
        for module in ("vllm.v1.worker.gpu_input_batch", "vllm.v1.worker.block_table"):
            monkeypatch.setattr(f"{module}.PIN_MEMORY", False, raising=False)

    @pytest.mark.parametrize(
        "specs",
        [
            pytest.param((_full(16),), id="single-full"),
            pytest.param((_swa(16, 128), _mla(32), _full(64)), id="heterogeneous"),
        ],
    )
    def test_input_batch_constructs(self, vllm_config, specs):
        from vllm.v1.worker.gpu_input_batch import InputBatch

        params = build_input_batch_group_params(
            _config(*specs), vllm_config, MAX_MODEL_LEN
        )
        batch = InputBatch(
            max_num_reqs=4,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=256,
            device=torch.device("cpu"),
            vocab_size=128,
            block_sizes=params.block_sizes,
            kernel_block_sizes=params.kernel_block_sizes,
            max_num_blocks_per_req=params.max_num_blocks_per_req,
            slot_mapping_modes=params.slot_mapping_modes,
            logitsprocs=None,
            logitsprocs_need_output_token_ids=False,
            is_pooling_model=False,
        )
        assert len(batch.block_table.block_tables) == len(params.block_sizes)

    def test_pin_memory_is_no_longer_a_parameter(self):
        """The specific 0.26 removal that broke the old call site.

        Kept as its own assertion because passing it raised ``TypeError`` at
        engine startup -- late, on hardware. If a future release reinstates it,
        this fails and the call site can be revisited deliberately.
        """
        import inspect

        from vllm.v1.worker.gpu_input_batch import InputBatch

        assert "pin_memory" not in inspect.signature(InputBatch.__init__).parameters
