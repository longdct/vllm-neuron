# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch

from vllm_neuron.model.deepseek_v4 import model as deepseek_model
from vllm_neuron.model.deepseek_v4 import nki_compressor
from vllm_neuron.model.deepseek_v4.model import DeepseekV4Compressor
from vllm_neuron.model.deepseek_v4.nki_compressor import (
    _completion_candidates,
    _paged_candidate_windows,
)


def test_compressor_kernel_has_one_runtime_candidate_body():
    source = inspect.getsource(nki_compressor._paged_gated_compressor_kernel)

    assert "nl.fori_loop" in source
    assert "nl.affine_range" not in source
    assert "scalar_offset=window_base_reg" in source
    assert "scalar_offset=output_base_reg" in source
    assert "vector_offset=offsets" in source
    assert "buffer=nl.shared_hbm" in source  # output only
    assert "(candidate_count, window, state_width)" not in source


def test_packed_compressor_uses_nki_before_portable_per_query_gather():
    source = inspect.getsource(DeepseekV4Compressor.forward_packed)

    assert "VLLM_NEURON_DSV4_NKI_COMPRESSOR" in source
    assert "paged_gated_compressor(" in source
    assert source.index("paged_gated_compressor(") < source.index(
        "gather_recent_window_batched("
    )
    assert inspect.signature(DeepseekV4Compressor.forward_packed).return_annotation in (
        None,
        "None",
    )


def test_completion_candidates_are_boundary_only_and_reject_padded_tail():
    ratio = 128
    query = 512
    start = 37
    real = 460
    positions = torch.cat(
        (
            torch.arange(start, start + real),
            torch.full((query - real,), start + real - 1),
        )
    )
    owners = torch.zeros(query, dtype=torch.long)
    owners[real:] = 1
    slots = torch.where(
        ((positions + 1) % ratio == 0) & (torch.arange(query) < real),
        torch.arange(query) + 11,
        torch.full((query,), -1),
    )

    indices, rope_positions, selected_slots, valid = _completion_candidates(
        positions, owners, slots, ratio
    )

    assert indices.tolist() == [90, 218, 346, 474]
    assert rope_positions.tolist() == [0, 128, 256, 384]
    assert valid.tolist() == [True, True, True, False]
    assert selected_slots[:3].tolist() == [101, 229, 357]


def test_candidate_windows_cross_shuffled_pages_and_mask_null_blocks():
    ratio = 4
    head_dim = 128
    width = 2 * head_dim
    page = 4
    state_cache = torch.zeros(6, 1, page, 2 * width, dtype=torch.bfloat16)
    # Logical pages 0,1,2 map non-monotonically; page 1 is a null block.
    block_tables = torch.tensor([[4, -1, 2]], dtype=torch.long)
    positions = torch.arange(3, 11, dtype=torch.long)
    owners = torch.zeros(8, dtype=torch.long)
    output_slots = torch.where(
        (positions + 1) % ratio == 0,
        torch.arange(8, dtype=torch.long),
        torch.full((8,), -1, dtype=torch.long),
    )

    slots, mask, candidate_positions, selected_slots, valid = _paged_candidate_windows(
        state_cache,
        positions,
        owners,
        block_tables,
        output_slots,
        ratio=ratio,
        overlap=True,
    )

    assert slots.shape == (2, 8)
    assert candidate_positions.tolist() == [0, 4]
    assert selected_slots.tolist() == [0, 4]
    assert valid.tolist() == [True, True]
    # First candidate: negative prior history is masked. Second candidate:
    # logical page 1 is null while the surrounding shuffled pages remain live.
    assert (mask[0] == float("-inf")).tolist() == [True] * 4 + [False] * 4
    assert (mask[1] == float("-inf")).tolist() == [False] * 4 + [True] * 4
    assert slots[0, 4:].tolist() == [16, 17, 18, 19]


def test_candidate_count_is_one_for_q1_and_ceil_query_over_ratio_otherwise():
    for query, ratio, expected in (
        (1, 128, 1),
        (129, 128, 2),
        (257, 128, 3),
        (9, 4, 3),
    ):
        positions = torch.arange(query)
        owners = torch.zeros(query, dtype=torch.long)
        output_slots = torch.where(
            (positions + 1) % ratio == 0,
            positions,
            torch.full_like(positions, -1),
        )
        indices, _, _, _ = _completion_candidates(
            positions, owners, output_slots, ratio
        )
        assert indices.shape == (expected,)


def test_completion_candidates_are_partitioned_per_padded_request():
    ratio = 4
    positions = torch.tensor(
        [
            0,
            1,
            2,
            3,
            3,
            3,
            3,
            3,  # request 0: four real rows, then padding
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,  # request 1: full segment
            *([0] * 8),  # request 2: decode-bucket padding row
            *([20] * 8),  # request 3: just admitted, no completed window
        ]
    )
    owners = torch.arange(4).repeat_interleave(8)
    slots = torch.full((32,), -1, dtype=torch.long)
    slots[3], slots[9], slots[13] = 103, 201, 205

    indices, rope_positions, selected_slots, valid = _completion_candidates(
        positions, owners, slots, ratio, num_requests=4
    )

    assert indices.tolist() == [3, 7, 9, 13, 19, 23, 27, 31]
    assert rope_positions.tolist() == [0, 4, 8, 12, 0, 4, 20, 24]
    assert valid.tolist() == [True, False, True, True, False, False, False, False]
    assert selected_slots[valid].tolist() == [103, 201, 205]


def test_candidate_windows_use_each_candidates_request_block_table():
    ratio = 4
    page = 4
    width = 256
    positions = torch.tensor(
        [0, 1, 2, 3, 3, 3, 3, 3, 10, 11, 12, 13, 14, 15, 16, 17]
    )
    owners = torch.tensor([0] * 8 + [1] * 8)
    output_slots = torch.full((16,), -1, dtype=torch.long)
    output_slots[3], output_slots[9], output_slots[13] = 3, 9, 13
    block_tables = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [10, 11, 12, 13, 14],
        ]
    )
    state_cache = torch.zeros(20, 1, page, 2 * width, dtype=torch.bfloat16)

    slots, _, _, _, valid = _paged_candidate_windows(
        state_cache,
        positions,
        owners,
        block_tables,
        output_slots,
        ratio=ratio,
        overlap=True,
    )

    assert valid.tolist() == [True, False, True, True]
    # Request 1's two windows cover logical 4..11 and 8..15. They must use
    # physical blocks 11/12 and 12/13 from row 1, never row 0's 2/3 and 3/4.
    assert slots[2].tolist() == list(range(44, 52))
    assert slots[3].tolist() == list(range(48, 56))


def test_two_candidate_kernel_launch_is_padded_to_four(monkeypatch):
    captured = {}

    class FakeKernel:
        def __getitem__(self, lnc):
            captured["lnc"] = lnc

            def launch(_cache, slots, _mask, valid, _bias, _overlap):
                captured["slots"] = slots
                captured["valid"] = valid
                return torch.zeros(slots.shape[0], 128)

            return launch

    monkeypatch.setattr(nki_compressor, "can_run_kernel", lambda _: True)
    monkeypatch.setattr(
        nki_compressor, "_wrapped_paged_gated_compressor", FakeKernel()
    )
    positions = torch.arange(3, 11)
    output_slots = torch.where(
        (positions + 1) % 4 == 0,
        torch.arange(8),
        torch.full((8,), -1),
    )

    reduced, _, _, _ = nki_compressor.paged_gated_compressor(
        torch.zeros(1, 1, 128, 512, dtype=torch.bfloat16),
        positions,
        torch.zeros(8, dtype=torch.long),
        torch.tensor([[0]]),
        output_slots,
        torch.zeros(4, 256, dtype=torch.bfloat16),
        ratio=4,
        overlap=True,
    )

    assert captured["lnc"] == 2
    assert captured["slots"].shape == (4, 8)
    assert captured["valid"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert reduced.shape == (2, 128)


def test_forward_packed_nki_path_finalizes_only_boundary_candidates(monkeypatch):
    query = 512
    head_dim = 128
    compressor = DeepseekV4Compressor(
        hidden_size=8,
        head_dim=head_dim,
        ratio=4,
        rms_norm_eps=1e-6,
        rotary_emb=lambda entry, **_: (entry[..., :1], entry[..., :1]),
        qk_rope_head_dim=2,
    ).to(torch.bfloat16)
    compressor.state_cache = torch.zeros(40, 1, 16, 4 * head_dim, dtype=torch.bfloat16)
    positions = torch.arange(5, 5 + query)
    owners = torch.zeros(query, dtype=torch.long)
    state_tables = torch.arange(40).reshape(1, -1)
    state_slots = positions.clone()
    output_slots = torch.where(
        (positions + 1) % 4 == 0,
        torch.arange(query),
        torch.full((query,), -1),
    )
    reduced = torch.randn(query // 4, head_dim)
    selected = output_slots[torch.nonzero(output_slots >= 0).reshape(-1)]
    candidate_positions = (positions[(positions + 1) % 4 == 0] // 4) * 4
    scatters = []

    monkeypatch.setattr(deepseek_model, "can_run_kernel", lambda _: True)
    monkeypatch.setattr(
        deepseek_model,
        "paged_gated_compressor",
        lambda *_, **__: (
            reduced,
            candidate_positions,
            selected,
            torch.ones(query // 4, dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(
        deepseek_model,
        "finalize_compressed_entries",
        lambda entry, *_: entry,
    )
    monkeypatch.setattr(
        deepseek_model,
        "scatter_paged_latent",
        lambda cache, slots, values: scatters.append((slots, values)),
    )
    monkeypatch.setattr(
        deepseek_model,
        "gather_recent_window_batched",
        lambda *_: (_ for _ in ()).throw(AssertionError("portable gather used")),
    )

    result = compressor.forward_packed(
        torch.zeros(query, 8, dtype=torch.bfloat16),
        positions=positions,
        token_to_request=owners,
        state_block_tables=state_tables,
        state_slot_mapping=state_slots,
        mla_cache=torch.zeros(40, 1, 4, head_dim, dtype=torch.bfloat16),
        mla_slot_mapping=output_slots,
    )

    assert result is None
    assert len(scatters) == 2
    assert scatters[0][1].shape == (query, 4 * head_dim)
    assert torch.equal(scatters[1][0], selected)
    assert torch.equal(scatters[1][1], reduced)


def test_forward_packed_environment_switch_uses_portable_fallback(monkeypatch):
    compressor = DeepseekV4Compressor(
        hidden_size=8,
        head_dim=512,
        ratio=128,
        rms_norm_eps=1e-6,
        rotary_emb=lambda entry, **_: (entry[..., :1], entry[..., :1]),
        qk_rope_head_dim=2,
    ).to(torch.bfloat16)
    compressor.state_cache = torch.zeros(1, 1, 128, 1024, dtype=torch.bfloat16)
    monkeypatch.setenv("VLLM_NEURON_DSV4_NKI_COMPRESSOR", "0")
    monkeypatch.setattr(deepseek_model, "can_run_kernel", lambda _: True)
    monkeypatch.setattr(deepseek_model, "scatter_paged_latent", lambda *_: None)
    monkeypatch.setattr(
        deepseek_model,
        "paged_gated_compressor",
        lambda *_: (_ for _ in ()).throw(AssertionError("NKI path used")),
    )
    monkeypatch.setattr(
        deepseek_model,
        "gather_recent_window_batched",
        lambda *_: (_ for _ in ()).throw(RuntimeError("portable fallback selected")),
    )

    with pytest.raises(RuntimeError, match="portable fallback selected"):
        compressor.forward_packed(
            torch.zeros(1, 8, dtype=torch.bfloat16),
            positions=torch.tensor([127]),
            token_to_request=torch.tensor([0]),
            state_block_tables=torch.tensor([[0]]),
            state_slot_mapping=torch.tensor([127]),
            mla_cache=torch.zeros(1, 1, 1, 512, dtype=torch.bfloat16),
            mla_slot_mapping=torch.tensor([0]),
        )
