# SPDX-License-Identifier: Apache-2.0
"""Whole-model parity: the assembled Qwen3.5 decoder against HuggingFace.

This is the milestone the plan drives to. It exercises the real forward
contract -- attn_metadata-driven, two cache groups, paged KV for the
full-attention layers and per-request state for the Gated DeltaNet layers --
against ``transformers``' own ``Qwen3_5TextModel`` on CPU in fp32.

Weights are transferred through the production loaders rather than by hand, so
the ``+1`` norm fold and the per-head query/gate fusion are exercised in situ:
a mistake in either shows up here as a numerical difference, not just as a
loader unit-test failure.
"""

import pytest

torch = pytest.importorskip("torch")
hf_modeling = pytest.importorskip(
    "transformers.models.qwen3_5.modeling_qwen3_5",
    reason="requires a transformers build carrying the Qwen3.5 architecture",
)
from transformers.models.qwen3_5.configuration_qwen3_5 import (  # noqa: E402
    Qwen3_5TextConfig as HFQwen3_5TextConfig,
)

from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig  # noqa: E402
from vllm_neuron.model.qwen3_5.model import (  # noqa: E402
    Qwen3_5TextForCausalLM,
    attention_layer_name,
    linear_layer_name,
)
from vllm_neuron.model.qwen3_5.parallel import resolve_sharding  # noqa: E402
from vllm_neuron.model.qwen3_5.weight_loaders import (  # noqa: E402
    gated_o_proj_weight_loader,
    gated_qkv_weight_loader,
)

BLOCK_SIZE = 16
NUM_BLOCKS = 16


class FakeSlice:
    def __init__(self, tensor):
        self._tensor = tensor

    def __getitem__(self, key):
        return self._tensor[key]


def _config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=64,
        rms_norm_eps=1e-6,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        torch_dtype=torch.float32,
    )


def _hf_config(ours):
    return HFQwen3_5TextConfig(
        hidden_size=ours.hidden_size,
        intermediate_size=ours.intermediate_size,
        num_hidden_layers=ours.num_hidden_layers,
        num_attention_heads=ours.num_attention_heads,
        num_key_value_heads=ours.num_key_value_heads,
        head_dim=ours.head_dim,
        vocab_size=ours.vocab_size,
        rms_norm_eps=ours.rms_norm_eps,
        linear_num_key_heads=ours.linear_num_key_heads,
        linear_num_value_heads=ours.linear_num_value_heads,
        linear_key_head_dim=ours.linear_key_head_dim,
        linear_value_head_dim=ours.linear_value_head_dim,
        linear_conv_kernel_dim=ours.linear_conv_kernel_dim,
        hidden_act="silu",
        layer_types=ours.layer_types,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": ours.rope_theta,
            "mrope_interleaved": True,
            "mrope_section": ours.mrope_section,
            "partial_rotary_factor": ours.partial_rotary_factor,
        },
        partial_rotary_factor=ours.partial_rotary_factor,
        attn_implementation="eager",
    )


def _transfer_weights(ours_model, hf_model, config):
    """Copy HF weights into our parameter layout, through the real loaders."""
    policy = resolve_sharding(config, 1)
    hf = dict(hf_model.named_parameters())
    target = {}

    target["model.embed_tokens.weight"] = hf["embed_tokens.weight"]
    # HF's Qwen3_5RMSNorm applies (1 + w); we fold.
    target["model.norm.weight"] = 1.0 + hf["norm.weight"]

    for i, kind in enumerate(config.layer_types):
        p = f"model.layers.{i}"
        h = f"layers.{i}"

        target[f"{p}.input_layernorm.weight"] = 1.0 + hf[f"{h}.input_layernorm.weight"]
        target[f"{p}.post_attention_layernorm.weight"] = (
            1.0 + hf[f"{h}.post_attention_layernorm.weight"]
        )

        target[f"{p}.mlp.gate_proj_weight"] = hf[f"{h}.mlp.gate_proj.weight"].t()
        target[f"{p}.mlp.up_proj_weight"] = hf[f"{h}.mlp.up_proj.weight"].t()
        target[f"{p}.mlp.down_proj_weight"] = hf[f"{h}.mlp.down_proj.weight"].t()

        if kind == "full_attention":
            slices = [
                FakeSlice(hf[f"{h}.self_attn.q_proj.weight"]),
                FakeSlice(hf[f"{h}.self_attn.k_proj.weight"]),
                FakeSlice(hf[f"{h}.self_attn.v_proj.weight"]),
            ]
            target[f"{p}.self_attn.qkv_proj_weight"] = gated_qkv_weight_loader(
                config, policy
            ).load(slices, rank=0)
            target[f"{p}.self_attn.o_proj_weight"] = gated_o_proj_weight_loader(
                config, policy
            ).load([FakeSlice(hf[f"{h}.self_attn.o_proj.weight"])], rank=0)
            target[f"{p}.self_attn.q_norm.weight"] = (
                1.0 + hf[f"{h}.self_attn.q_norm.weight"]
            )
            target[f"{p}.self_attn.k_norm.weight"] = (
                1.0 + hf[f"{h}.self_attn.k_norm.weight"]
            )
        else:
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                target[f"{p}.linear_attn.{name}.weight"] = hf[
                    f"{h}.linear_attn.{name}.weight"
                ]
            target[f"{p}.linear_attn.conv1d.weight"] = hf[f"{h}.linear_attn.conv1d.weight"]
            target[f"{p}.linear_attn.dt_bias"] = hf[f"{h}.linear_attn.dt_bias"]
            target[f"{p}.linear_attn.A_log"] = hf[f"{h}.linear_attn.A_log"]
            # The gated norm keeps HF's convention -- no fold.
            target[f"{p}.linear_attn.norm.weight"] = hf[f"{h}.linear_attn.norm.weight"]

    target = {k: v.detach().clone().float() for k, v in target.items()}
    missing, unexpected = ours_model.load_state_dict(target, strict=False)
    # lm_head is not part of the HF text model; everything else must land.
    assert not unexpected, unexpected
    assert all("lm_head" in name for name in missing), missing


def _caches(model, config):
    policy = resolve_sharding(config, 1)
    caches = {}
    for i, kind in enumerate(config.layer_types):
        if kind == "full_attention":
            shape = (
                NUM_BLOCKS,
                policy.kv_heads_per_rank,
                BLOCK_SIZE,
                config.head_dim,
            )
            caches[attention_layer_name(i)] = [
                torch.zeros(shape),
                torch.zeros(shape),
            ]
        else:
            caches[linear_layer_name(i)] = [
                torch.zeros(
                    NUM_BLOCKS, config.conv_dim, config.linear_conv_kernel_dim - 1
                ),
                torch.zeros(
                    NUM_BLOCKS,
                    policy.v_heads_per_rank,
                    config.linear_key_head_dim,
                    policy.v_dim_per_rank,
                ),
            ]
    return caches


def _metadata(config, num_tokens, cached_seq_len=0, is_decode=False, start=0):
    blocks = (num_tokens + cached_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table = torch.arange(max(blocks, 1), dtype=torch.int32).view(1, -1)
    slot_mapping = torch.arange(start, start + num_tokens, dtype=torch.int64)

    attn_meta = {
        "block_table_tensor": block_table,
        "slot_mapping": slot_mapping,
        "max_query_len": 1 if is_decode else num_tokens,
        "block_size": BLOCK_SIZE,
        "max_blocks_per_seq": block_table.shape[1],
        "decode_token_threshold": 1,
        "cached_seq_len": torch.tensor([[cached_seq_len]], dtype=torch.int32),
        "kv_segment_size": 0,
    }
    state_meta = {
        # One page per request: a single-column block table.
        "block_table_tensor": torch.zeros(1, 1, dtype=torch.int32),
        "cached_seq_len": torch.tensor([[cached_seq_len]], dtype=torch.int32),
    }

    metadata = {}
    for i, kind in enumerate(config.layer_types):
        if kind == "full_attention":
            metadata[attention_layer_name(i)] = attn_meta
        else:
            metadata[linear_layer_name(i)] = state_meta
    return metadata


def _build():
    config = _config()
    torch.manual_seed(0)
    hf_model = hf_modeling.Qwen3_5TextModel(_hf_config(config)).eval().float()
    ours = Qwen3_5TextForCausalLM(config).eval().float()
    _transfer_weights(ours, hf_model, config)
    ours.bind_kv_cache(_caches(ours, config))
    return config, hf_model, ours


# ---------------------------------------------------------------------------


def test_kv_spec_declares_both_groups_on_the_schedule():
    config = _config()
    ours = Qwen3_5TextForCausalLM(config)
    spec = ours.get_kv_spec()

    from vllm_neuron.model.kv_cache import CacheKind

    kinds = {layer.name: layer.cache_kind for layer in spec.layers}
    assert len(kinds) == config.num_hidden_layers
    for i, kind in enumerate(config.layer_types):
        if kind == "full_attention":
            assert kinds[attention_layer_name(i)] is CacheKind.FULL
        else:
            assert kinds[linear_layer_name(i)] is CacheKind.MAMBA


def test_bind_kv_cache_rejects_a_missing_group():
    config = _config()
    ours = Qwen3_5TextForCausalLM(config)
    caches = _caches(ours, config)
    caches.pop(next(iter(caches)))
    with pytest.raises(KeyError, match="not initialized"):
        ours.bind_kv_cache(caches)


def test_prefill_hidden_states_match_hf():
    """The milestone: a full hybrid stack agreeing with HuggingFace."""
    config, hf_model, ours = _build()

    tokens = 12
    input_ids = torch.randint(0, config.vocab_size, (tokens,))
    positions = torch.arange(tokens, dtype=torch.int32)

    with torch.no_grad():
        expected = hf_model(
            input_ids=input_ids.unsqueeze(0),
            position_ids=positions[None, None, :].expand(3, 1, -1).contiguous(),
        ).last_hidden_state.squeeze(0)

        actual, _ = ours.model(
            input_ids,
            positions,
            attn_metadata=_metadata(config, tokens),
        )

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_every_layer_kind_actually_contributes():
    """Guard against a stack that matches for the wrong reason.

    If the linear layers were silently no-ops the prefill test could still pass
    on a lucky config, so perturb one GDN layer's output projection and require
    the result to move.
    """
    config, hf_model, ours = _build()

    tokens = 12
    input_ids = torch.randint(0, config.vocab_size, (tokens,))
    positions = torch.arange(tokens, dtype=torch.int32)

    with torch.no_grad():
        before, _ = ours.model(
            input_ids, positions, attn_metadata=_metadata(config, tokens)
        )
        gdn_index = config.linear_layer_indices[0]
        ours.model.layers[gdn_index].linear_attn.out_proj.weight.add_(0.05)
        ours.bind_kv_cache(_caches(ours, config))  # reset state
        after, _ = ours.model(
            input_ids, positions, attn_metadata=_metadata(config, tokens)
        )

    assert not torch.allclose(before, after, rtol=1e-3, atol=1e-3)


def test_mrope_positions_are_uniform_for_text_only():
    config = _config()
    ours = Qwen3_5TextForCausalLM(config)
    positions, delta = ours.get_mrope_input_positions([1, 2, 3, 4], mm_features=[])

    assert positions.shape == (3, 4)
    assert delta == 0
    # All three axes carry the same value when there is no vision input.
    torch.testing.assert_close(positions[0], positions[1])
    torch.testing.assert_close(positions[1], positions[2])
