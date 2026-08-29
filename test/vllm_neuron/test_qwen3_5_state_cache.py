# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet state declaration and its mapping onto vLLM's MambaSpec.

The recurrent state is per *request*, not per token, which is what separates it
from every other cache kind in this backend. A paged-per-token layout would
cost ``state_bytes x seq_len`` per layer -- infeasible -- so it maps onto
``MambaSpec``, whose manager vLLM registers (single_type_kv_cache_manager.py)
and which allocates one page per request.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

from vllm_neuron.model.kv_cache import CacheKind, KVSpec, LayerSpec  # noqa: E402
from vllm_neuron.model.qwen3_5.config import Qwen3_5TextConfig  # noqa: E402
from vllm_neuron.vllm.worker.kv_spec_conversion import (  # noqa: E402
    layer_spec_to_vllm_spec,
)


def gdn_layer(name="layers.0.linear_attn", **overrides):
    kwargs = dict(
        name=name,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        cache_kind=CacheKind.MAMBA,
        state_shapes=((10240, 3), (48, 128, 128)),
        state_dtypes=(torch.bfloat16, torch.float32),
    )
    kwargs.update(overrides)
    return LayerSpec(**kwargs)


# ---------------------------------------------------------------------------
# LayerSpec validation
# ---------------------------------------------------------------------------


def test_mamba_layer_spec_accepts_two_state_tensors():
    spec = gdn_layer()
    assert spec.cache_kind is CacheKind.MAMBA
    assert len(spec.state_shapes) == len(spec.state_dtypes) == 2


def test_mamba_requires_state_geometry():
    with pytest.raises(ValueError, match="requires state_shapes and state_dtypes"):
        gdn_layer(state_shapes=None, state_dtypes=None)


def test_mamba_requires_one_dtype_per_state_tensor():
    with pytest.raises(ValueError, match="one dtype per state tensor"):
        gdn_layer(state_dtypes=(torch.float32,))


def test_mamba_rejects_a_sliding_window():
    """The state is an unbounded accumulator; a window would be a lie."""
    with pytest.raises(ValueError, match="unbounded accumulator"):
        gdn_layer(sliding_window_size=4)


def test_state_fields_rejected_on_other_cache_kinds():
    with pytest.raises(ValueError, match="only valid for mamba caches"):
        LayerSpec(
            name="layers.0.self_attn",
            num_kv_heads=4,
            head_size=256,
            dtype=torch.bfloat16,
            cache_kind=CacheKind.FULL,
            state_shapes=((4, 4),),
            state_dtypes=(torch.float32,),
        )


# ---------------------------------------------------------------------------
# Conversion to vLLM
# ---------------------------------------------------------------------------


def test_converts_to_mamba_spec_preserving_shapes_and_dtypes():
    spec = layer_spec_to_vllm_spec(gdn_layer(), block_size=32, dtype=torch.bfloat16)

    assert isinstance(spec, MambaSpec)
    assert spec.shapes == ((10240, 3), (48, 128, 128))
    assert spec.dtypes == (torch.bfloat16, torch.float32)


def test_recurrent_state_stays_fp32_under_an_fp8_kv_cache_dtype():
    """--kv-cache-dtype must not reach the accumulator.

    Every other cache kind inherits cache_config.cache_dtype. Downcasting a
    recurrent state would compound drift across every chunk of every sequence.
    """
    spec = layer_spec_to_vllm_spec(
        gdn_layer(), block_size=32, dtype=torch.float8_e4m3fn
    )
    assert spec.dtypes == (torch.bfloat16, torch.float32)


def test_page_size_covers_both_state_tensors():
    spec = layer_spec_to_vllm_spec(gdn_layer(), block_size=32, dtype=torch.bfloat16)
    expected = 10240 * 3 * 2 + 48 * 128 * 128 * 4
    assert spec.page_size_bytes == expected


def test_one_page_per_request_under_the_default_cache_mode():
    """The block table must be a single column, not max_model_len // 1."""
    from vllm.utils.math_utils import cdiv

    max_model_len = 262144
    spec = layer_spec_to_vllm_spec(
        gdn_layer(block_size=max_model_len), block_size=32, dtype=torch.bfloat16
    )
    assert cdiv(max_model_len, spec.block_size) == 1


# ---------------------------------------------------------------------------
# Realistic geometry for the shipped 27B
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp", [1, 8, 16, 32])
def test_27b_state_footprint_per_request(tp):
    """Sanity-check the per-request cost, which caps max_num_seqs."""
    from vllm_neuron.model.qwen3_5.parallel import resolve_sharding

    config = Qwen3_5TextConfig()
    policy = resolve_sharding(config, tp)

    # policy.conv_dim_per_rank, not config.conv_dim // tp: the two agree at
    # every degree up to 16 and diverge at 32, where the query and key halves
    # are replicated across the ranks sharing a key head. Sizing this cache by
    # the quotient there allocates 320 of the 448 channels the layer writes.
    conv_shape = (policy.conv_dim_per_rank, config.linear_conv_kernel_dim - 1)
    recurrent_shape = (
        policy.v_heads_per_rank,
        config.linear_key_head_dim,
        policy.v_dim_per_rank,
    )
    spec = layer_spec_to_vllm_spec(
        gdn_layer(state_shapes=(conv_shape, recurrent_shape)),
        block_size=32,
        dtype=torch.bfloat16,
    )

    per_layer = spec.page_size_bytes
    total = per_layer * len(config.linear_layer_indices)

    # The recurrent state dominates and scales down with tp.
    assert per_layer > 0
    assert total == per_layer * 48
    # At tp=16 the whole 48-layer state must stay well inside a GB per request.
    if tp == 16:
        assert total < 16 * 1024 * 1024, total
    if tp == 32:
        # Going 16 -> 32 halves the recurrent state but *grows* the conv state,
        # because q and k stop being split. Still a net win, just not a halving.
        assert policy.conv_dim_per_rank == 448
        assert total < 16 * 1024 * 1024, total


def test_kv_spec_can_mix_mamba_and_full_attention_layers():
    """The 3:1 hybrid declares two group kinds; both must validate together."""
    config = Qwen3_5TextConfig(num_hidden_layers=8)
    layers = []
    for i, kind in enumerate(config.layer_types):
        if kind == "full_attention":
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=config.num_key_value_heads,
                    head_size=config.head_dim,
                    dtype=torch.bfloat16,
                )
            )
        else:
            layers.append(gdn_layer(name=f"layers.{i}.linear_attn"))

    spec = KVSpec(layers=layers)
    kinds = {layer.cache_kind for layer in spec.layers}
    assert kinds == {CacheKind.FULL, CacheKind.MAMBA}
    assert len(spec.layers) == 8


# ---------------------------------------------------------------------------
# The two states must be declared at one dtype
# ---------------------------------------------------------------------------
#
# layer_spec_to_vllm_spec above carries a mixed pair happily, and vLLM's own
# MambaSpec is built for that -- kda_state_dtype ships a genuinely mixed pair,
# because on GPU the state writes happen inside custom kernels the tracer never
# enters. This backend traces them, so the Qwen3.5 model must not emit a mixed
# pair: the runner carves both states from a single raw page as strided views
# over one storage and the model mutates both in place, which fails with
# "aot_autograd() does not yet handle input mutations on views with different
# dtypes". No cast at the call site dodges it -- the write still lands in the
# view -- so the pair has to be equal where it is declared.


def _small_config(**overrides):
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        torch_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return Qwen3_5TextConfig(**kwargs)


@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
def test_gdn_states_are_declared_at_a_single_dtype(state_dtype):
    from vllm_neuron.model.qwen3_5.model import Qwen3_5TextForCausalLM

    config = _small_config(mamba_state_dtype=state_dtype)
    spec = Qwen3_5TextForCausalLM(config).get_kv_spec()

    mamba = [s for s in spec.layers if s.cache_kind is CacheKind.MAMBA]
    assert mamba, "the 3:1 schedule must declare linear-attention layers"
    for layer in mamba:
        assert len(layer.state_dtypes) == 2
        assert layer.state_dtypes[0] == layer.state_dtypes[1], layer.state_dtypes
        assert layer.state_dtypes[0] == getattr(torch, state_dtype)


def test_bfloat16_states_halve_the_page():
    """Halving the page is the point of the knob, and vLLM's own default."""
    from vllm_neuron.model.qwen3_5.model import Qwen3_5TextForCausalLM

    pages = {}
    for state_dtype in ("float32", "bfloat16"):
        config = _small_config(mamba_state_dtype=state_dtype)
        spec = Qwen3_5TextForCausalLM(config).get_kv_spec()
        layer = next(s for s in spec.layers if s.cache_kind is CacheKind.MAMBA)
        pages[state_dtype] = layer_spec_to_vllm_spec(
            layer, block_size=32, dtype=torch.bfloat16
        ).page_size_bytes

    assert pages["bfloat16"] * 2 == pages["float32"]


def test_an_unsupported_state_dtype_is_rejected_at_config_time():
    """Fail where the name is written, not at the first device compile."""
    with pytest.raises(ValueError, match="mamba_state_dtype"):
        _small_config(mamba_state_dtype="float16")


# ---------------------------------------------------------------------------
# Hybrid page alignment
# ---------------------------------------------------------------------------
#
# vLLM unifies two cache groups only when the largest page is an exact multiple
# of every smaller one, or when the smaller layer's backend indexes KV by block
# stride. This backend does neither, and a Mamba page has no reason to be a
# multiple of an attention page -- so a real Qwen3.5 checkpoint raised
# NotImplementedError before any layer was built. align_mamba_pages pads the
# Mamba page up via MambaSpec.page_size_padded so the first branch applies.


def _attention_spec(page_bytes, block_size=8):
    """A FullAttentionSpec whose page is exactly `page_bytes`."""
    # page = block_size * num_kv_heads * head_size * 2 (K and V) * dtype size
    head_size = page_bytes // (block_size * 2 * 2)
    return layer_spec_to_vllm_spec(
        LayerSpec(name="a", num_kv_heads=1, head_size=head_size, dtype=torch.bfloat16),
        block_size=block_size,
        dtype=torch.bfloat16,
    )


def test_align_pads_a_misaligned_mamba_page_to_a_multiple():
    from vllm_neuron.vllm.worker.kv_spec_conversion import align_mamba_pages

    attn = _attention_spec(8192)
    mamba = layer_spec_to_vllm_spec(gdn_layer(), block_size=32, dtype=torch.bfloat16)
    assert mamba.page_size_bytes % attn.page_size_bytes != 0, "fixture must misalign"

    aligned = align_mamba_pages({"a": attn, "m": mamba})

    assert aligned["a"] is attn, "attention specs must not be rewritten"
    padded = aligned["m"].page_size_bytes
    assert padded % attn.page_size_bytes == 0
    assert padded >= mamba.page_size_bytes
    # Smallest such multiple: padding must not over-allocate.
    assert padded - mamba.page_size_bytes < attn.page_size_bytes


def test_align_leaves_an_already_aligned_page_untouched():
    from vllm_neuron.vllm.worker.kv_spec_conversion import align_mamba_pages

    mamba = layer_spec_to_vllm_spec(gdn_layer(), block_size=32, dtype=torch.bfloat16)
    # An attention page that already divides the Mamba page exactly.
    attn = _attention_spec(mamba.page_size_bytes // 4)

    aligned = align_mamba_pages({"a": attn, "m": mamba})
    assert aligned["m"] is mamba


def test_align_is_a_no_op_without_both_groups():
    from vllm_neuron.vllm.worker.kv_spec_conversion import align_mamba_pages

    mamba = layer_spec_to_vllm_spec(gdn_layer(), block_size=32, dtype=torch.bfloat16)
    only_mamba = {"m": mamba}
    assert align_mamba_pages(only_mamba) is only_mamba

    attn = _attention_spec(8192)
    only_attn = {"a": attn}
    assert align_mamba_pages(only_attn) is only_attn


def test_align_satisfies_vllms_own_unification_rule():
    """The real check: vLLM's condition must hold after padding.

    Asserting the arithmetic is not enough -- what matters is that the branch
    which used to raise now takes the divisible path instead.
    """
    from vllm_neuron.vllm.worker.kv_spec_conversion import align_mamba_pages

    attn = _attention_spec(8192)
    mamba = layer_spec_to_vllm_spec(gdn_layer(), block_size=32, dtype=torch.bfloat16)
    aligned = align_mamba_pages({"a": attn, "m": mamba})

    pages = {s.page_size_bytes for s in aligned.values()}
    largest = max(pages)
    for page in pages:
        assert largest % page == 0, (largest, page)
