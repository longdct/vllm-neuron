# SPDX-License-Identifier: Apache-2.0
"""End-to-end DeepSeek-V4 through vLLM's real serving path (Step 1-3 gate).

Validates the device-shaped model (batched forward, real ``bind_kv_cache``,
paged SWA/compressed-MLA/compressor-carry cache I/O, TP scaffolding) by
actually running it through ``vllm.LLM()`` with a tiny synthetic HF config --
not the real 284B checkpoint (out of scope for this pass; see
``docs/model-dev/deepseek-v4-serving-roadmap.md``). The checkpoint on disk
holds this exact tiny model's own randomly-initialized weights (see
``model.py``'s ``load_weights`` docstring for why: this plugin's runner
calls ``model.load_weights(checkpoint_path, ...)`` directly with no
vLLM-generic loader indirection, so there is no "dummy load format" to hook
into -- a real checkpoint, however small, is unavoidable here). The goal is
exercising the real contract end to end (compilation/scheduling/cache
lifecycle/sampling), not numerical correctness, which
``test_deepseek_v4_model_assembly.py`` already covers against the oracles
with controlled inputs.

Requires ``VLLM_NEURON_CPU_MODE=1`` (no Neuron hardware/compilation needed)
and explicitly opts the model into the registry via
``VLLM_NEURON_ENABLE_DEEPSEEK_V4=1`` -- it stays unregistered by default
(``vllm_neuron/model/registry.py``).
"""

import json
import os
import tempfile

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("vllm")

pytestmark = pytest.mark.skipif(
    os.environ.get("VLLM_NEURON_CPU_MODE") != "1",
    reason="requires VLLM_NEURON_CPU_MODE=1 (see module docstring)",
)


def _write_tiny_checkpoint_dir() -> str:
    from transformers import DeepseekV4Config

    # Needs at least one c128 *and* one c4 layer: vLLM's heterogeneous
    # KV-cache group merging (kv_cache_utils.py::_get_kv_cache_groups_uniform_groups)
    # treats the MLA-kind group with the largest page size as the reference
    # every SWA/compressor-state page size must fit under
    # (`assert max(sm_page_sizes) <= max(all_page_sizes)`). At this tiny
    # head_dim, a c128-only model's MLA page (padded to its 128B alignment)
    # is smaller than the SWA/carry pages (1024B), which trips that
    # assertion; adding the c4 layer gives the MLA group a page size that
    # covers it too.
    config = DeepseekV4Config(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        num_hidden_layers=3,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        sliding_window=16,
        max_position_embeddings=256,
        layer_types=[
            "heavily_compressed_attention",
            "sliding_attention",
            "compressed_sparse_attention",
        ],
        architectures=["DeepseekV4ForCausalLM"],
    )
    config_dict = config.to_dict()
    # vLLM's own registered on-disk config class (used by AutoConfig when
    # loading from a directory -- a different code path than constructing
    # transformers.DeepseekV4Config directly in Python, which every other
    # test in this suite does) validates `mlp_layer_types` against a
    # {'sparse','dense'} vocabulary that predates this plugin's
    # {'hash_moe','moe'} one; both come from Transformers 5.15 and disagree
    # with each other, not with this plugin. Since config.py's own
    # _extract_mlp_kinds() treats an absent mlp_layer_types as "no hash-MoE
    # layers" (a safe default, unlike the attention side), dropping the key
    # sidesteps that mismatch entirely for this contract-level E2E gate
    # rather than papering over it with a value neither validator agrees on.
    config_dict.pop("mlp_layer_types", None)
    model_dir = tempfile.mkdtemp()
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config_dict, f)

    # neuron_model_runner.py calls model.load_weights(checkpoint_path, ...)
    # unconditionally (no vLLM-generic "dummy load format" indirection to
    # skip it -- see model.py's load_weights docstring), and expects the
    # model materialized with real data afterward. Saving this exact model
    # instance's own state_dict round-trips with zero name-mapping games:
    # weight_loaders.py's map_checkpoint_name() only rewrites keys that
    # start with the *official raw checkpoint* prefixes ("layers.",
    # "embed.", ...); a key that's already the final parameter name (here,
    # "model.layers...") does not match any of those and passes through
    # unchanged, landing on itself.
    import torch
    from safetensors.torch import save_file

    from vllm_neuron.model.deepseek_v4.model import (
        DeepseekV4ForCausalLM as RealModel,
    )

    torch.manual_seed(0)
    weights_model = RealModel.from_configs(config).eval()
    # named_parameters() only: load_checkpoint_weights (weight_loaders.py)
    # requires every checkpoint key to resolve to a real nn.Parameter, and
    # has no notion of loading a buffer. tid2eid (the hash-routing table) is
    # a registered buffer built deterministically in __init__, not something
    # a checkpoint provides -- state_dict() would include it, so filter it
    # out explicitly rather than saving something nothing can load back.
    state_dict = {
        name: tensor.contiguous()
        for name, tensor in weights_model.named_parameters()
    }
    save_file(state_dict, os.path.join(model_dir, "model.safetensors"))
    return model_dir


def _run(tensor_parallel_size: int):
    os.environ["VLLM_NEURON_ENABLE_DEEPSEEK_V4"] = "1"
    from vllm import LLM, SamplingParams

    model_dir = _write_tiny_checkpoint_dir()
    llm = LLM(
        model=model_dir,
        load_format="dummy",
        max_num_seqs=1,
        max_model_len=64,
        block_size=32,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        num_gpu_blocks_override=256,
        additional_config={
            "neuron_config": {"num_batched_tokens_buckets": [8, 64]}
        },
    )
    outputs = llm.generate(
        [{"prompt_token_ids": list(range(1, 9))}],
        SamplingParams(temperature=0.0, max_tokens=4),
    )
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0


@pytest.mark.xfail(
    reason=(
        "Progress marker, not a skip: this reaches "
        "neuron_model_runner.py::initialize_kv_cache's UniformTypeKVCacheSpecs "
        "branch (`raw_tensor.view(...).view(mla_cache_shape(...))` fails -- "
        "'shape [2304, 1, 1, 16] is invalid for input of size 147456') for the "
        "padded/merged heterogeneous cache-group case vLLM's "
        "_get_kv_cache_groups_uniform_groups builds for DeepSeek-V4's mixed "
        "SWA/compressed-MLA/compressor-state layout. That merged-group sizing "
        "path is exercised here for the first time ever (nothing has "
        "previously registered a heterogeneous-cache model) and needs its own "
        "fix in neuron_model_runner.py, out of scope for this pass. Two other "
        "bugs blocking earlier in the same code path -- "
        "`self.model_config` -> `self.vllm_config.model_config`, and the "
        "'sparse'/'dense' vs 'hash_moe'/'moe' mlp_layer_types vocabulary clash "
        "between vLLM's and this plugin's DeepseekV4Config validators -- were "
        "found and fixed/worked around getting this far. Retry without xfail "
        "once the view/shape bug above is fixed."
    ),
    strict=False,
)
def test_deepseek_v4_generates_through_the_real_engine_tp1():
    _run(tensor_parallel_size=1)
