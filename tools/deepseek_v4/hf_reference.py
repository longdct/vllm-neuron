#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Load a native-named DeepSeek-V4 slice into the ``transformers`` reference model.

``vllm_neuron.model.deepseek_v4.weight_loaders.map_checkpoint_name`` maps the
checkpoint's native names onto *this plugin's* module tree (it mirrors vLLM's
``WeightsMapper``). The ``transformers`` reference uses a different tree again --
``self_attn`` rather than ``attn``, ``input_layernorm`` rather than ``attn_norm``,
the indexer nested under the compressor, and per-expert weights stacked into two
3-D parameters. This module is that second mapping, which is what makes an
independent oracle possible.

Traps worth naming, all of which produce plausible-looking wrong numbers rather
than errors:

* ``gate_up_proj[e] = cat([w1, w3], dim=0)`` -- gate first, then up, concatenated
  on the *output* dimension, because ``_apply_gate`` does ``chunk(2, dim=-1)`` on
  the ``F.linear`` result.
* ``indexer`` sits under ``compressor`` in the reference but is a sibling of it in
  the checkpoint, and the checkpoint's ``indexer.compressor.*`` collapses to
  ``compressor.indexer.*``.
* ``rotary_emb.*_inv_freq`` are derived buffers that no checkpoint contains; the
  model builds them itself, so they are expected to be absent from the state dict.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

#: Applied in order to the part of the name after ``layers.N.``.
#: Longest/most specific first -- ``indexer.compressor`` must be rewritten before
#: any bare ``compressor`` rule can see it.
_LAYER_RULES: tuple[tuple[str, str], ...] = (
    ("attn.indexer.compressor.", "attn.compressor.indexer."),
    ("attn.indexer.weights_proj", "attn.compressor.indexer.scorer.weights_proj"),
    ("attn.indexer.", "attn.compressor.indexer."),
    # Indexer leaves need their own rules: re-parenting puts ``indexer.`` between
    # ``compressor.`` and the leaf, so the generic ``compressor.<leaf>`` rules
    # below no longer match them.
    ("indexer.wgate", "indexer.gate_proj"),
    ("indexer.wkv", "indexer.kv_proj"),
    ("indexer.norm.weight", "indexer.kv_norm.weight"),
    ("indexer.ape", "indexer.position_bias"),
    # Leaf renames, safe once the indexer has been re-parented.
    ("compressor.wgate", "compressor.gate_proj"),
    ("compressor.wkv", "compressor.kv_proj"),
    ("compressor.norm.weight", "compressor.kv_norm.weight"),
    ("compressor.ape", "compressor.position_bias"),
    ("attn.wq_a", "attn.q_a_proj"),
    ("attn.wq_b", "attn.q_b_proj"),
    ("attn.wkv", "attn.kv_proj"),
    ("attn.wo_a", "attn.o_a_proj"),
    ("attn.wo_b", "attn.o_b_proj"),
    ("attn.q_norm", "attn.q_a_norm"),
    ("attn.attn_sink", "attn.sinks"),
    ("indexer.wq_b", "indexer.q_b_proj"),
    # Container renames last, so the rules above can match on ``attn.``/``ffn.``.
    ("attn_norm.weight", "input_layernorm.weight"),
    ("ffn_norm.weight", "post_attention_layernorm.weight"),
    ("hc_attn_", "attn_hc."),
    ("hc_ffn_", "ffn_hc."),
    ("attn.", "self_attn."),
    ("ffn.gate.bias", "mlp.gate.e_score_correction_bias"),
    ("ffn.shared_experts.w1", "mlp.shared_experts.gate_proj"),
    ("ffn.shared_experts.w2", "mlp.shared_experts.down_proj"),
    ("ffn.shared_experts.w3", "mlp.shared_experts.up_proj"),
    ("ffn.", "mlp."),
)

_TOP_LEVEL = {
    "embed.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "head.weight": "lm_head.weight",
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_scale": "model.hc_head.hc_scale",
}

_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")


def map_to_reference(name: str) -> str | None:
    """Native checkpoint name -> ``transformers`` parameter name.

    Returns ``None`` for per-expert weights, which are not 1:1 and are stacked
    separately by :func:`build_reference_state_dict`.
    """
    if name in _TOP_LEVEL:
        return _TOP_LEVEL[name]
    if _EXPERT_RE.match(name):
        return None
    match = re.match(r"^layers\.(\d+)\.(.+)$", name)
    if not match:
        raise ValueError(f"unrecognised checkpoint tensor: {name!r}")
    layer, tail = match.group(1), match.group(2)
    for source, target in _LAYER_RULES:
        if source in tail:
            tail = tail.replace(source, target)
    if tail.endswith("shared_experts.gate_proj") or tail.endswith(
        ("shared_experts.down_proj", "shared_experts.up_proj")
    ):
        tail += ".weight"
    return f"model.layers.{layer}.{tail}"


def build_reference_state_dict(slice_dir: Path) -> dict[str, torch.Tensor]:
    """Read a native-named slice and return a ``transformers``-shaped state dict."""
    experts: dict[tuple[int, int, str], torch.Tensor] = {}
    state: dict[str, torch.Tensor] = {}

    with safe_open(str(Path(slice_dir) / "model.safetensors"), "pt") as handle:
        for name in handle.keys():
            expert = _EXPERT_RE.match(name)
            if expert:
                layer, index, which = (
                    int(expert.group(1)), int(expert.group(2)), expert.group(3)
                )
                experts[(layer, index, which)] = handle.get_tensor(name)
                continue
            state[map_to_reference(name)] = handle.get_tensor(name)

    layers = sorted({layer for layer, _, _ in experts})
    for layer in layers:
        indices = sorted({i for l, i, _ in experts if l == layer})
        # gate first then up, concatenated on the output dim: `_apply_gate`
        # chunks the F.linear result in two along the last axis.
        gate_up = torch.stack(
            [
                torch.cat([experts[(layer, i, "w1")], experts[(layer, i, "w3")]], dim=0)
                for i in indices
            ]
        )
        down = torch.stack([experts[(layer, i, "w2")] for i in indices])
        state[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = gate_up
        state[f"model.layers.{layer}.mlp.experts.down_proj"] = down
    return state


def load_reference_model(slice_dir: Path, dtype: torch.dtype = torch.bfloat16):
    """Instantiate ``DeepseekV4ForCausalLM`` on the slice and load real weights."""
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

    slice_dir = Path(slice_dir)
    config = DeepseekV4Config(**json.loads((slice_dir / "config.json").read_text()))
    model = DeepseekV4ForCausalLM(config).to(dtype).eval()
    state = build_reference_state_dict(slice_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # inv_freq buffers are derived, never stored -- anything else missing is a bug.
    real_missing = [n for n in missing if "inv_freq" not in n]
    if real_missing or unexpected:
        raise RuntimeError(
            f"state dict mismatch: missing={real_missing[:8]} unexpected={unexpected[:8]}"
        )
    return model, config
