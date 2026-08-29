# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 text decoder for the Neuron backend.

A 3:1 hybrid stack: three Gated DeltaNet layers carrying a recurrent state for
every one full-attention layer carrying a paged KV cache. Two cache groups are
declared, and every layer index belongs to exactly one of them.

  # >>> PARALLELISM: ... <<<   reusable parallelism code
  # <-- MODEL-SPECIFIC: ...    Qwen3.5-specific

Deliberate departures from ``vllm_neuron/model/qwen3``:

*The fused decode megakernel is not used.* ``NF.attention_decode`` fuses the QKV
projection, RoPE and ``o_proj`` into one call. Qwen3.5 needs partial RoPE
(rotating 64 of 256 channels, pairing i with i+32) which the kernel's
``head_dim // 2``-wide cos/sin cannot express, and an output gate applied
between attention and ``o_proj``, which the fused ``W_out`` leaves nowhere to
insert. Correctness first; fusing is a later, measured change.

*Segmented prefill is unavailable.* ``attention_segmented_cte`` raises above
head_dim 128 -- a head must fit in one SBUF partition -- and Qwen3.5's is 256.
``NF.flash_attention`` falls back to torch at that width rather than raising, so
prefill uses it and ``max_model_len`` is bounded by the single-shot limit.

*Every cache read is a fixed-size gather plus a mask.* No Python ``int`` derived
from ``cached_seq_len`` may size a slice: nine consecutive DeepSeek-V4 device
runs were spent removing exactly that
(docs/model-dev/deepseek-v4-024-device-validation.md).
"""

import logging

import torch
from torch import nn
from transformers import PretrainedConfig

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import CacheKind, KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from .attention import (
    Qwen3_5RMSNorm,
    Qwen3_5RotaryEmbedding,
    apply_output_gate,
    apply_partial_rotary_pos_emb,
    double_freqs,
    split_query_and_gate,
)
from .config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen3_5TextConfig
from .gated_deltanet import Qwen3_5GatedDeltaNet
from .parallel import resolve_sharding, resolve_tp_context
from .weight_loaders import (
    gated_o_proj_weight_loader,
    gated_qkv_weight_loader,
    needs_plus_one_fold,
    norm_plus_one_loader,
    plain_loader,
    text_weight_mappings,
)

logger = logging.getLogger(__name__)


def attention_layer_name(index: int) -> str:
    return f"layers.{index}.self_attn"


def linear_layer_name(index: int) -> str:
    return f"layers.{index}.linear_attn"


# ===========================================================================
# Dense MLP
# ===========================================================================


class Qwen3_5MLP(nn.Module):
    """SwiGLU MLP, present on every layer of both kinds.

    >>> PARALLELISM: intermediate dim sharded across TP; SP collectives. <<<
    """

    def __init__(self, config: Qwen3_5TextConfig, policy):
        super().__init__()
        tp = resolve_tp_context()
        self.tp_group = tp.group
        self.world_size = tp.world_size
        self.hidden_size = config.hidden_size
        self.intermediate_per_rank = policy.intermediate_per_rank

        dtype = config.torch_dtype
        self.gate_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.intermediate_per_rank, dtype=dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.intermediate_per_rank, dtype=dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_per_rank, config.hidden_size, dtype=dtype)
        )

        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)
        return output


# ===========================================================================
# Full attention
# ===========================================================================


class Qwen3_5Attention(nn.Module):
    """Gated GQA with per-head QK-norm and partial mRoPE.

    <-- MODEL-SPECIFIC: 24 Q / 4 KV heads at head_dim 256, an output gate
    interleaved per head in q_proj, and RoPE over only the leading 64 channels.
    """

    def __init__(self, config: Qwen3_5TextConfig, policy, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.dtype = config.torch_dtype
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.hidden_size = config.hidden_size
        self.scaling = config.head_dim**-0.5

        tp = resolve_tp_context()
        self.tp_group = tp.group
        self.world_size = tp.world_size

        self.q_heads_per_rank = policy.q_heads_per_rank
        self.kv_heads_per_rank = policy.kv_heads_per_rank
        self.num_kv_replicas = policy.num_kv_replicas
        self.num_kv_groups = self.q_heads_per_rank // self.kv_heads_per_rank

        # <-- MODEL-SPECIFIC: the query block is twice as wide as usual because
        # each head carries its gate alongside its query.
        self.q_gate_size = self.q_heads_per_rank * 2 * self.head_dim
        self.kv_size = self.kv_heads_per_rank * self.head_dim

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.q_gate_size + 2 * self.kv_size,
                dtype=self.dtype,
            )
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(
                self.q_heads_per_rank * self.head_dim,
                config.hidden_size,
                dtype=self.dtype,
            )
        )

        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        self.k_cache = None
        self.v_cache = None

        set_weight_loader(
            self.qkv_proj_weight, gated_qkv_weight_loader(config, policy)
        )
        set_weight_loader(
            self.o_proj_weight, gated_o_proj_weight_loader(config, policy)
        )

    # -- projection + norm + rope, shared by both paths --------------------

    def _project(self, hidden_states, cos, sin):
        tokens = hidden_states.shape[0]

        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)

        # Explicit slices: Tensor.split with a list of sizes is divergence #1.
        qg = qkv[..., : self.q_gate_size]
        k = qkv[..., self.q_gate_size : self.q_gate_size + self.kv_size]
        v = qkv[..., self.q_gate_size + self.kv_size :]

        query, gate = split_query_and_gate(qg, self.q_heads_per_rank, self.head_dim)
        gate = gate.reshape(tokens, -1)

        query = self.q_norm(query).transpose(0, 1)  # [Nq, T, D]
        key = self.k_norm(
            k.view(tokens, self.kv_heads_per_rank, self.head_dim)
        ).transpose(0, 1)
        value = v.view(tokens, self.kv_heads_per_rank, self.head_dim).transpose(0, 1)

        query = apply_partial_rotary_pos_emb(query, cos, sin, self.rotary_dim)
        key = apply_partial_rotary_pos_emb(key, cos, sin, self.rotary_dim)
        return query, key, value, gate

    def _write_cache(self, key, value, slot_mapping, block_size):
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        heads = self.kv_heads_per_rank

        k_flat = key.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_flat = value.reshape(-1, self.head_dim).to(self.v_cache.dtype)

        head_idx = torch.arange(
            heads, dtype=torch.long, device=key.device
        ).repeat_interleave(slot_mapping.shape[0])
        block_idx = block_indices.repeat(heads)
        pos_idx = position_indices.repeat(heads)

        self.k_cache.index_put_((block_idx, head_idx, pos_idx), k_flat)
        self.v_cache.index_put_((block_idx, head_idx, pos_idx), v_flat)

    def _gather_cache(self, block_table):
        """Fixed-size gather of the whole addressable context, then mask.

        Deliberately reads every block the table can address rather than
        ``cached_seq_len`` blocks: a Python-int length would be a data-dependent
        shape and Dynamo would refuse to trace it. Validity is applied as a mask
        downstream, never as a slice.
        """
        # k_cache: [num_blocks, heads, block_size, head_dim]
        gathered_k = self.k_cache[block_table]  # [B, NB, H, BS, D]
        gathered_v = self.v_cache[block_table]
        b, nb, h, bs, d = gathered_k.shape
        gathered_k = gathered_k.permute(0, 2, 1, 3, 4).reshape(b, h, nb * bs, d)
        gathered_v = gathered_v.permute(0, 2, 1, 3, 4).reshape(b, h, nb * bs, d)
        return gathered_k, gathered_v

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata):
        meta = attn_metadata[attention_layer_name(self.layer_idx)]
        is_decode = meta["max_query_len"] <= meta["decode_token_threshold"]

        cos, sin = position_embeddings
        cos, sin = double_freqs(cos, sin)

        if is_decode:
            return self.forward_decode(hidden_states, positions, cos, sin, meta)

        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(hidden_states, cos, sin, meta)

    def forward_prefill(self, hidden_states, cos, sin, meta):
        hidden_states = hidden_states.to(self.dtype)
        query, key, value, gate = self._project(hidden_states, cos, sin)

        self._write_cache(key, value, meta["slot_mapping"], meta["block_size"])

        # <-- MODEL-SPECIFIC: head_dim 256 makes segmented attention illegal;
        # flash falls back to torch at this width rather than raising.
        key_rep = key.repeat_interleave(self.num_kv_groups, dim=0)
        value_rep = value.repeat_interleave(self.num_kv_groups, dim=0)

        attn_output = NF.flash_attention(
            query.transpose(1, 2),
            key_rep.transpose(1, 2),
            value_rep,
            scale=self.scaling,
            tp_q=False,
            tp_out=True,
        )

        # <-- MODEL-SPECIFIC: with tp_out=True the kernel returns head-major
        # [Nh, D, T]. qwen3 hands that straight to NF.o_proj, which accepts it
        # as the [B, N, D, S] form -- but the output gate has to be applied
        # *before* o_proj and is token-major [T, Nh * D], so the layout has to
        # be converted here rather than deferred. A plain reshape would
        # silently interleave heads with positions.
        attn_output = attn_output.permute(2, 0, 1).reshape(
            hidden_states.shape[0], -1
        )
        attn_output = apply_output_gate(attn_output, gate)
        attn_output = NF.o_proj(
            attn_output.unsqueeze(0), self.o_proj_weight, None
        ).squeeze(0)

        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)
        return attn_output.contiguous()

    def forward_decode(self, hidden_states, positions, cos, sin, meta):
        hidden_states = hidden_states.to(self.dtype)
        block_table = meta["block_table_tensor"]
        batch = block_table.shape[0]
        tokens = hidden_states.shape[0]
        steps = tokens // batch

        query, key, value, gate = self._project(hidden_states, cos, sin)
        self._write_cache(key, value, meta["slot_mapping"], meta["block_size"])

        gathered_k, gathered_v = self._gather_cache(block_table)
        context = gathered_k.shape[2]

        # [Nq, T, D] -> [B, Nq, steps, D]
        q = query.transpose(0, 1).reshape(batch, steps, self.q_heads_per_rank, -1)
        q = q.transpose(1, 2)

        k = gathered_k.repeat_interleave(self.num_kv_groups, dim=1)
        v = gathered_v.repeat_interleave(self.num_kv_groups, dim=1)

        scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) * self.scaling

        # Mask by absolute position: a key slot is visible only at or before the
        # query's position. Built from tensors, never from a Python length.
        key_pos = torch.arange(context, device=scores.device).view(1, 1, 1, context)
        query_pos = positions.view(batch, 1, steps, 1)
        scores = scores.masked_fill(key_pos > query_pos, float("-inf"))

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        attn_output = torch.matmul(probs, v)  # [B, Nq, steps, D]

        attn_output = attn_output.transpose(1, 2).reshape(tokens, -1)
        attn_output = apply_output_gate(attn_output, gate)
        attn_output = NF.o_proj(
            attn_output.unsqueeze(0), self.o_proj_weight, None
        ).squeeze(0)

        if self.world_size > 1:
            self.tp_group.all_reduce(attn_output)
        return attn_output


# ===========================================================================
# Decoder layer
# ===========================================================================


class Qwen3_5DecoderLayer(nn.Module):
    """One layer of either kind. The MLP and the norms are identical for both."""

    def __init__(self, config: Qwen3_5TextConfig, policy, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

        if self.layer_type == FULL_ATTENTION:
            self.self_attn = Qwen3_5Attention(config, policy, layer_idx)
        else:
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx, policy)

        self.mlp = Qwen3_5MLP(config, policy)

    @property
    def is_linear(self) -> bool:
        return self.layer_type == LINEAR_ATTENTION

    def _is_decode(self, attn_metadata) -> bool:
        # Both kinds read the schedule from the full-attention group, which is
        # the one that always exists.
        for key, meta in attn_metadata.items():
            if key.endswith("self_attn"):
                return meta["max_query_len"] <= meta["decode_token_threshold"]
        raise KeyError("no full-attention metadata group present")

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata):
        is_decode = self._is_decode(attn_metadata)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.is_linear:
            # The state tensors are bound onto the module by bind_kv_cache.
            hidden_states = self.linear_attn.forward_paged(
                hidden_states, attn_metadata, is_decode=is_decode
            )
        else:
            hidden_states = self.self_attn(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        return residual + hidden_states


# ===========================================================================
# Backbone
# ===========================================================================


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, policy):
        super().__init__()
        self.config = config
        tp = resolve_tp_context()
        self.tp_group = tp.group
        self.world_size = tp.world_size
        self.rank = tp.rank

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=tp.device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(config, policy, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)

    def forward(
        self,
        input_ids,
        positions,
        attn_metadata=None,
        rank=None,
        inputs_embeds=None,
        is_token_ids=None,
        rotary_position_ids=None,
    ):
        first = next(k for k in attn_metadata if k.endswith("self_attn"))
        meta = attn_metadata[first]
        is_prefill = meta["max_query_len"] > meta["decode_token_threshold"]

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        if (
            is_prefill
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]

        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        # <-- MODEL-SPECIFIC: mRoPE. Text-only inputs carry the same value on
        # all three axes, so 1-D positions are a valid degenerate case.
        rope_positions = (
            rotary_position_ids if rotary_position_ids is not None else positions
        )
        position_embeddings = self.rotary_emb(
            rope_positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, positions, position_embeddings, attn_metadata
            )

        hidden_states = self.norm(hidden_states)

        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states, []


# ===========================================================================
# LM head, cache declaration, weight loading
# ===========================================================================


class Qwen3_5TextForCausalLM(nn.Module):
    """Qwen3.5-family text decoder with an LM head.

    ``requires_independent_kv_cache_tensors`` is set because the linear layers
    write their state functionally: Neuron's alias-output rewrite lets a later
    whole-view output clobber preceding state when raw byte pools are shared
    between cache groups
    (docs/model-dev/deepseek-v4-tiny-tp1-neuron-investigation.md).
    """

    requires_independent_kv_cache_tensors = True

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        tp = resolve_tp_context()
        self.tp_group = tp.group
        self.world_size = tp.world_size
        self.rank = tp.rank

        # Both layer kinds consume this: full attention pads its 24 query
        # heads up to a multiple of the degree, and the Gated DeltaNet shards
        # its 16 key / 48 value heads, falling back to splitting each value
        # head's dimension once the degree exceeds 16.
        self.policy = resolve_sharding(config, self.world_size)

        self.model = Qwen3_5TextModel(config, self.policy)

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        )

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=tp.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

        self._install_norm_loaders()

    def _install_norm_loaders(self):
        """Attach the ``+1`` fold to exactly the HF ``Qwen3_5RMSNorm`` weights."""
        for name, param in self.named_parameters():
            if name.endswith("linear_attn.norm.weight"):
                # Owned by Qwen3_5GatedDeltaNet, which installs a loader that
                # slices the weight to this rank's value width. Overwriting it
                # with plain_loader() here would hand every rank the full
                # 128-wide tensor and fail to load at tp=32.
                continue
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                loader = (
                    norm_plus_one_loader() if needs_plus_one_fold(name) else plain_loader()
                )
                set_weight_loader(param, loader)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig | Qwen3_5TextConfig, neuron_config: NeuronConfig
    ):
        config = (
            hf_config
            if isinstance(hf_config, Qwen3_5TextConfig)
            else Qwen3_5TextConfig.from_configs(hf_config, neuron_config)
        )
        return cls(config)

    # -- forward -----------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds=None,
        is_token_ids=None,
        attn_metadata=None,
        sampling_positions=None,
        sampling_params=None,
        spec_decode_metadata=None,
        logit_mask=None,
        rank=None,
        rotary_position_ids=None,
        **kwargs,
    ):
        positions = positions.to(torch.int32)

        hidden_states, _ = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
            rotary_position_ids=rotary_position_ids,
        )

        hidden_states = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        logits = self.lm_head(hidden_states)

        if self.on_device_sampling_config is None:
            return logits

        sampled = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        gathered = None
        if self._gather_logits and self.tp_group is not None:
            gathered = self.tp_group.all_gather(logits, dim=1)
        return sampled, gathered

    # -- mRoPE -------------------------------------------------------------

    def get_mrope_input_positions(self, input_tokens, mm_features):
        """Text-only mRoPE: all three axes carry the same position.

        vLLM decides ``uses_mrope`` from the presence of ``mrope_section`` in the
        config, which is true for every shipped Qwen3.5-family checkpoint even
        without a vision tower, so this must exist or ``_init_mrope_positions``
        raises.
        """
        length = len(input_tokens)
        positions = torch.arange(length, dtype=torch.int64)
        return positions[None, :].expand(3, length).contiguous(), 0

    # -- cache -------------------------------------------------------------

    def get_kv_spec(self) -> KVSpec:
        """Two groups: paged KV for full attention, per-request state for GDN."""
        config = self.config
        policy = self.policy
        layers = []

        for i, layer_type in enumerate(config.layer_types):
            if layer_type == FULL_ATTENTION:
                layers.append(
                    LayerSpec(
                        name=attention_layer_name(i),
                        num_kv_heads=policy.kv_heads_per_rank,
                        head_size=config.head_dim,
                        dtype=config.torch_dtype,
                    )
                )
            else:
                # policy.conv_dim_per_rank, never config.conv_dim // tp: the
                # query and key halves are replicated when the value dim is
                # split, so the naive form undersizes this cache at tp=32 and
                # is correct at every smaller degree.
                conv_shape = (
                    policy.conv_dim_per_rank,
                    config.linear_conv_kernel_dim - 1,
                )
                recurrent_shape = (
                    policy.v_heads_per_rank,
                    config.linear_key_head_dim,
                    policy.v_dim_per_rank,
                )
                layers.append(
                    LayerSpec(
                        name=linear_layer_name(i),
                        num_kv_heads=1,
                        head_size=1,
                        dtype=config.state_dtype,
                        cache_kind=CacheKind.MAMBA,
                        state_shapes=(conv_shape, recurrent_shape),
                        # One dtype for both states, never a mixed pair. The
                        # runner carves the conv window and the recurrent state
                        # from a single raw page as two strided views over one
                        # storage (neuron_model_runner.py:8546, mirroring
                        # vLLM's gpu_model_runner.py:7160), and the model
                        # mutates both in place to persist them. Two views over
                        # one storage with *different* dtypes, each mutated, is
                        # rejected while tracing: "aot_autograd() does not yet
                        # handle input mutations on views with different
                        # dtypes". Note this is a tracer limitation, not a
                        # layout one -- vLLM's own MambaSpec takes a per-state
                        # dtype tuple and kda_state_dtype ships a genuinely
                        # mixed pair, because its state writes happen inside
                        # custom kernels the tracer never enters.
                        #
                        # Casting cannot dodge it: the write must still land
                        # back in the view, so the condition survives any cast
                        # of the value. Either both states move up to fp32 or
                        # both move down to bf16; config.mamba_state_dtype
                        # picks which, and every consumer upcasts to fp32
                        # internally either way. Giving each state its own
                        # allocation would lift the restriction entirely, but
                        # that changes cache accounting and is left open.
                        state_dtypes=(config.state_dtype, config.state_dtype),
                    )
                )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        expected = {spec.name for spec in self.get_kv_spec().layers}
        missing = expected - set(kv_caches)
        if missing:
            raise KeyError(f"KV cache not initialized for: {sorted(missing)}")

        for i, layer in enumerate(self.model.layers):
            if layer.is_linear:
                conv_state, recurrent_state = kv_caches[linear_layer_name(i)]
                layer.linear_attn.conv_state_cache = conv_state
                layer.linear_attn.recurrent_state_cache = recurrent_state
            else:
                k_cache, v_cache = kv_caches[attention_layer_name(i)]
                layer.self_attn.k_cache = k_cache
                layer.self_attn.v_cache = v_cache

    # -- weights -----------------------------------------------------------

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        mappings = text_weight_mappings(self.config)

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            # The GDN gates and state stay fp32; everything else takes the
            # model dtype.
            if name.endswith(("A_log", "dt_bias")):
                continue
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
