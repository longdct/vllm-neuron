# SPDX-License-Identifier: Apache-2.0
"""Device-shaped DeepSeek-V4 model: batched forward, real paged cache I/O.

This is the Step 1-3 rewrite described in
``docs/model-dev/deepseek-v4-serving-roadmap.md``: the model's ``forward``
now takes the real vLLM Neuron runner contract (``attn_metadata``-driven,
batched, no Python token loop), ``bind_kv_cache`` attaches cache tensors to
the attention/compressor submodules instead of validating and storing a
dict, and the compressed-MLA/compressor-carry caches get real paged I/O.
See ``docs/model-dev/deepseek-v4-carry-cache-design.md`` for the carry-cache
addressing design and its cross-validation against vLLM's own upstream
DeepSeek-V4 GPU backend.

Scope deliberately held simplified for this pass (documented, not silent):

* Attention is now the real multi-head q_lora/kv_proj/partial-RoPE MLA
  architecture end to end, cross-validated (0.0 diff on the final output)
  against
  ``transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Attention``
  -- q_lora down/norm/up projection, a shared single-latent kv_proj+norm
  (K=V, broadcast to every query head -- there is no separate per-head K/V
  up-projection in the real architecture either), partial interleaved RoPE
  on the trailing ``qk_rope_head_dim`` channels (real
  ``DeepseekV4RotaryEmbedding``, reused directly rather than
  reimplemented), attention sinks, the real architecture's "K=V, so undo
  RoPE on the attended output at the query's own position" step, and the
  real grouped low-rank output projection (``o_a_proj``/``o_b_proj``,
  ``DeepseekV4GroupedLinear`` -- see that class's docstring), not a plain
  dense ``Linear`` approximation. The compressor's RMSNorm+RoPE
  finalization also uses real RoPE (``rope_layer_type="compress"``),
  matching the query side.
* Expert-parallel MoE is numerically correct at any ``ep_degree`` (each rank
  masks contributions to its own contiguous expert range and the group
  all-reduces the sum -- partitioned experts summed via all-reduce
  reconstructs the exact top-k sum), but does not use the all-to-all
  dispatch in ``vllm_neuron.parallel.all2all``: every rank still runs dense
  compute over all tokens rather than gathering only the tokens routed to
  its local experts. Fine for this pass's correctness goal; a follow-up for
  throughput. ``DeepseekV4Expert``'s per-expert FFN now matches the real
  ``DeepseekV4MLP``/``DeepseekV4Experts`` exactly: ``[out, in]``-layout
  ``gate_up_proj``/``down_proj`` driven through ``F.linear`` and a
  ``swiglu_limit``-clamped gate/up before the SiLU*up product, not the
  earlier unclamped ``[in, out]`` approximation.
* Real checkpoint loading/quantization and P7-P9 memory calibration are out
  of scope; ``load_weights`` still delegates to the existing, unchanged
  ``weight_loaders.py`` contract.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..kv_cache import CacheKind, KVSpec, LayerSpec
from .attention import (
    apply_partial_rotary,
    compressed_entry_slot_mapping,
    gather_paged_latent,
    mla_attention_reference,
    scatter_paged_latent,
)
from .compressor import (
    carry_gather_length,
    carry_replay_already_emitted,
    compress_csa_chunk,
    compress_hca_chunk,
    finalize_compressed_entries,
)
from .config import normalize_config
from .mhc import apply_hyperconnection, hyperconnection_reference
from .moe import hash_topk, routed_topk
from .tiny_model import (
    TinyDeepseekV4Config,
    TinyLayerConfig,
)
from .weight_loaders import ExpertDType, load_checkpoint_weights


def reference_config_from_hf(hf_config) -> TinyDeepseekV4Config:
    """Build a small-expert reference geometry from a validated HF config."""
    normalized = normalize_config(hf_config)
    hidden_size = int(hf_config.hidden_size)
    vocab_size = int(hf_config.vocab_size)
    if hidden_size < 1 or vocab_size < 1:
        raise ValueError("DeepSeek-V4 hidden and vocabulary sizes must be positive")
    return TinyDeepseekV4Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        latent_size=normalized.head_dim,
        sliding_window=normalized.sliding_window,
        num_experts=normalized.n_routed_experts,
        topk=normalized.num_experts_per_tok,
        hc_mult=normalized.hc_mult,
        hc_sinkhorn_iters=normalized.hc_sinkhorn_iters,
        rms_norm_eps=float(getattr(hf_config, "rms_norm_eps", 1e-6)),
        hc_eps=float(getattr(hf_config, "hc_eps", 1e-6)),
        swiglu_limit=float(getattr(hf_config, "swiglu_limit", 10.0)),
        layers=tuple(
            TinyLayerConfig(layer.compress_ratio, layer.mlp.value)
            for layer in normalized.layers
        ),
    )


def _decode_token_threshold(attn_metadata: dict, name: str) -> tuple[bool, dict]:
    entry = attn_metadata[name]
    is_decode = entry["max_query_len"] <= entry["decode_token_threshold"]
    return is_decode, entry


class DeepseekV4HyperConnection(nn.Module):
    """Thin owner of the mHC parameters; math lives in ``mhc.py``."""

    def __init__(self, config: TinyDeepseekV4Config):
        super().__init__()
        hc = config.hc_mult
        mix = (2 + hc) * hc
        self.fn = nn.Parameter(torch.randn(mix, hc * config.hidden_size) * 0.02)
        self.base = nn.Parameter(torch.zeros(mix))
        # Named hc_scale, not scale: weight_loaders.py::map_checkpoint_name
        # treats any parameter ending in ".scale" as an FP8 dequant scale
        # (renamed to ".weight_scale_inv") -- this is a plain learned mixing
        # weight, not a quantization scale, and the collision breaks
        # checkpoint loading if named "scale" (discovered via the E2E test
        # in test_deepseek_v4_device_e2e.py). tiny_model.py's
        # TinyHyperConnection keeps the original "scale" name -- it is never
        # loaded through weight_loaders.py, so the collision never surfaces
        # there.
        self.hc_scale = nn.Parameter(torch.ones(3))
        self.config = config

    def forward(self, streams: torch.Tensor):
        return hyperconnection_reference(
            streams,
            self.fn,
            self.base,
            self.hc_scale,
            norm_eps=self.config.rms_norm_eps,
            hc_eps=self.config.hc_eps,
            iterations=self.config.hc_sinkhorn_iters,
        )


class DeepseekV4HyperHead(nn.Module):
    def __init__(self, config: TinyDeepseekV4Config):
        super().__init__()
        hc = config.hc_mult
        self.fn = nn.Parameter(torch.randn(hc, hc * config.hidden_size) * 0.02)
        self.base = nn.Parameter(torch.zeros(hc))
        self.hc_scale = nn.Parameter(torch.ones(1))  # see HyperConnection's note
        self.config = config

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = streams.flatten(start_dim=-2).float()
        flat = flat * torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
        )
        weights = torch.sigmoid(
            F.linear(flat, self.fn.float()) * self.hc_scale + self.base
        ) + self.config.hc_eps
        return (weights.unsqueeze(-1) * streams).sum(dim=-2).to(streams.dtype)


class DeepseekV4Compressor(nn.Module):
    """Real gated HCA/CSA compressor with a paged carry-cache.

    Owns the ``fused_wkv_wgate`` projection (checkpoint-compatible with
    ``weight_loaders.py``'s ``.compressor.wkv``/``.compressor.wgate`` stacked
    shard) and the ``ape`` position bias. Reads/writes its carry against the
    bound ``state_cache`` tensor -- see ``carry_gather_length`` in
    ``compressor.py`` and ``docs/model-dev/deepseek-v4-carry-cache-design.md``
    for why replaying gathered raw rows through the stateless functions is
    equivalent to the incremental-state form.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        ratio: int,
        rms_norm_eps: float,
        *,
        rotary_emb,
        qk_rope_head_dim: int,
    ):
        super().__init__()
        if ratio not in (4, 128):
            raise ValueError(f"unsupported compressor ratio {ratio}")
        self.ratio = ratio
        self.overlap = ratio == 4
        self.coff = 1 + self.overlap
        self.head_dim = head_dim
        self.width = self.coff * head_dim
        self.rms_norm_eps = rms_norm_eps
        self.fused_wkv_wgate = nn.Linear(hidden_size, 2 * self.width, bias=False)
        self.ape = nn.Parameter(torch.randn(ratio, self.width) * 0.02)
        self.norm_weight = nn.Parameter(torch.ones(head_dim))
        # Shared with DeepseekV4Attention: one rotary_emb per model (matches
        # the real architecture's single model.rotary_emb), selected here via
        # rope_layer_type="compress" -- see DeepseekV4RotaryEmbedding.forward.
        self.rotary_emb = rotary_emb
        self.qk_rope_head_dim = qk_rope_head_dim
        # Bound by bind_kv_cache: raw [blocks, 1, slots, 2*width] carry cache.
        self.state_cache: torch.Tensor | None = None

    def _carry_rows(
        self, block_table_row: torch.Tensor, cached_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.state_cache is not None
        gather_n = carry_gather_length(
            cached_seq_len, self.ratio, needs_overlap=self.overlap
        )
        if gather_n == 0:
            empty = self.state_cache.new_zeros((1, 0, 2 * self.width))
            return empty[..., : self.width], empty[..., self.width :]
        # The state cache is itself a sliding-window group (window =
        # coff*ratio, matching get_kv_spec); the scheduler remaps blocks
        # older than that to a null block once evicted, same as
        # DeepseekV4Attention._swa_history. gather_n never exceeds coff*ratio
        # by construction (carry_gather_length), so this cap always covers it
        # while never walking into evicted/null territory.
        window = self.coff * self.ratio
        rows = gather_paged_latent(
            self.state_cache, block_table_row, min(cached_seq_len, window)
        )
        rows = rows.squeeze(1)[-gather_n:].unsqueeze(0)
        return rows[..., : self.width], rows[..., self.width :]

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        cached_seq_len: int,
        block_table_row: torch.Tensor,
        state_slot_mapping: torch.Tensor,
        mla_cache: torch.Tensor,
        mla_slot_mapping: torch.Tensor,
    ) -> None:
        """Compress ``hidden``'s new tokens and write results into caches.

        ``hidden`` is ``[tokens, hidden_size]`` for this one request's chunk.
        Writes new compressed entries into ``mla_cache`` (via
        ``mla_slot_mapping``, already restricted to this compressor's
        ``compress_ratio``) and the raw per-token kv/gate projection into
        ``self.state_cache`` (via ``state_slot_mapping``, a plain per-token
        sliding-window write). No return value -- this module only writes.
        """
        kv_gate = self.fused_wkv_wgate(hidden).unsqueeze(0)
        kv_new, gate_new = kv_gate[..., : self.width], kv_gate[..., self.width :]

        carry_kv, carry_gate = self._carry_rows(block_table_row, cached_seq_len)
        replay_kv = torch.cat((carry_kv, kv_new), dim=1)
        replay_gate = torch.cat((carry_gate, gate_new), dim=1)

        compress_fn = compress_csa_chunk if self.overlap else compress_hca_chunk
        compressed, _ = compress_fn(replay_kv, replay_gate, self.ape, None)
        drop = carry_replay_already_emitted(
            cached_seq_len, self.ratio, needs_overlap=self.overlap
        )
        compressed = compressed[:, drop:]

        # mla_slot_mapping carries one entry per *raw* token in this chunk
        # (-1 for both padding and non-window-completing positions -- see
        # compressed_entry_slot_mapping); `compressed`'s row count instead
        # reflects positional window-boundary arithmetic with no notion of
        # padding. Under this plugin's right-padding convention any window
        # touching padding is the trailing one(s), so filtering to the valid
        # (>= 0) slots and truncating both sides to the shorter length keeps
        # only genuinely-complete, non-padding windows -- dropping a
        # padding-corrupted trailing window rather than writing it.
        valid_slots = mla_slot_mapping[mla_slot_mapping >= 0]
        write_count = min(int(valid_slots.shape[0]), compressed.shape[1])
        compressed = compressed[:, :write_count]
        if compressed.shape[1]:
            # Real RoPE, matching the real architecture's compressor exactly
            # (rope_layer_type="compress"): window k's entry position is
            # first_window_position + k*ratio, where first_window_position is
            # the absolute raw-token position of the first NEW window this
            # call writes -- (cached_seq_len // ratio) windows were already
            # complete (and their positions already emitted) before this call.
            first_window_position = (cached_seq_len // self.ratio) * self.ratio
            entry_positions = (
                first_window_position
                + torch.arange(compressed.shape[1], device=compressed.device) * self.ratio
            ).unsqueeze(0)
            cos, sin = self.rotary_emb(
                compressed, position_ids=entry_positions, layer_type="compress"
            )
            finalized = finalize_compressed_entries(
                compressed, self.norm_weight, self.rms_norm_eps, cos, sin
            )
            entry_slots = valid_slots[:write_count]
            scatter_paged_latent(mla_cache, entry_slots, finalized.squeeze(0))

        # Raw per-token projection feeds the *next* chunk's carry replay.
        scatter_paged_latent(
            self.state_cache, state_slot_mapping, kv_gate.squeeze(0)
        )


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear -- the real architecture's output
    projection's first stage (``o_a_proj``).

    Splits the ``num_heads*head_dim``-wide attention output into
    ``n_groups`` independent chunks and projects each to a
    ``out_features // n_groups``-wide intermediate with its own weight
    block (all blocks packed into one ``[out_features, in_features_per_group]``
    parameter), rather than one huge dense projection -- the real
    architecture's perf optimization for very wide ``num_heads*head_dim``
    (e.g. V4-Flash: 32768). Ported directly from
    ``transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4GroupedLinear``
    (0.0 diff; see
    ``test_deepseek_v4_matches_real_architecture.py``'s
    ``test_output_projection_matches_real_module``).
    """

    def __init__(
        self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False
    ):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


class DeepseekV4Attention(nn.Module):
    """Real multi-head q_lora/kv_proj/partial-RoPE MLA attention.

    Cross-validated (0.0 diff pre-output-projection; see
    ``test_deepseek_v4_matches_real_architecture.py``) against
    ``transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Attention``.
    K=V in this architecture -- ``kv_proj``/``kv_norm`` produce one shared
    latent per token, broadcast to every query head via an identity
    "up-projection" (there is no learned per-head K/V matrix to begin with,
    real or otherwise), matching the real architecture exactly rather than
    approximating it. RoPE is real (``DeepseekV4RotaryEmbedding``, shared
    with the compressor), applied to the trailing ``qk_rope_head_dim``
    channels of q and kv, with the real architecture's "undo RoPE on the
    attended output at the query's own position" step afterward (needed
    because K=V means the attended output -- a weighted average of RoPE'd
    V -- inherits a position-dependent rotation that has to be removed
    before the output projection mixes heads). The output projection is now
    the real grouped low-rank ``o_a_proj``/``o_b_proj``
    (``DeepseekV4GroupedLinear``) too, not a plain dense ``Linear`` -- see
    that class's docstring.

    >>> PARALLELISM: TP <<<
    ``kv_proj``/``kv_norm``/``q_a_proj``/``q_a_norm`` stay fully replicated
    (the shared latent must be identical on every rank -- it feeds the
    compressed cache). ``q_b_proj``'s output and ``o_a_proj``'s input are
    head-sharded (``num_heads // world_size`` heads per rank, the standard
    MLA TP split); ``o_groups`` is sharded the same way
    (``o_groups // world_size`` groups per rank, each covering exactly the
    same ``num_heads*head_dim // o_groups``-wide input the unsharded real
    architecture would, since both numerator and denominator scale down by
    ``world_size`` together), with an all-reduce after ``o_b_proj`` (a
    row-parallel linear over the local group slice, same pattern as the
    existing dense ``o_proj`` all-reduce). Not exercised at ``world_size >
    1`` on real hardware -- see
    ``docs/model-dev/deepseek-v4-serving-roadmap.md``.
    """

    def __init__(
        self,
        config: TinyDeepseekV4Config,
        ratio: int,
        rms_norm_eps: float,
        *,
        hf_config,
        rotary_emb,
    ):
        super().__init__()
        self.ratio = ratio
        self.hidden_size = config.hidden_size
        self.head_dim = config.latent_size
        self.sliding_window = config.sliding_window
        self.qk_rope_head_dim = int(hf_config.qk_rope_head_dim)
        self.rope_layer_type = "main" if ratio == 0 else "compress"
        self.rotary_emb = rotary_emb
        num_heads = int(hf_config.num_attention_heads)
        q_lora_rank = int(hf_config.q_lora_rank)

        from vllm.distributed.parallel_state import get_tp_group

        try:
            self.tp_group = get_tp_group()
            self.world_size = self.tp_group.world_size
        except AssertionError:
            # No distributed process group set up (e.g. a module-level CPU
            # test that constructs this class directly rather than through
            # vLLM's engine, which always initializes one first). Degrades
            # to the same world_size=1 identity behavior a real single-rank
            # group would give.
            self.tp_group = None
            self.world_size = 1
        if num_heads % self.world_size:
            raise ValueError(
                f"num_attention_heads={num_heads} must be divisible by TP "
                f"world_size={self.world_size}"
            )
        self.heads_per_rank = num_heads // self.world_size

        self.q_a_proj = nn.Linear(config.hidden_size, q_lora_rank, bias=False)
        self.q_a_norm = nn.RMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = nn.Linear(
            q_lora_rank, self.heads_per_rank * self.head_dim, bias=False
        )
        # q_b_norm is unweighted (transformers.DeepseekV4UnweightedRMSNorm --
        # variance normalization, no learned scale) and applied inline in
        # _forward_one_token; no parameters to own here.
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)

        o_groups = int(getattr(hf_config, "o_groups", 8))
        o_lora_rank = int(getattr(hf_config, "o_lora_rank", 1024))
        if o_groups % self.world_size:
            raise ValueError(
                f"o_groups={o_groups} must be divisible by TP "
                f"world_size={self.world_size}"
            )
        self.o_groups = o_groups // self.world_size
        local_width = self.heads_per_rank * self.head_dim
        if local_width % self.o_groups:
            raise ValueError(
                f"heads_per_rank*head_dim={local_width} must be divisible by "
                f"o_groups (post-TP-shard)={self.o_groups}"
            )
        self.o_a_proj = DeepseekV4GroupedLinear(
            local_width // self.o_groups, self.o_groups * o_lora_rank, self.o_groups
        )
        self.o_b_proj = nn.Linear(
            self.o_groups * o_lora_rank, config.hidden_size, bias=False
        )
        self.sinks = nn.Parameter(torch.zeros(self.heads_per_rank))
        # K=V broadcast to every head: mla_attention_reference projects a
        # shared latent to per-head K/V via key_weight/value_weight matrices,
        # designed for a *learned* up-projection. The real architecture has
        # none -- an identity matrix per head reproduces the real "same
        # latent, every head" broadcast exactly (validated directly; see the
        # class docstring) without needing a second oracle function.
        self.register_buffer(
            "identity_kv_weight",
            torch.eye(self.head_dim).unsqueeze(0).expand(self.heads_per_rank, -1, -1),
            persistent=False,
        )

        self.compressor = (
            DeepseekV4Compressor(
                config.hidden_size,
                self.head_dim,
                ratio,
                rms_norm_eps,
                rotary_emb=rotary_emb,
                qk_rope_head_dim=self.qk_rope_head_dim,
            )
            if ratio
            else None
        )

        # Bound by bind_kv_cache: raw [blocks, 1, slots, latent] tensors.
        self.swa_cache: torch.Tensor | None = None
        self.mla_cache: torch.Tensor | None = None

    def _swa_history(
        self, block_table_row: torch.Tensor, cached_seq_len: int
    ) -> torch.Tensor:
        """Prior uncompressed latents, capped at the sliding window.

        The SWA cache is a true sliding-window group: the scheduler remaps
        blocks older than the window to a null block once the window has
        rolled past them (T1's
        ``test_sliding_window_remapping_uses_null_blocks_but_latents_remain_stable``
        covers this). Gathering the raw ``cached_seq_len`` count would walk
        into those remapped/invalid blocks once a request outgrows the
        window, so the gather length is capped here, mirroring the same cap
        the scheduler itself applies to the block table.
        """
        if cached_seq_len == 0:
            return self.swa_cache.new_zeros((0, self.swa_cache.shape[-1]))
        gather_len = min(cached_seq_len, self.sliding_window)
        return gather_paged_latent(
            self.swa_cache, block_table_row, gather_len
        ).squeeze(1)

    def _compressed_history(
        self, block_table_row: torch.Tensor, cached_seq_len: int
    ) -> torch.Tensor:
        """Prior compressed entries.

        This group's cache stores one physical row per *compressed entry*,
        not per raw token (``storage_block_size = block_size //
        compress_ratio`` -- ``kv_spec_conversion.py``), while
        ``cached_seq_len`` is raw-token-scaled like every other group (see
        ``compressed_entry_slot_mapping``). The two address spaces differ by
        exactly ``compress_ratio``, matching the write side's
        ``raw_slot // compress_ratio``.
        """
        num_entries = cached_seq_len // self.ratio
        if num_entries == 0:
            return self.mla_cache.new_zeros((0, self.mla_cache.shape[-1]))
        return gather_paged_latent(
            self.mla_cache, block_table_row, num_entries
        ).squeeze(1)

    def _forward_one_token(
        self,
        hidden: torch.Tensor,
        *,
        self_attn_name: str,
        attn_metadata: dict,
        request: int,
        local_index: int,
        cached_seq_len: int,
    ) -> torch.Tensor:
        """Attend and update caches for exactly one token.

        Called once per token, even during prefill: a compressed window
        completed by an earlier token in the *same* chunk must be visible to
        later queries in that chunk, and a flat batched history built once
        for the whole chunk cannot express that (compression boundaries are
        positional, not chunk-aligned). Reusing the decode-shaped per-token
        path for prefill too keeps exactly one code path to validate against
        the oracle, at a real throughput cost documented in the module
        docstring -- correctness, not performance, is this pass's bar.
        """
        swa_entry = attn_metadata[f"{self_attn_name}.swa_cache"]
        swa_block_table = swa_entry["block_table_tensor"][request]
        swa_slot = swa_entry["slot_mapping"][local_index : local_index + 1]

        # cos/sin at this token's own absolute position -- shared by q's
        # forward rotation, kv's forward rotation (baked into what gets
        # cached), and the attended output's inverse rotation at the end
        # (real architecture's "K=V, so undo RoPE on the output" step; see
        # the class docstring).
        position_ids = hidden.new_tensor([[cached_seq_len]], dtype=torch.long)
        cos, sin = self.rotary_emb(hidden, position_ids=position_ids, layer_type=self.rope_layer_type)

        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(hidden)))
        q = q.view(1, 1, self.heads_per_rank, self.head_dim)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.q_a_norm.eps).to(q.dtype)
        cos_h, sin_h = cos.unsqueeze(2), sin.unsqueeze(2)  # broadcast over heads
        q_roped = apply_partial_rotary(q, cos_h, sin_h, rope_dim=self.qk_rope_head_dim)

        kv = self.kv_norm(self.kv_proj(hidden))  # [1, head_dim]
        kv_roped = apply_partial_rotary(
            kv.unsqueeze(0), cos, sin, rope_dim=self.qk_rope_head_dim
        ).squeeze(0)
        scatter_paged_latent(self.swa_cache, swa_slot, kv_roped)

        swa_history = self._swa_history(swa_block_table, cached_seq_len)
        history = torch.cat((swa_history, kv_roped), dim=0)
        if len(history) > self.sliding_window:
            history = history[-self.sliding_window :]

        if self.compressor is not None:
            mla_entry = attn_metadata[self_attn_name]
            state_entry = attn_metadata[f"{self_attn_name}.compressor.state_cache"]
            mla_slot = compressed_entry_slot_mapping(
                mla_entry["slot_mapping"][local_index : local_index + 1], self.ratio
            )
            self.compressor(
                hidden,
                cached_seq_len=cached_seq_len,
                block_table_row=state_entry["block_table_tensor"][request],
                state_slot_mapping=state_entry["slot_mapping"][
                    local_index : local_index + 1
                ],
                mla_cache=self.mla_cache,
                mla_slot_mapping=mla_slot,
            )
            compressed_history = self._compressed_history(
                mla_entry["block_table_tensor"][request], cached_seq_len
            )
            history = torch.cat((compressed_history, history), dim=0)

        attended = mla_attention_reference(
            q_roped,
            history.view(1, -1, history.shape[-1]),
            self.identity_kv_weight,
            self.identity_kv_weight,
            attention_sinks=self.sinks,
            sliding_window=self.sliding_window if self.ratio == 0 else None,
        )  # [1, 1, heads_per_rank, head_dim]
        attended = apply_partial_rotary(
            attended, cos_h, sin_h, rope_dim=self.qk_rope_head_dim, inverse=True
        )
        return attended.reshape(1, self.heads_per_rank * self.head_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        self_attn_name: str,
        attn_metadata: dict,
    ) -> torch.Tensor:
        """``hidden`` is ``[tokens, hidden_size]`` for one request's chunk."""
        request = 0  # see module docstring: one request per forward call.
        base_cached_seq_len = int(
            attn_metadata[f"{self_attn_name}.swa_cache"]["cached_seq_len"][request][0]
        )
        attended_rows = []
        for local_index in range(hidden.shape[0]):
            attended_rows.append(
                self._forward_one_token(
                    hidden[local_index : local_index + 1],
                    self_attn_name=self_attn_name,
                    attn_metadata=attn_metadata,
                    request=request,
                    local_index=local_index,
                    cached_seq_len=base_cached_seq_len + local_index,
                )
            )
        attended = torch.cat(attended_rows, dim=0)

        grouped = attended.reshape(attended.shape[0], self.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(1)
        out = self.o_b_proj(grouped)
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out


class DeepseekV4Expert(nn.Module):
    """One dense SwiGLU expert (shared expert, or one routed-expert row).

    Matches the real architecture's ``DeepseekV4MLP``/``DeepseekV4Experts``
    exactly: ``[out, in]``-layout weights driven through ``F.linear`` (not
    ``[in, out]`` driven through plain ``@``), and gate/up clamped to
    ``swiglu_limit`` before the SiLU*up product -- an unclamped SwiGLU is a
    real numerical divergence for any input large enough to matter, not just
    a cosmetic layout difference. See
    ``test_deepseek_v4_matches_real_architecture.py``'s
    ``test_expert_wrapper_matches_real_module``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2 * intermediate_size, hidden_size) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(hidden_size, intermediate_size) * 0.02)
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(hidden, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.linear(F.silu(gate) * up, self.down_proj)


class DeepseekV4MoE(nn.Module):
    """Routed + shared MoE with dense-compute, all-reduced expert parallelism.

    >>> PARALLELISM: EP <<<
    Routing (gate, hash table, correction bias) is fully replicated -- small
    and must be identical on every rank. Each rank computes the FFN only for
    tokens whose selected expert falls in its own contiguous local range,
    masking everything else to zero, then the EP group all-reduces the sum.
    Because experts are partitioned (non-overlapping) across ranks, summing
    each rank's local-only contribution reconstructs the exact top-k output
    -- numerically correct at any ``ep_degree``, unlike a scheme that would
    need real all-to-all token dispatch (``vllm_neuron.parallel.all2all``,
    not wired here -- see module docstring) to be *efficient*.
    """

    def __init__(self, config: TinyDeepseekV4Config, kind: str):
        super().__init__()
        self.kind = kind
        self.topk = config.topk
        self.num_experts = config.num_experts
        intermediate = config.hidden_size * 2

        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_degree,
            get_neuron_ep_rank,
        )

        self.ep_degree = get_neuron_ep_degree()
        if self.num_experts % self.ep_degree:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by "
                f"ep_degree={self.ep_degree}"
            )
        self.num_local_experts = self.num_experts // self.ep_degree
        ep_rank = get_neuron_ep_rank()
        self.local_start = ep_rank * self.num_local_experts

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                DeepseekV4Expert(config.hidden_size, intermediate, config.swiglu_limit)
                for _ in range(self.num_local_experts)
            ]
        )
        self.shared_experts = DeepseekV4Expert(
            config.hidden_size, intermediate, config.swiglu_limit
        )
        self.correction_bias = nn.Parameter(torch.zeros(config.num_experts))
        self._vocab_size = config.vocab_size
        self.register_buffer(
            "tid2eid",
            torch.stack(
                [
                    (torch.arange(config.vocab_size) + slot).remainder(
                        config.num_experts
                    )
                    for slot in range(config.topk)
                ],
                dim=1,
            ),
        )

    def reinitialize_deterministic_buffers(self) -> None:
        """Recompute ``tid2eid`` in place, e.g. after ``to_empty()``.

        ``tid2eid`` is derived purely from config (no checkpoint provides
        it), but vLLM constructs models on the meta device -- the
        ``arange``/``remainder`` that built it in ``__init__`` never actually
        ran, it only recorded shape/dtype. ``to_empty()`` gives it real,
        uninitialized storage; ``load_weights`` calls this afterward to fill
        it with the real values (see the class docstring's EP note -- this
        buffer must be identical across ranks).
        """
        device = self.tid2eid.device
        self.tid2eid.copy_(
            torch.stack(
                [
                    (torch.arange(self._vocab_size, device=device) + slot).remainder(
                        self.num_experts
                    )
                    for slot in range(self.topk)
                ],
                dim=1,
            )
        )

    def forward(self, hidden: torch.Tensor, token_id: torch.Tensor) -> torch.Tensor:
        logits = self.gate(hidden)
        if self.kind == "hash_moe":
            ids, weights = hash_topk(logits, token_id, self.tid2eid)
        else:
            ids, weights = routed_topk(logits, self.correction_bias, self.topk)

        routed = torch.zeros_like(hidden)
        for slot in range(self.topk):
            eid = ids[:, slot]
            local_idx = eid - self.local_start
            is_local = (local_idx >= 0) & (local_idx < self.num_local_experts)
            safe_idx = local_idx.clamp(0, self.num_local_experts - 1)
            for local_expert in range(self.num_local_experts):
                mask = is_local & (safe_idx == local_expert)
                # Always compute, even for an expert no token in this batch
                # routes to (mask all-False contributes exactly zero either
                # way) -- the previous `if not bool(mask.any()): continue`
                # skip was a real optimization but `bool(tensor)` is a
                # device->host sync and an unconditional Dynamo graph break
                # ("Data-dependent branching"), found compiling the full
                # device-shaped model on real Trn2 silicon (see
                # docs/model-dev/deepseek-v4-024-device-validation.md).
                # Dense per-token dispatch is already this pass's documented
                # throughput tradeoff (see this class's docstring); this
                # just makes the per-*expert* skip the same tradeoff.
                contribution = self.experts[local_expert](hidden)
                routed = routed + contribution * (
                    weights[:, slot : slot + 1] * mask.unsqueeze(-1)
                )

        if self.ep_degree > 1:
            from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_group

            routed = get_neuron_ep_group().all_reduce(routed)

        return routed + self.shared_experts(hidden)


class DeepseekV4DecoderLayer(nn.Module):
    """
    Confirmed by direct comparison against ``transformers``'s real
    ``DeepseekV4DecoderLayer.forward`` (see
    ``docs/model-dev/deepseek-v4-carry-cache-design.md``'s testing note): the
    real layer applies a plain RMSNorm to the mHC-collapsed hidden state
    before *both* attention and the MoE block
    (``self.input_layernorm``/``self.post_attention_layernorm``), which an
    earlier version of this class omitted entirely. attn_hc/ffn_hc, the MoE
    gate/routing, and the compressor's projection+reduction+norm all matched
    the real architecture exactly (0.0 diff) once driven by the same
    weights; this norm was the one real structural gap that comparison
    found, and it is now applied here.
    """

    def __init__(
        self,
        config: TinyDeepseekV4Config,
        layer: TinyLayerConfig,
        *,
        hf_config,
        rotary_emb,
    ):
        super().__init__()
        self.ratio = layer.compress_ratio
        self.attention = DeepseekV4Attention(
            config,
            layer.compress_ratio,
            config.rms_norm_eps,
            hf_config=hf_config,
            rotary_emb=rotary_emb,
        )
        self.moe = DeepseekV4MoE(config, layer.mlp_kind)
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        streams: torch.Tensor,
        token_id: torch.Tensor,
        *,
        self_attn_name: str,
        attn_metadata: dict,
    ) -> torch.Tensor:
        post, comb, hidden = self.attn_hc(streams)
        attended = self.attention(
            self.input_layernorm(hidden),
            self_attn_name=self_attn_name,
            attn_metadata=attn_metadata,
        )
        streams = apply_hyperconnection(streams, attended, post, comb)
        post, comb, hidden = self.ffn_hc(streams)
        moe_out = self.moe(self.post_attention_layernorm(hidden), token_id)
        streams = apply_hyperconnection(streams, moe_out, post, comb)
        return streams


class DeepseekV4Model(nn.Module):
    """Decoder body with checkpoint-compatible top-level parameter names."""

    def __init__(self, config: TinyDeepseekV4Config, hf_config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # One rotary_emb shared by every layer's attention and compressor,
        # matching the real architecture's single model.rotary_emb -- it
        # holds both the "main" (sliding-window layers) and "compress"
        # (compressed-attention layers and their compressors) inv_freq
        # tables internally, selected per call via layer_type.
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4RotaryEmbedding,
        )

        rotary_emb = DeepseekV4RotaryEmbedding(hf_config)
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    config, layer, hf_config=hf_config, rotary_emb=rotary_emb
                )
                for layer in config.layers
            ]
        )
        # Named hc_head, matching the real transformers.DeepseekV4Model
        # attribute (confirmed by direct comparison) and
        # weight_loaders.py's existing "hc_head" checkpoint-prefix rule in
        # map_checkpoint_name.
        self.hc_head = DeepseekV4HyperHead(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_metadata: dict,
    ) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError("DeepSeek-V4 expects a flat [tokens] input_ids tensor")
        hidden = self.embed_tokens(input_ids)
        streams = hidden.unsqueeze(-2).expand(-1, self.config.hc_mult, -1)
        for index, layer in enumerate(self.layers):
            self_attn_name = f"model.layers.{index}.self_attn"
            streams = layer(
                streams,
                input_ids,
                self_attn_name=self_attn_name,
                attn_metadata=attn_metadata,
            )
        return self.norm(self.hc_head(streams))


class DeepseekV4ForCausalLM(nn.Module):
    """Device-shaped DeepSeek-V4: batched, ``attn_metadata``-driven forward."""

    is_text_generation_model = True

    def __init__(self, config: TinyDeepseekV4Config, hf_config):
        super().__init__()
        self.config = config
        self.model = DeepseekV4Model(config, hf_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None):
        del neuron_config
        return cls(reference_config_from_hf(hf_config), hf_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def get_kv_spec(self) -> KVSpec:
        """Declare the three engine-owned cache layouts used by V4 layers.

        All of these are single-vector MLA layouts. Every layer owns an
        uncompressed sliding window; compressed layers additionally own their
        long-context latent pages and the compressor's fp32 carry pages.
        """
        specs: list[LayerSpec] = []
        for index, layer in enumerate(self.config.layers):
            prefix = f"model.layers.{index}.self_attn"
            specs.append(
                LayerSpec(
                    name=f"{prefix}.swa_cache",
                    num_kv_heads=1,
                    head_size=self.config.latent_size,
                    dtype=torch.bfloat16,
                    cache_kind=CacheKind.SLIDING_WINDOW_MLA,
                    sliding_window_size=self.config.sliding_window,
                    alignment=128,
                )
            )
            if layer.compress_ratio == 0:
                continue
            ratio = layer.compress_ratio
            specs.append(
                LayerSpec(
                    name=prefix,
                    num_kv_heads=1,
                    head_size=self.config.latent_size,
                    dtype=torch.bfloat16,
                    cache_kind=CacheKind.MLA,
                    compress_ratio=ratio,
                    # The compressed-entry addressing (attention.py::
                    # compressed_entry_slot_mapping) relies on this group's
                    # page block_size being a multiple of every supported
                    # compress_ratio (4 and 128); the Neuron platform default
                    # of 32 is not. Set explicitly rather than depend on the
                    # caller passing --block-size 128 -- see
                    # vllm_neuron/vllm/platform.py::register_custom_kv_cache_specs,
                    # whose required MLAAttentionSpec fixtures use exactly 128.
                    block_size=128,
                    alignment=128,
                )
            )
            overlap = ratio == 4
            specs.append(
                LayerSpec(
                    name=f"{prefix}.compressor.state_cache",
                    num_kv_heads=1,
                    head_size=2 * (1 + overlap) * self.config.latent_size,
                    dtype=torch.float32,
                    block_size=4 if overlap else 8,
                    cache_kind=CacheKind.COMPRESSOR_STATE,
                    sliding_window_size=(1 + overlap) * ratio,
                    alignment=512,
                )
            )
        return KVSpec(specs)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        """Attach externally-allocated cache tensors to their owning modules.

        Mirrors ``llama3/model.py``'s ``bind_kv_cache`` (attaches onto the
        attention module rather than storing a validate-only dict), adapted
        for DeepSeek-V4's three cache groups per compressed layer.
        """
        expected = {spec.name for spec in self.get_kv_spec().layers}
        actual = set(kv_caches)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"DeepSeek-V4 KV cache keys do not match: missing={missing}, "
                f"unexpected={unexpected}"
            )
        wrong_arity = sorted(
            name for name, tensors in kv_caches.items() if len(tensors) != 1
        )
        if wrong_arity:
            raise ValueError(
                "DeepSeek-V4 caches use one latent tensor; invalid layers: "
                f"{wrong_arity}"
            )
        for index, layer in enumerate(self.model.layers):
            prefix = f"model.layers.{index}.self_attn"
            layer.attention.swa_cache = kv_caches[f"{prefix}.swa_cache"][0]
            if layer.attention.compressor is not None:
                layer.attention.mla_cache = kv_caches[prefix][0]
                layer.attention.compressor.state_cache = kv_caches[
                    f"{prefix}.compressor.state_cache"
                ][0]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        sampling_positions: torch.Tensor,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions, sampling_params, spec_decode_metadata, logit_mask, rank
        del inputs_embeds, is_token_ids
        hidden = self.model(input_ids, attn_metadata)
        selected = torch.index_select(hidden, 0, sampling_positions)
        return self.compute_logits(selected)

    def load_weights(
        self,
        checkpoint_path: str,
        device,
        cache_dir: str | None,
        *,
        expert_dtype: ExpertDType = "bf16",
    ) -> None:
        """Load a checkpoint directory's safetensors weights.

        Matches the runner's real call signature (``neuron_model_runner.py``
        calls ``self.model.load_weights(model, device, download_dir)``
        directly -- there is no vLLM-generic loader indirection to hook into
        here, unlike the standard ``DefaultModelLoader`` path other backends
        use, so ``load_format="dummy"`` is not honored automatically). When
        the checkpoint directory has no ``.safetensors`` files (as in a
        dummy-load smoke test with only a ``config.json``), this leaves the
        randomly-initialized parameters as-is rather than erroring -- the
        real-weight path below is exercised whenever files are present.
        """
        import glob
        import os

        from safetensors import safe_open

        search_root = checkpoint_path
        if not os.path.isdir(search_root):
            from vllm_neuron.utils.checkpoints import _get_checkpoint_source

            source = _get_checkpoint_source(checkpoint_path, ".safetensors", cache_dir)
            for name in source.get_file_names():
                source.download_file(name)
            search_root = os.path.dirname(source.get_file_path(source.get_file_names()[0]))

        files = sorted(glob.glob(os.path.join(search_root, "*.safetensors")))

        # vLLM constructs the model on the meta device before load_weights
        # runs, and the runner moves it onto `device` right afterward
        # regardless of whether a checkpoint was found -- to_empty() must
        # run unconditionally, or that later .to(device) hits the same
        # "cannot copy out of meta tensor" error load_weights exists to
        # avoid. Deterministic buffers (moe routing tables, never provided
        # by a checkpoint) are recomputed explicitly since nothing else
        # touches them; real parameters are left as to_empty()'s
        # uninitialized values when no checkpoint is present (a smoke-test
        # scenario only -- see the docstring above).
        self.to_empty(device=device)
        for module in self.modules():
            if isinstance(module, DeepseekV4MoE):
                module.reinitialize_deterministic_buffers()
        if not files:
            return

        def _weights():
            for path in files:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        yield name, handle.get_tensor(name)

        load_checkpoint_weights(self, _weights(), expert_dtype=expert_dtype)
