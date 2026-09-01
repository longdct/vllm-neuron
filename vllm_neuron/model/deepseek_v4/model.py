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

import logging
import os
import re

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_neuron.utils.neuron_utils import can_run_kernel

from ..kv_cache import CacheKind, KVSpec, LayerSpec
from .attention import (
    SharedLatentMLAInputs,
    apply_partial_rotary,
    compressed_entry_slot_mapping,
    gather_recent_window,
    gather_recent_window_batched,
    logical_to_physical_slots_batched,
    mla_attention_reference,
    read_compressed_history,
    read_compressed_history_batched,
    recent_compressed_logical_indices,
    recent_sliding_logical_indices,
    scatter_paged_latent,
    visible_compressed_entries,
)
from .compressor import (
    carry_gather_length_tensor,
    compress_csa_chunk,
    compress_hca_chunk,
    finalize_compressed_entries,
)
from .config import DeepseekV4ModelConfig
from .indexer import (
    fixed_prefix_compressed_entries,
    lightning_index_scores,
    select_compressed_entries,
    selection_mask_from_indices,
    streaming_topk_compressed_entries,
)
from .mhc import apply_hyperconnection, hyperconnection_reference
from .moe import dense_expert_affinities, hash_topk, routed_topk
from .nki_compressor import paged_gated_compressor
from .nki_indexer import paged_projected_bf16_indexer
from .nki_mla import _HCA_COUNT_BUCKETS, paged_shared_latent_mla
from .parallel import (
    resolve_output_projection_partition,
    resolve_parallel_topology,
)
from .weight_loaders import ExpertDType, load_checkpoint_weights


logger = logging.getLogger(__name__)


def _decode_token_threshold(attn_metadata: dict, name: str) -> tuple[bool, dict]:
    entry = attn_metadata[name]
    is_decode = entry["max_query_len"] <= entry["decode_token_threshold"]
    return is_decode, entry


class DeepseekV4RMSNorm(nn.Module):
    """RMSNorm that computes its variance in FP32, like the real architecture.

    ``torch.nn.RMSNorm`` reduces in the input dtype. Under BF16 that is a
    materially different computation from
    ``transformers.models.deepseek_v4.DeepseekV4RMSNorm``, which upcasts to FP32
    for the mean-square and rsqrt and only returns to the input dtype to apply
    the weight.

    The gap is not cosmetic. Driving both forms with the same real ``kv_proj``
    output measures ``max|diff| = 1.56e-2`` on a 512-wide vector -- which is
    essentially the entire divergence observed at ``kv_norm`` when comparing this
    plugin against the reference on identical weights, and it then compounds
    through attention into the logits.

    Applying ``weight`` *after* the downcast (rather than in FP32) is also part of
    the contract: it is what the reference does, and doing it in FP32 leaves a
    residual difference.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        promoted = hidden_states.to(torch.float32)
        variance = promoted.pow(2).mean(-1, keepdim=True)
        promoted = promoted * torch.rsqrt(variance + self.eps)
        return self.weight * promoted.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class DeepseekV4HyperConnection(nn.Module):
    """Thin owner of the mHC parameters; math lives in ``mhc.py``."""

    def __init__(self, config: DeepseekV4ModelConfig):
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
    def __init__(self, config: DeepseekV4ModelConfig):
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
        weights = (
            torch.sigmoid(F.linear(flat, self.fn.float()) * self.hc_scale + self.base)
            + self.config.hc_eps
        )
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
        self, block_table_row: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fixed-size carry window ending one token before ``position_ids``.

        Dynamo-shape-static, mirroring ``DeepseekV4Attention._swa_history``'s
        ``gather_recent_window``-based redesign: always gathers exactly
        ``coff*ratio - 1`` rows (a compile-time-constant Python int, never a
        traced value) via ``gather_recent_window`` -- one less than the full
        carry window, since the new token itself (not yet scattered into
        ``self.state_cache`` at the point this runs) is appended separately
        as the replay's final row by the caller. Returns
        ``(carry_kv, carry_gate, carry_valid)`` where ``carry_valid`` is
        ``[coff*ratio]`` long: the gathered rows' own existence mask
        (``gather_recent_window``'s ``exists``, for rows before generation
        produced them) ANDed with an "unconsumed" mask built from
        ``carry_gather_length_tensor`` (rows already consumed/emitted by an
        earlier call must not be replayed a second time -- same accounting
        as the old Python-int ``gather_n``, now tensor-valued), plus a
        trailing ``True`` for the new token's own always-valid slot.
        Invalid rows are neutralized downstream via
        ``compress_hca_chunk``/``compress_csa_chunk``'s gate-softmax masking
        rather than sliced away -- no more data-dependent-length slice (see
        docs/model-dev/deepseek-v4-swa-null-block-bug.md item 3).

        The state cache is itself a sliding-window group (window =
        coff*ratio, matching get_kv_spec); like ``_swa_history``,
        ``gather_recent_window`` reads columns covering exactly
        ``[position_ids - coff*ratio, position_ids - 1]`` (the live window),
        never a null-remapped column, regardless of how much eviction has
        happened.
        """
        assert self.state_cache is not None
        carry_window = self.coff * self.ratio - 1
        gathered, exists = gather_recent_window(
            self.state_cache, block_table_row, carry_window, position_ids - 1
        )
        gathered = gathered.squeeze(1).unsqueeze(0)  # [1, carry_window, 2*width]
        cached_seq_len = position_ids.view(()).long()
        gather_n = carry_gather_length_tensor(
            cached_seq_len, self.ratio, needs_overlap=self.overlap
        )
        idx = torch.arange(carry_window, device=gathered.device)
        carry_valid = exists & (idx >= (carry_window - gather_n))
        full_valid = torch.cat((carry_valid, carry_valid.new_ones(1)))
        return gathered[..., : self.width], gathered[..., self.width :], full_valid

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        block_table_row: torch.Tensor,
        state_slot_mapping: torch.Tensor,
        mla_cache: torch.Tensor,
        mla_slot_mapping: torch.Tensor,
    ) -> None:
        """Compress ``hidden``'s one new token and write results into caches.

        ``hidden`` is ``[1, hidden_size]`` -- this module's only call site
        (``DeepseekV4Attention._forward_one_token``) is itself inside a
        per-token loop, so exactly one new raw token is compressed per call;
        see the class docstring. Writes a new compressed entry into
        ``mla_cache`` (via ``mla_slot_mapping``, already restricted to this
        compressor's ``compress_ratio``, ``-1`` if this token does not
        complete a window) and the raw per-token kv/gate projection into
        ``self.state_cache`` (via ``state_slot_mapping``, a plain per-token
        sliding-window write). No return value -- this module only writes.
        """
        kv_gate = self.fused_wkv_wgate(hidden).unsqueeze(0)
        kv_new, gate_new = kv_gate[..., : self.width], kv_gate[..., self.width :]

        carry_kv, carry_gate, carry_valid = self._carry_rows(
            block_table_row, position_ids
        )
        replay_kv = torch.cat((carry_kv, kv_new), dim=1)
        replay_gate = torch.cat((carry_gate, gate_new), dim=1)

        # replay_{kv,gate} are always exactly [1, coff*ratio, width]: a fixed
        # carry_window=coff*ratio-1 rows plus this one new token. So
        # compress_fn always produces exactly `coff` candidate rows (a plain
        # Python int, not a traced value -- see compress_hca_chunk's
        # docstring) and the currently-completing window (if this token
        # completes one) is unconditionally the LAST of them. Earlier rows
        # can be all-(-inf)-gated -> NaN when no real prior window exists yet
        # (e.g. CSA's block 0 on a request's first-ever completed window) and
        # must never be read -- only `entry` (the last row) is.
        compress_fn = compress_csa_chunk if self.overlap else compress_hca_chunk
        compressed, _ = compress_fn(
            replay_kv, replay_gate, self.ape, None, carry_valid=carry_valid
        )
        entry = compressed[:, -1:]

        # Real RoPE, matching the real architecture's compressor exactly
        # (rope_layer_type="compress"): this entry's absolute position is the
        # start of the window it completes -- (position_ids // ratio) windows
        # were already complete (and their positions already emitted) before
        # this call.
        first_window_position = (
            torch.div(position_ids, self.ratio, rounding_mode="floor") * self.ratio
        )
        cos, sin = self.rotary_emb(
            entry, position_ids=first_window_position, layer_type="compress"
        )
        finalized = finalize_compressed_entries(
            entry, self.norm_weight, self.rms_norm_eps, cos, sin
        )
        # mla_slot_mapping is already [1]-shaped and 1:1 with `finalized`'s
        # single row by construction of the per-token call site;
        # scatter_paged_latent's existing slot_mapping==-1 filtering (already
        # Dynamo-safe) handles "don't write, this token didn't complete a
        # window" with no extra code here.
        scatter_paged_latent(mla_cache, mla_slot_mapping, finalized.squeeze(0))

        # Raw per-token projection feeds the *next* chunk's carry replay.
        scatter_paged_latent(self.state_cache, state_slot_mapping, kv_gate.squeeze(0))

    def forward_packed(
        self,
        hidden: torch.Tensor,
        *,
        positions: torch.Tensor,
        token_to_request: torch.Tensor,
        state_block_tables: torch.Tensor,
        state_slot_mapping: torch.Tensor,
        mla_cache: torch.Tensor,
        mla_slot_mapping: torch.Tensor,
    ) -> None:
        """Project packed tokens and write boundary-completed cache entries.

        Raw projections are scattered first, so a fixed window ending at each
        completion candidate naturally includes earlier tokens from the same
        prefill chunk. On Neuron, the boundary-only NKI path gathers and reduces
        only ``ceil(Q / ratio)`` candidate windows. CPU and the explicit
        ``VLLM_NEURON_DSV4_NKI_COMPRESSOR=0`` fallback retain the portable
        per-query oracle.
        """
        assert self.state_cache is not None
        kv_gate = self.fused_wkv_wgate(hidden)
        scatter_paged_latent(self.state_cache, state_slot_mapping, kv_gate)
        use_nki = can_run_kernel(hidden) and os.environ.get(
            "VLLM_NEURON_DSV4_NKI_COMPRESSOR", "1"
        ) != "0"
        if use_nki:
            compressed, entry_positions, output_slots, _ = paged_gated_compressor(
                self.state_cache,
                positions.reshape(-1),
                token_to_request.reshape(-1),
                state_block_tables,
                mla_slot_mapping,
                self.ape.to(torch.bfloat16),
                ratio=self.ratio,
                overlap=self.overlap,
            )
            entry = compressed.unsqueeze(1)
            entry_positions = entry_positions.reshape(-1, 1)
            cos, sin = self.rotary_emb(
                entry, position_ids=entry_positions, layer_type="compress"
            )
            finalized = finalize_compressed_entries(
                entry, self.norm_weight, self.rms_norm_eps, cos, sin
            ).squeeze(1)
            scatter_paged_latent(mla_cache, output_slots, finalized)
            return

        window = self.coff * self.ratio
        replay, valid = gather_recent_window_batched(
            self.state_cache,
            state_block_tables,
            token_to_request,
            window,
            positions,
        )
        replay = replay.squeeze(2)
        kv, gate = replay[..., : self.width], replay[..., self.width :]
        compress_fn = compress_csa_chunk if self.overlap else compress_hca_chunk
        compressed, _ = compress_fn(kv, gate, self.ape, carry_valid=valid)
        entry = compressed[:, -1:]
        entry_positions = (
            torch.div(positions.reshape(-1, 1), self.ratio, rounding_mode="floor")
            * self.ratio
        )
        cos, sin = self.rotary_emb(
            entry, position_ids=entry_positions, layer_type="compress"
        )
        finalized = finalize_compressed_entries(
            entry, self.norm_weight, self.rms_norm_eps, cos, sin
        ).squeeze(1)
        scatter_paged_latent(mla_cache, mla_slot_mapping, finalized)


class DeepseekV4Indexer(nn.Module):
    """Lightning indexer: picks which compressed entries CSA attends to.

    Only ``compress_ratio == 4`` (CSA) layers have one -- c128/HCA layers
    attend to their whole compressed history, and the real checkpoint ships
    indexer tensors for the c4 layers alone.

    The shape of this module is the surprising part: **the indexer runs a
    second, complete compressor of its own**, at ``index_head_dim`` instead of
    the model's ``head_dim``, over the same windows with the same overlap
    layout and the same rope theta. It is not a projection of the outer
    compressor's output -- it has its own ``wkv``/``wgate``/``ape``/``norm``
    weights and its own cache state, and the reference keeps the two side by
    side under the keys ``"compressor"`` and ``"indexer"``
    (``DeepseekV4CSACache``). That is why this owns a ``DeepseekV4Compressor``
    rather than sharing the attention layer's: same class, different width,
    different weights, separate caches.

    Its output is a boolean mask over compressed entries, ANDed into the
    attention's ``key_valid``. The reference spells the same thing as an
    additive ``-inf``/0 ``block_bias``; with one query token per call the two
    are the same statement, and a mask is what this plugin's attention already
    takes.

    Replicated across tensor-parallel ranks. Upstream head-shards ``q_b_proj``
    and ``weights_proj`` and all-reduces the scores so every rank picks the
    same entries; that is a valid optimization, but the ranks must agree on the
    selection exactly, and replication makes that true by construction rather
    than by a collective.
    """

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        ratio: int,
        rms_norm_eps: float,
        *,
        rotary_emb,
        qk_rope_head_dim: int,
    ):
        super().__init__()
        if ratio != 4:
            raise ValueError(
                f"only compressed_sparse_attention (ratio 4) layers carry a "
                f"lightning indexer, got ratio {ratio}"
            )
        self.ratio = ratio
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.index_topk = index_topk
        self.qk_rope_head_dim = qk_rope_head_dim
        # Same class as the outer compressor, at the indexer's own width. The
        # rotary is shared (one per model, selected by layer_type="compress"),
        # which is also what keeps query and key rotations on the same theta --
        # without that, ``q · k`` would carry a position-dependent skew.
        self.compressor = DeepseekV4Compressor(
            hidden_size,
            index_head_dim,
            ratio,
            rms_norm_eps,
            rotary_emb=rotary_emb,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        self.rotary_emb = rotary_emb
        self.q_b_proj = nn.Linear(
            q_lora_rank, index_n_heads * index_head_dim, bias=False
        )
        self.weights_proj = nn.Linear(hidden_size, index_n_heads, bias=False)
        # Bound by bind_kv_cache: this indexer's own compressed-entry pages.
        self.mla_cache: torch.Tensor | None = None
        self.mla_raw_block_size: int | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        q_residual: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        block_table_row: torch.Tensor,
        state_block_table_row: torch.Tensor,
        state_slot_mapping: torch.Tensor,
        mla_slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Select entries for this one token; returns a ``[max_entries]`` mask.

        ``hidden`` is ``[1, hidden_size]`` and ``q_residual`` is
        ``[1, q_lora_rank]`` -- the post-``q_a_norm`` residual the attention
        layer already computes for its own query projection, reused here rather
        than recomputed, exactly as the reference reuses it.

        Compress-then-read, in that order: the entry this token completes (if
        it completes one) must be visible to this very token, which is what
        ``visible_compressed_entries`` encodes.
        """
        self.compressor(
            hidden,
            position_ids=position_ids,
            block_table_row=state_block_table_row,
            state_slot_mapping=state_slot_mapping,
            mla_cache=self.mla_cache,
            mla_slot_mapping=mla_slot_mapping,
        )
        keys, valid = read_compressed_history(
            self.mla_cache,
            block_table_row,
            position_ids,
            compress_ratio=self.ratio,
            raw_block_size=self.mla_raw_block_size,
        )

        cos, sin = self.rotary_emb(
            hidden, position_ids=position_ids, layer_type="compress"
        )
        query = self.q_b_proj(q_residual).view(1, self.n_heads, self.head_dim)
        # Rotated in the same rank-3 layout the KV path uses: Neuron's lowering
        # of the rank-4 partial-RoPE concat zeroed the rotary channels (see
        # DeepseekV4Attention._forward_one_token).
        query = apply_partial_rotary(
            query, cos, sin, rope_dim=self.qk_rope_head_dim
        ).view(1, 1, self.n_heads, self.head_dim)

        gate = self.weights_proj(hidden).view(1, 1, self.n_heads)
        scores = lightning_index_scores(query, keys.unsqueeze(0), gate)
        visible = visible_compressed_entries(
            position_ids.view(()).long(), self.ratio
        ).view(1, 1)
        chosen = select_compressed_entries(scores, visible, self.index_topk)
        selected = selection_mask_from_indices(chosen, keys.shape[0])[0, 0]
        # ``valid`` is already implied by the causal mask inside the selection,
        # but ANDing keeps the two statements of "this entry is real" joined at
        # the point of use rather than relying on them having stayed equal.
        return valid & selected

    def forward_packed(
        self,
        hidden: torch.Tensor,
        q_residual: torch.Tensor,
        *,
        positions: torch.Tensor,
        token_to_request: torch.Tensor,
        block_tables: torch.Tensor,
        state_block_tables: torch.Tensor,
        state_slot_mapping: torch.Tensor,
        mla_slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized lightning selection for every packed query."""
        self.compressor.forward_packed(
            hidden,
            positions=positions,
            token_to_request=token_to_request,
            state_block_tables=state_block_tables,
            state_slot_mapping=state_slot_mapping,
            mla_cache=self.mla_cache,
            mla_slot_mapping=mla_slot_mapping,
        )
        pos2 = positions.reshape(-1, 1)
        cos, sin = self.rotary_emb(hidden, position_ids=pos2, layer_type="compress")
        query = self.q_b_proj(q_residual).view(-1, self.n_heads, self.head_dim)
        query = apply_partial_rotary(
            query, cos, sin, rope_dim=self.qk_rope_head_dim
        ).unsqueeze(1)
        gate = self.weights_proj(hidden).view(-1, 1, self.n_heads)
        visible = visible_compressed_entries(positions.reshape(-1), self.ratio)[:, None]
        if can_run_kernel(query):
            return paged_projected_bf16_indexer(
                query,
                gate,
                self.mla_cache,
                block_tables,
                token_to_request,
                visible[:, 0],
                logical_slots_per_block=self.mla_raw_block_size // self.ratio,
            )
        keys, valid = read_compressed_history_batched(
            self.mla_cache,
            block_tables,
            token_to_request,
            positions,
            compress_ratio=self.ratio,
            raw_block_size=self.mla_raw_block_size,
        )
        return streaming_topk_compressed_entries(
            query,
            keys,
            gate,
            visible,
            topk=self.index_topk,
            key_valid=valid,
        )


class NeuronDeepseekV4RotaryEmbedding(
    __import__(
        "transformers.models.deepseek_v4.modeling_deepseek_v4",
        fromlist=["DeepseekV4RotaryEmbedding"],
    ).DeepseekV4RotaryEmbedding
):
    """DeepSeek-V4 RoPE without an in-graph cross-device copy.

    The inherited frequency buffers move with the model. Calling
    ``.to(x.device)`` again inside ``forward`` makes FX-to-HLO replay attempt
    an unsupported XLA-to-Neuron copy.
    """

    @torch.no_grad()
    def reinitialize_deterministic_buffers(self) -> None:
        """Restore derived RoPE state after meta-device ``to_empty()``.

        These buffers are deliberately non-persistent in Transformers, so a
        Hugging Face checkpoint does not contain them. Materializing a model
        constructed on ``meta`` therefore has to recompute them from config.
        """
        for layer_type in self.layer_types:
            inv_freq_buffer = getattr(self, f"{layer_type}_inv_freq")
            rope_init_fn = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            # Always build on CPU, never on the buffer's own device. The
            # fallback here is stock ``ROPE_INIT_FUNCTIONS`` (yarn, which the
            # real DeepSeek-V4 configs select), and those do
            # ``torch.arange(...).to(device=device, dtype=torch.float)`` --
            # a device move and a dtype cast in one step, which Neuron
            # rejects. Only ``compute_default_rope_parameters`` above is
            # device-safe, so a config using default RoPE never exposed this.
            # ``copy_`` does the transfer, dtypes already matching.
            inv_freq, attention_scaling = rope_init_fn(
                self.config, torch.device("cpu"), layer_type=layer_type
            )
            inv_freq_buffer.copy_(inv_freq)
            getattr(self, f"{layer_type}_original_inv_freq").copy_(inv_freq)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @torch.no_grad()
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = (
            inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
        cos = freqs.cos() * attention_scaling
        sin = freqs.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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
        self,
        in_features_per_group: int,
        out_features: int,
        n_groups: int,
        bias: bool = False,
    ):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep this as a fixed Python loop. The mathematically equivalent
        # grouped ``bmm`` used by Transformers is vectorized by neuronx-cc in
        # a way that fails with NCC_IMGN901 for the singleton-token decode
        # graph. Independent rank-2 linears avoid that compiler path while
        # preserving the packed parameter layout (and therefore checkpoint
        # names and TP sharding) exactly.
        outputs = []
        out_features_per_group = self.out_features // self.n_groups
        for group in range(self.n_groups):
            start = group * out_features_per_group
            end = start + out_features_per_group
            outputs.append(
                F.linear(x[..., group, :], self.weight[start:end], bias=None)
            )
        return torch.stack(outputs, dim=-2)


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
    MLA TP split). Up through TP8, whole ``o_groups`` are assigned to ranks.
    Above TP8, consecutive ranks split one group's input columns and replicate
    that group's ``o_b_proj`` columns. The all-reduce after ``o_b_proj`` then
    reconstructs both the within-group column sum and the sum across groups.
    This extends the grouped projection through the production TP32 and TP64
    geometries without gathering attention heads.
    """

    def __init__(
        self,
        config: DeepseekV4ModelConfig,
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

        self.topology = resolve_parallel_topology()

        try:
            self.tp_group = get_tp_group()
            self.world_size = self.topology.tp_degree
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
        self.q_a_norm = DeepseekV4RMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = nn.Linear(
            q_lora_rank, self.heads_per_rank * self.head_dim, bias=False
        )
        # q_b_norm is unweighted (transformers.DeepseekV4UnweightedRMSNorm --
        # variance normalization, no learned scale) and applied inline in
        # _forward_one_token; no parameters to own here.
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=rms_norm_eps)

        o_groups = int(getattr(hf_config, "o_groups", 8))
        self.o_lora_rank = int(getattr(hf_config, "o_lora_rank", 1024))
        self.output_partition = resolve_output_projection_partition(
            tp_degree=self.world_size,
            tp_rank=self.topology.tp_rank if self.world_size > 1 else 0,
            output_groups=o_groups,
            total_input_width=num_heads * self.head_dim,
        )
        self.o_groups = self.output_partition.group_count
        local_width = self.heads_per_rank * self.head_dim
        expected_local_width = self.o_groups * self.output_partition.input_width
        if local_width != expected_local_width:
            raise ValueError(
                f"heads_per_rank*head_dim={local_width} does not match grouped "
                f"output projection width={expected_local_width}"
            )
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.output_partition.input_width,
            self.o_groups * self.o_lora_rank,
            self.o_groups,
        )
        self.o_b_proj = nn.Linear(
            self.o_groups * self.o_lora_rank, config.hidden_size, bias=False
        )
        self.sinks = nn.Parameter(torch.zeros(self.heads_per_rank))
        # Hookable no-op boundaries for focused CPU/Neuron accuracy capture.
        # They have no parameters and compile away when no capture hook is
        # registered, while avoiding unconditional manual-capture outputs in
        # ordinary tensor-capture runs.
        self.capture_q_roped = nn.Identity()
        self.capture_kv_roped = nn.Identity()
        self.capture_history = nn.Identity()
        self.capture_key_valid = nn.Identity()
        self.capture_attended_roped = nn.Identity()
        self.capture_attended = nn.Identity()

        # Head-sharded checkpoint parameters. Replicated q_a/kv parameters
        # intentionally have no loader and therefore remain identical on ranks.
        self._setup_weight_loaders(
            config, ratio, rms_norm_eps, rotary_emb, q_lora_rank
        )

    def _setup_weight_loaders(
        self,
        config=None,
        ratio=None,
        rms_norm_eps=None,
        rotary_emb=None,
        q_lora_rank=None,
    ) -> None:
        """Attach rank-aware loaders (also called after ``to_empty``)."""
        if self.world_size > 1:
            from vllm_neuron.utils.weight_loader import (
                SafetensorsWeightLoader,
                set_weight_loader,
                sharding_weight_loader,
                with_rank_override,
            )

            def loader(dim, size):
                return with_rank_override(
                    sharding_weight_loader(
                        shard_dim=dim, shard_size=size, num_shards=self.world_size
                    ),
                    rank=self.topology.tp_rank,
                )

            set_weight_loader(
                self.q_b_proj.weight,
                loader(0, self.q_b_proj.out_features),
            )
            partition = self.output_partition
            row_start = partition.group_start * self.o_lora_rank
            row_end = row_start + partition.group_count * self.o_lora_rank
            input_start = partition.input_offset
            input_end = input_start + partition.input_width
            set_weight_loader(
                self.o_a_proj.weight,
                SafetensorsWeightLoader(
                    transform=lambda slices, _: slices[0][
                        row_start:row_end, input_start:input_end
                    ]
                ),
            )
            set_weight_loader(
                self.o_b_proj.weight,
                SafetensorsWeightLoader(
                    transform=lambda slices, _: slices[0][:, row_start:row_end]
                ),
            )
            set_weight_loader(
                self.sinks,
                loader(0, self.heads_per_rank),
            )
        if config is None:
            return
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

        # Only CSA (ratio 4) layers select; c128/HCA attends to all of its
        # compressed history, and the checkpoint ships indexer tensors for the
        # c4 layers alone.
        self.indexer = (
            DeepseekV4Indexer(
                config.hidden_size,
                q_lora_rank,
                config.index_n_heads,
                config.index_head_dim,
                config.index_topk,
                ratio,
                rms_norm_eps,
                rotary_emb=rotary_emb,
                qk_rope_head_dim=self.qk_rope_head_dim,
            )
            if ratio == 4
            else None
        )

        # Bound by bind_kv_cache: raw [blocks, 1, slots, latent] tensors.
        self.swa_cache: torch.Tensor | None = None
        self.mla_cache: torch.Tensor | None = None

    @torch.no_grad()
    def reinitialize_deterministic_buffers(self) -> None:
        """Restore the identity K/V projection after meta materialization."""
        # Built on CPU and copied across, rather than assembled on the
        # parameter's own device. Neuron rejects a non-contiguous source in
        # ``copy_`` *and* cannot run the ``.contiguous()`` that would fix it,
        # so the broadcast has to be materialized somewhere it works. CPU
        # accepts the ``expand`` view either way, which is why this was
        # invisible until the first real device load. ``copy_`` moves it
        # across devices exactly as weight loading does, once per layer at
        # initialization.
        identity = torch.eye(self.head_dim, dtype=self.identity_kv_weight.dtype)
        self.identity_kv_weight.copy_(
            identity.unsqueeze(0).expand(self.heads_per_rank, -1, -1).contiguous()
        )

    def _swa_history(
        self,
        block_table_row: torch.Tensor,
        current_position: torch.Tensor,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed-size ``sliding_window`` window of uncompressed latents
        ending at (and including) this token's own absolute position.

        Dynamo-shape-static: always exactly ``self.sliding_window`` rows
        (``gather_recent_window``'s ``window`` is a compile-time constant),
        with a ``valid`` mask marking rows before generation has produced
        that much history yet -- only possible in the first
        ``sliding_window`` tokens of a request. Unlike the old
        variable-length gather (``gather_len = min(cached_seq_len,
        sliding_window)``, driven by a Python-int ``cached_seq_len`` that
        Dynamo cannot guard on -- see
        docs/model-dev/deepseek-v4-024-device-validation.md Step 5d),
        ``current_position`` is a real tensor end to end.

        This token's own kv is already physically written into
        ``self.swa_cache`` by the time this runs (``_forward_one_token``
        scatters before calling this), so the returned window directly *is*
        the full attention history for this token -- no separate
        concatenation of the freshly-computed kv is needed afterward, unlike
        before this redesign.

        The SWA cache is a true sliding-window group: the scheduler remaps
        blocks older than the window to a null block once the window has
        rolled past them (T1's
        ``test_sliding_window_remapping_uses_null_blocks_but_latents_remain_stable``
        covers this), but it never compacts the block table -- the live
        window's real data keeps living at ever-higher column indices as
        generation continues, it does not stay at column 0.
        ``gather_recent_window`` reads columns covering exactly
        ``[current_position + 1 - sliding_window, current_position]``, the
        live window, never a null-remapped column -- see
        docs/model-dev/deepseek-v4-swa-null-block-bug.md.
        """
        gathered, valid = gather_recent_window(
            (self.swa_cache if cache is None else cache),
            block_table_row,
            self.sliding_window,
            current_position,
        )
        return gathered.squeeze(1), valid

    def _compressed_history(
        self, block_table_row: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All prior compressed entries, Dynamo-shape-static.

        This group's cache stores one physical row per *compressed entry*,
        not per raw token (``storage_block_size = block_size //
        compress_ratio`` -- ``kv_spec_conversion.py``), while
        ``cached_seq_len`` is raw-token-scaled like every other group (see
        ``compressed_entry_slot_mapping``). The two address spaces differ by
        exactly ``compress_ratio``, matching the write side's
        ``raw_slot // compress_ratio``.

        Unlike ``_swa_history``/``_carry_rows``'s *sliding* windows, this
        group never evicts -- entries are addressed from 0 and simply
        accumulate, so "the first ``num_entries`` columns" is a fixed,
        growing *prefix*, not a moving window. That makes the Dynamo-static
        fix simpler than either of those: no tensor-derived offset is
        needed, just gather the *entire* block-table-addressable capacity
        (``max_entries`` -- a plain Python int from real tensor shapes,
        ``block_table_row``'s own column count times this cache's
        ``storage_block_size``, sized for the whole ``max_model_len`` by the
        same per-group convention documented in
        docs/model-dev/deepseek-v4-swa-null-block-bug.md) and mask off
        entries beyond the real current count -- rather than branching on
        ``cached_seq_len``'s value at all. Trades throughput for
        compilability (gathers the full capacity every call, not just the
        real entries so far), the same tradeoff ``DeepseekV4MoE.forward``'s
        always-compute redesign documents; not a concern for this pass's
        synthetic validation config, a real cost at production scale.
        """
        # ``(pos + 1) // ratio``, not ``pos // ratio``: a token completes a
        # window when ``(pos + 1) % ratio == 0`` (the write side's own rule, in
        # ``compressed_entry_slot_mapping``), and the compressor above has
        # already written that entry before this read. The reference gates
        # visibility the same way -- ``causal_threshold = (position_ids + 1) //
        # compress_rate`` in ``DeepseekV4CSACompressor``. Counting with
        # ``pos // ratio`` hides each new entry from the very query that
        # completes it, so outputs diverge at exactly the positions
        # ``pos % ratio == ratio - 1`` and agree everywhere else. That rule
        # lives in ``read_compressed_history`` so the indexer's own read of its
        # parallel cache cannot drift from this one.
        return read_compressed_history(
            self.mla_cache,
            block_table_row,
            position_ids,
            compress_ratio=self.ratio,
            raw_block_size=self.mla_raw_block_size,
        )

    def _forward_one_token(
        self,
        hidden: torch.Tensor,
        *,
        self_attn_name: str,
        attn_metadata: dict,
        request: int,
        local_index: int,
        position_ids: torch.Tensor,
        swa_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        # the class docstring). ``position_ids`` arrives as a real ``[1, 1]``
        # tensor built by the caller directly from ``attn_metadata`` --
        # never round-tripped through a Python/SymInt scalar embedded in a
        # fresh ``new_tensor([[...]])`` list here. That round trip is what
        # broke Dynamo/FakeTensor tracing (see
        # docs/model-dev/deepseek-v4-024-device-validation.md Step 5d):
        # ``int(fake_tensor)`` produces a symbolic (not a plain Python) int
        # under Dynamo, and building a brand-new tensor from a nested
        # Python list around that symbolic value goes through a plain
        # ``torch.tensor``-style eager-construction path that FakeTensorMode
        # doesn't understand, so the very next real op on it
        # (``rotary_emb``'s internal ``unsqueeze``) sees a tensor that was
        # never properly faked. Deriving ``position_ids`` purely through
        # tensor ops (slice + add + view, all real proxied ops) keeps it
        # symbolic-int-friendly end to end.
        cos, sin = self.rotary_emb(
            hidden, position_ids=position_ids, layer_type=self.rope_layer_type
        )

        # Bound rather than inlined: the lightning indexer projects its own
        # queries from this same residual (reference: DeepseekV4Attention hands
        # `q_residual` to the compressor's indexer).
        q_residual = self.q_a_norm(self.q_a_proj(hidden))
        q = self.q_b_proj(q_residual)
        # Rotate the query in the same rank-3 layout as KV.  Neuron's
        # lowering of the rank-4 partial-RoPE concat zeroed the two rotary
        # channels while the otherwise identical rank-3 KV path was correct.
        # The leading dimension is one token here by construction; restore
        # the singleton attention-time dimension after rotation.
        q = q.view(1, self.heads_per_rank, self.head_dim)
        q = q * torch.rsqrt(
            q.float().square().mean(-1, keepdim=True) + self.q_a_norm.eps
        ).to(q.dtype)
        cos_h, sin_h = cos.unsqueeze(2), sin.unsqueeze(2)  # broadcast over heads
        q_roped = apply_partial_rotary(
            q, cos, sin, rope_dim=self.qk_rope_head_dim
        ).unsqueeze(1)
        q_roped = self.capture_q_roped(q_roped)

        kv = self.kv_norm(self.kv_proj(hidden))  # [1, head_dim]
        kv_roped = apply_partial_rotary(
            kv.unsqueeze(0), cos, sin, rope_dim=self.qk_rope_head_dim
        ).squeeze(0)
        kv_roped = self.capture_kv_roped(kv_roped)
        updated_swa_cache = scatter_paged_latent(swa_cache, swa_slot, kv_roped)

        # position_ids is already this token's absolute position as a real
        # tensor (see the docstring above) -- reused directly as
        # _swa_history's current_position rather than reintroducing a
        # Python-int cached_seq_len dependency here.
        history, key_valid = self._swa_history(
            swa_block_table, position_ids, cache=updated_swa_cache
        )

        if self.compressor is not None:
            mla_entry = attn_metadata[self_attn_name]
            state_entry = attn_metadata[f"{self_attn_name}.compressor.state_cache"]
            mla_slot = compressed_entry_slot_mapping(
                mla_entry["slot_mapping"][local_index : local_index + 1],
                self.ratio,
                self.mla_raw_block_size,
                self.mla_cache.shape[2],
            )
            self.compressor(
                hidden,
                position_ids=position_ids,
                block_table_row=state_entry["block_table_tensor"][request],
                state_slot_mapping=state_entry["slot_mapping"][
                    local_index : local_index + 1
                ],
                mla_cache=self.mla_cache,
                mla_slot_mapping=mla_slot,
            )
            compressed_history, compressed_valid = self._compressed_history(
                mla_entry["block_table_tensor"][request], position_ids
            )
            if self.indexer is not None:
                indexer_entry = attn_metadata[f"{self_attn_name}.indexer"]
                indexer_state = attn_metadata[
                    f"{self_attn_name}.indexer.compressor.state_cache"
                ]
                # The indexer's pages are addressed exactly like the outer
                # compressor's -- same ratio, same block size -- so the same
                # slot arithmetic applies; only the stored width differs.
                indexer_slot = compressed_entry_slot_mapping(
                    indexer_entry["slot_mapping"][local_index : local_index + 1],
                    self.ratio,
                    self.indexer.mla_raw_block_size,
                    self.indexer.mla_cache.shape[2],
                )
                selected = self.indexer(
                    hidden,
                    q_residual,
                    position_ids=position_ids,
                    block_table_row=indexer_entry["block_table_tensor"][request],
                    state_block_table_row=indexer_state["block_table_tensor"][request],
                    state_slot_mapping=indexer_state["slot_mapping"][
                        local_index : local_index + 1
                    ],
                    mla_slot_mapping=indexer_slot,
                )
                # The whole point: entries the indexer did not pick are not
                # attended to. Below the dense bound this is a no-op, because
                # selecting the top-k is selecting everything.
                compressed_valid = compressed_valid & selected
            history = torch.cat((compressed_history, history), dim=0)
            # compressed_history is now a fixed-size (max_entries) buffer,
            # not just the real entries so far -- compressed_valid marks
            # which of those rows are real (see _compressed_history's
            # docstring). Padding rows sit between the real compressed
            # entries and the swa window below, but that's safe: causal
            # order among *real* content is still preserved (every real
            # compressed entry precedes every swa-window row regardless of
            # any padding gap), and mla_attention_reference's key_valid mask
            # excludes the padding itself from attention entirely.
            key_valid = torch.cat((compressed_valid, key_valid), dim=0)

        history = self.capture_history(history)
        key_valid = self.capture_key_valid(key_valid)
        attended = mla_attention_reference(
            q_roped,
            history.view(1, -1, history.shape[-1]),
            self.identity_kv_weight,
            self.identity_kv_weight,
            attention_sinks=self.sinks,
            sliding_window=self.sliding_window if self.ratio == 0 else None,
            key_valid=key_valid,
        )  # [1, 1, heads_per_rank, head_dim]
        attended = self.capture_attended_roped(attended)
        attended = apply_partial_rotary(
            attended, cos_h, sin_h, rope_dim=self.qk_rope_head_dim, inverse=True
        )
        attended = self.capture_attended(attended)
        return attended.reshape(
            1, self.heads_per_rank * self.head_dim
        ), updated_swa_cache

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        self_attn_name: str,
        attn_metadata: dict,
    ) -> torch.Tensor:
        """Dispatch to a single batched prefill or decode graph."""
        entry = attn_metadata[f"{self_attn_name}.swa_cache"]
        is_decode = entry["max_query_len"] <= entry["decode_token_threshold"]
        if (
            "token_to_request" not in entry
            and entry["block_table_tensor"].shape[0] != 1
        ):
            raise ValueError(
                "DeepSeek-V4 batched attention metadata requires token_to_request"
            )
        if is_decode:
            attended = self.forward_decode(
                hidden,
                positions,
                self_attn_name=self_attn_name,
                attn_metadata=attn_metadata,
            )
        else:
            attended = self.forward_prefill(
                hidden,
                positions,
                self_attn_name=self_attn_name,
                attn_metadata=attn_metadata,
            )

        grouped = attended.reshape(attended.shape[0], self.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(1)
        out = self.o_b_proj(grouped)
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out

    def forward_prefill(self, hidden, positions, *, self_attn_name, attn_metadata):
        return self._forward_packed(
            hidden,
            positions,
            self_attn_name=self_attn_name,
            attn_metadata=attn_metadata,
        )

    def forward_decode(self, hidden, positions, *, self_attn_name, attn_metadata):
        return self._forward_packed(
            hidden,
            positions,
            self_attn_name=self_attn_name,
            attn_metadata=attn_metadata,
        )

    def _forward_packed(self, hidden, positions, *, self_attn_name, attn_metadata):
        """One attention body for all scheduled tokens; no token unrolling."""
        entry = attn_metadata[f"{self_attn_name}.swa_cache"]
        owners = (
            entry["token_to_request"].reshape(-1).long()
            if "token_to_request" in entry
            else torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
        )
        if owners.shape[0] != hidden.shape[0]:
            raise ValueError("token_to_request must contain one id per packed token")
        pos = positions.reshape(-1).long()
        pos2 = pos[:, None]
        cos, sin = self.rotary_emb(
            hidden, position_ids=pos2, layer_type=self.rope_layer_type
        )

        q_residual = self.q_a_norm(self.q_a_proj(hidden))
        q = self.q_b_proj(q_residual).view(-1, self.heads_per_rank, self.head_dim)
        q = q * torch.rsqrt(
            q.float().square().mean(-1, keepdim=True) + self.q_a_norm.eps
        ).to(q.dtype)
        q_roped = self.capture_q_roped(
            apply_partial_rotary(q, cos, sin, rope_dim=self.qk_rope_head_dim).unsqueeze(
                1
            )
        )
        kv = self.kv_norm(self.kv_proj(hidden))
        kv_roped = self.capture_kv_roped(
            apply_partial_rotary(
                kv.unsqueeze(1), cos, sin, rope_dim=self.qk_rope_head_dim
            ).squeeze(1)
        )
        scatter_paged_latent(self.swa_cache, entry["slot_mapping"], kv_roped)
        sliding_logical, sliding_requested = recent_sliding_logical_indices(
            pos, count=self.sliding_window
        )
        sliding_slots, sliding_valid = logical_to_physical_slots_batched(
            sliding_logical,
            sliding_requested,
            entry["block_table_tensor"],
            owners,
            logical_slots_per_block=self.swa_cache.shape[2],
            physical_page_stride=self.swa_cache.shape[2],
            cache_blocks=self.swa_cache.shape[0],
        )
        compressed_slots = None
        compressed_valid = None
        compressed_uniform = False

        if self.compressor is not None:
            mla_entry = attn_metadata[self_attn_name]
            state_entry = attn_metadata[f"{self_attn_name}.compressor.state_cache"]
            mla_slots = compressed_entry_slot_mapping(
                mla_entry["slot_mapping"],
                self.ratio,
                self.mla_raw_block_size,
                self.mla_cache.shape[2],
            )
            self.compressor.forward_packed(
                hidden,
                positions=pos,
                token_to_request=owners,
                state_block_tables=state_entry["block_table_tensor"],
                state_slot_mapping=state_entry["slot_mapping"],
                mla_cache=self.mla_cache,
                mla_slot_mapping=mla_slots,
            )
            if self.indexer is None:
                # HCA uses a deterministic bounded suffix sized from the
                # compiled context capacity, in prefill as well as decode.
                #
                # The suffix can never hold more entries than the context can
                # produce (capacity / ratio), and every slot past that is masked
                # -inf. The online softmax absorbs those exactly: an all-invalid
                # tile has tile max -inf, so the merged max is unchanged,
                # prior_scale is exp(0) == 1, and its sum contribution is zero.
                # Prefill used to request 1024 entries unconditionally, which at
                # a 2048-token context is 64x more rows per query than can ever
                # be valid -- and each gathered row costs a DMA descriptor.
                #
                # This is vLLM's own rule (sparse_swa.py:218-223: entries are
                # bounded by cdiv(prefill_max_model_len, compress_ratio)),
                # rounded up to the nearest compiled bucket.  Rounding *up* is
                # load-bearing: it keeps `visible <= count` for every reachable
                # position, so `recent_compressed_logical_indices` returns
                # `start == 0` for every query and each query's requested rows
                # are identical.  That is exactly `compressed_uniform`, which
                # lets the kernel gather the stream once per launch instead of
                # once per query.  Never round down.
                #
                # `block_table_tensor.shape[1]` is a static shape, so this stays
                # a trace-time Python int and compiles to one specialization.
                capacity_entries = mla_entry["block_table_tensor"].shape[1] * (
                    self.mla_raw_block_size // self.ratio
                )
                eligible = [
                    bucket
                    for bucket in _HCA_COUNT_BUCKETS
                    if bucket >= capacity_entries
                ]
                if not eligible:
                    raise RuntimeError(
                        "DeepSeek-V4 HCA context capacity needs "
                        f"{capacity_entries} compressed entries; the largest "
                        f"compiled bucket is {max(_HCA_COUNT_BUCKETS)}"
                    )
                compressed_count = min(eligible)
                # Logical rows are identical at this capacity-derived bound,
                # but physical rows repeat only within one request. Batched HCA
                # therefore uses the per-query gather path.
                compressed_uniform = (
                    mla_entry["block_table_tensor"].shape[0] == 1
                )
                logical, requested = recent_compressed_logical_indices(
                    pos, compress_ratio=self.ratio, count=compressed_count
                )
                slots, compressed_valid = logical_to_physical_slots_batched(
                    logical,
                    requested,
                    mla_entry["block_table_tensor"],
                    owners,
                    logical_slots_per_block=self.mla_raw_block_size // self.ratio,
                    physical_page_stride=self.mla_cache.shape[2],
                    cache_blocks=self.mla_cache.shape[0],
                )
                compressed_slots = slots
            if self.indexer is not None:
                index_entry = attn_metadata[f"{self_attn_name}.indexer"]
                index_state = attn_metadata[
                    f"{self_attn_name}.indexer.compressor.state_cache"
                ]
                fixed_selection = os.environ.get(
                    "VLLM_NEURON_DSV4_FIXED_CSA_SELECTION", "0"
                ) == "1"
                if fixed_selection:
                    # Preserve the indexer's key-compressor/cache-write path;
                    # only replace scoring/top-k so the bisection removes one
                    # dependency edge at a time.
                    index_slots = compressed_entry_slot_mapping(
                        index_entry["slot_mapping"],
                        self.ratio,
                        self.indexer.mla_raw_block_size,
                        self.indexer.mla_cache.shape[2],
                    )
                    self.indexer.compressor.forward_packed(
                        hidden,
                        positions=pos,
                        token_to_request=owners,
                        state_block_tables=index_state["block_table_tensor"],
                        state_slot_mapping=index_state["slot_mapping"],
                        mla_cache=self.indexer.mla_cache,
                        mla_slot_mapping=index_slots,
                    )
                    physical_capacity = mla_entry["block_table_tensor"].shape[1] * (
                        self.mla_raw_block_size // self.ratio
                    )
                    # This is a runtime-hang bisection, not a production
                    # selector.  At large warmup buckets choose the first
                    # ``topk`` valid entries; the short diagnostic prompt has
                    # fewer live entries than ``topk``, so its selection
                    # remains dense-equivalent.
                    capacity = min(physical_capacity, self.indexer.index_topk)
                    selection = fixed_prefix_compressed_entries(
                        visible_compressed_entries(pos, self.ratio),
                        topk=self.indexer.index_topk,
                        capacity=capacity,
                    )
                else:
                    index_slots = compressed_entry_slot_mapping(
                        index_entry["slot_mapping"],
                        self.ratio,
                        self.indexer.mla_raw_block_size,
                        self.indexer.mla_cache.shape[2],
                    )
                    selection = self.indexer.forward_packed(
                        hidden,
                        q_residual,
                        positions=pos,
                        token_to_request=owners,
                        block_tables=index_entry["block_table_tensor"],
                        state_block_tables=index_state["block_table_tensor"],
                        state_slot_mapping=index_state["slot_mapping"],
                        mla_slot_mapping=index_slots,
                    )
                slots, compressed_valid = logical_to_physical_slots_batched(
                    selection.logical_indices,
                    selection.valid,
                    mla_entry["block_table_tensor"],
                    owners,
                    logical_slots_per_block=self.mla_raw_block_size // self.ratio,
                    physical_page_stride=self.mla_cache.shape[2],
                    cache_blocks=self.mla_cache.shape[0],
                )
                compressed_slots = slots

        attended = paged_shared_latent_mla(
            SharedLatentMLAInputs(
                query=q_roped,
                sliding_cache=self.swa_cache,
                sliding_slots=sliding_slots,
                sliding_valid=sliding_valid,
                compressed_cache=self.mla_cache
                if self.compressor is not None
                else None,
                compressed_slots=compressed_slots,
                compressed_valid=compressed_valid,
                sinks=self.sinks,
                sliding_contiguous=entry["block_table_tensor"].shape[0] == 1,
                compressed_uniform=compressed_uniform,
            )
        )
        attended = self.capture_attended_roped(attended)
        attended = apply_partial_rotary(
            attended,
            cos.unsqueeze(2),
            sin.unsqueeze(2),
            rope_dim=self.qk_rope_head_dim,
            inverse=True,
        )
        return self.capture_attended(attended).reshape(
            hidden.shape[0], self.heads_per_rank * self.head_dim
        )


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

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float,
        *,
        tp_degree: int = 1,
        tp_rank: int = 0,
        tp_group=None,
    ):
        super().__init__()
        if intermediate_size % tp_degree:
            raise ValueError(
                f"shared expert intermediate_size={intermediate_size} must be "
                f"divisible by tp_degree={tp_degree}"
            )
        self.full_intermediate_size = intermediate_size
        intermediate_size //= tp_degree
        self.tp_degree = tp_degree
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.gate_up_proj = nn.Parameter(
            torch.randn(2 * intermediate_size, hidden_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(hidden_size, intermediate_size) * 0.02
        )
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit
        # Checkpoints ship gate and up as separate ``w1``/``w3`` tensors while
        # this parameter is their concatenation on the output dim (``forward``
        # chunks it back apart). ``weight_loaders.load_checkpoint_weights``
        # places each half; no per-parameter shard loader is attached here
        # because ``to_empty()`` would drop it before loading runs.
        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        if self.tp_degree == 1:
            return
        from vllm_neuron.utils.weight_loader import (
            set_weight_loader,
            sharding_weight_loader,
            with_rank_override,
        )

        def loader(dim, size):
            return with_rank_override(
                sharding_weight_loader(
                    shard_dim=dim, shard_size=size, num_shards=self.tp_degree
                ),
                rank=self.tp_rank,
            )

        # w1/w3 are independently sliced by the model-specific stacked loader.
        set_weight_loader(self.down_proj, loader(1, self.intermediate_size))

    def forward_local(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return this rank's unreduced shared-expert TP partial.

        The ordinary DeepSeek-V4 MoE path combines this partial with the
        routed-expert partial before issuing its single TP all-reduce.  Keep
        the unreduced operation explicit so cross-DP EP can retain the two
        communication domains required by that topology.
        """
        gate_up = F.linear(hidden, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.linear(F.silu(gate) * up, self.down_proj)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = self.forward_local(hidden)
        if self.tp_degree > 1:
            output = self.tp_group.all_reduce(output)
        return output


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

    def __init__(self, config: DeepseekV4ModelConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.topk = config.topk
        self.num_experts = config.num_experts
        self.topology = resolve_parallel_topology()
        intermediate = config.expert_intermediate_size
        self.ep_degree = self.topology.ep_degree
        self.expert_tp_degree = self.topology.expert_tp_degree
        if intermediate % self.expert_tp_degree:
            raise ValueError(
                f"expert_intermediate_size={intermediate} must be divisible by "
                f"expert_tp_degree={self.expert_tp_degree}"
            )
        intermediate //= self.expert_tp_degree
        self.expert_tp_rank = self.topology.expert_tp_rank
        self.full_intermediate_size = config.expert_intermediate_size
        self.num_local_experts = self.num_experts // self.ep_degree
        self.local_start, self.local_end = self.topology.local_expert_interval(
            self.num_experts
        )

        try:
            from vllm.distributed.parallel_state import get_tp_group
            self.tp_group = get_tp_group()
        except AssertionError:
            self.tp_group = None

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.routed_gate_up = nn.Parameter(
            torch.randn(self.num_local_experts, config.hidden_size, 2, intermediate)
            * 0.02
        )
        self.routed_down = nn.Parameter(
            torch.randn(self.num_local_experts, intermediate, config.hidden_size) * 0.02
        )
        self.shared_experts = DeepseekV4Expert(
            config.hidden_size,
            self.full_intermediate_size * config.n_shared_experts,
            config.swiglu_limit,
            tp_degree=self.topology.tp_degree,
            tp_rank=self.topology.tp_rank,
            tp_group=self.tp_group,
        )
        self.routed_scaling_factor = config.routed_scaling_factor
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
        """Recompute the fallback ``tid2eid`` in place, e.g. after ``to_empty()``.

        vLLM constructs models on the meta device, so the ``arange``/
        ``remainder`` that built this buffer in ``__init__`` never actually ran
        -- it only recorded shape/dtype. ``to_empty()`` gives it real but
        uninitialized storage, and ``load_weights`` calls this to fill it (see
        the class docstring's EP note -- this buffer must be identical across
        ranks).

        This value is a **fallback only**. Real DeepSeek-V4 checkpoints ship a
        frozen ``ffn.gate.tid2eid`` table for every hash-routed layer, and the
        arange pattern here agrees with it at only about chance rate, so a
        checkpoint's table must win. ``load_weights`` runs this *before*
        ``load_checkpoint_weights``, which now also resolves buffers, so a
        provided table overwrites this. Keep that ordering.
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
        original_tokens = hidden.shape[0]
        cross_dp_ep = self.ep_degree > self.topology.tp_degree
        dp_group = wide_ep_group = None
        if cross_dp_ep:
            from vllm.distributed.parallel_state import get_dp_group, get_wide_ep_group

            dp_group = get_dp_group()
            wide_ep_group = get_wide_ep_group()
            hidden = dp_group.all_gather(hidden, dim=0)
            token_id = dp_group.all_gather(token_id, dim=0)
        logits = self.gate(hidden)
        if self.kind == "hash_moe":
            ids, weights = hash_topk(
                logits, token_id, self.tid2eid, self.routed_scaling_factor
            )
        else:
            ids, weights = routed_topk(
                logits, self.correction_bias, self.topk, self.routed_scaling_factor
            )

        # Routing normalization and duplicate-slot aggregation stay in FP32.
        # Hash routing can repeat an expert id; scatter_add preserves all six
        # contributions before block dispatch.
        affinities = dense_expert_affinities(ids, weights.float(), self.num_experts)
        if can_run_kernel(hidden):
            routed = self._forward_nki(hidden, affinities)
        else:
            routed = self._forward_portable(hidden, affinities)

        if cross_dp_ep:
            routed = wide_ep_group.all_reduce(routed)
            start = dp_group.rank_in_group * original_tokens
            routed = routed[start : start + original_tokens]
            hidden = hidden[start : start + original_tokens]
            # Routed experts reduce over the wide EP world while the shared
            # expert is sharded only over TP.  These domains are not
            # interchangeable, so preserve the legacy separate reductions.
            return routed + self.shared_experts(hidden)

        # When EP is wholly contained inside TP, both routed and shared
        # outputs are local partials over the same final TP domain.  Addition
        # is linear, so reduce their sum once instead of reducing each term:
        #   all_reduce(routed + shared) == all_reduce(routed) + all_reduce(shared)
        # ``VLLM_NEURON_DSV4_FUSED_MOE_REDUCTION=0`` restores the two separate
        # reductions for on-device A/B attribution of the saved collective.
        if os.environ.get("VLLM_NEURON_DSV4_FUSED_MOE_REDUCTION", "1") == "0":
            if self.topology.tp_degree > 1:
                routed = self.tp_group.all_reduce(routed)
            return routed + self.shared_experts(hidden)

        output = routed + self.shared_experts.forward_local(hidden)
        if self.topology.tp_degree > 1:
            output = self.tp_group.all_reduce(output)
        return output

    def _forward_portable(
        self, hidden: torch.Tensor, affinities: torch.Tensor
    ) -> torch.Tensor:
        routed = torch.zeros_like(hidden)
        for local_expert in range(self.num_local_experts):
            global_expert = self.local_start + local_expert
            gate_up = torch.einsum(
                "th,hgi->tgi", hidden, self.routed_gate_up[local_expert]
            )
            gate = gate_up[:, 0].clamp(max=self.shared_experts.swiglu_limit)
            up = gate_up[:, 1].clamp(
                min=-self.shared_experts.swiglu_limit,
                max=self.shared_experts.swiglu_limit,
            )
            expert_out = (F.silu(gate) * up) @ self.routed_down[local_expert]
            routed = (
                routed + expert_out * affinities[:, global_expert : global_expert + 1]
            )
        return routed

    def _forward_nki(
        self, hidden: torch.Tensor, affinities: torch.Tensor
    ) -> torch.Tensor:
        """One opaque BF16 shard-on-block routed-MoE call."""
        if hidden.dtype != torch.bfloat16:
            raise RuntimeError(
                f"DeepSeek-V4 NKI MoE requires bfloat16 activations, got {hidden.dtype}"
            )
        intermediate = self.routed_gate_up.shape[-1]
        if intermediate < 128 or intermediate % 16:
            raise RuntimeError(
                "DeepSeek-V4 NKI MoE requires expert intermediate size >=128 "
                f"and divisible by 16, got {intermediate}"
            )
        if (
            self.routed_gate_up.dtype != torch.bfloat16
            or self.routed_down.dtype != torch.bfloat16
        ):
            raise RuntimeError(
                "DeepSeek-V4 NKI MoE requires BF16 routed expert weights"
            )

        import nki.language as nl
        from nkilib.core.moe.moe_cte.moe_cte import (
            ActFnType,
            ExpertAffinityScaleMode,
            MoECTEImplementation,
        )

        import vllm_neuron.functional as NF
        from vllm.distributed import get_tp_group
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_degree,
            get_neuron_ep_tp_group,
        )

        original_tokens = hidden.shape[0]
        # The BF16 shard-on-block kernel statically unrolls one body per routed
        # block.  Q8192 with B128 exceeds the compiler's five-million-
        # instruction limit; B512 cuts that fanout by roughly four while
        # remaining within the kernel's documented 128..512 geometry. Decode
        # and ordinary prefills retain B128 to avoid unnecessary padding.
        moe_block_size = 512 if original_tokens > 4096 else 128
        # Carry inert rows through routing and trim the kernel result. Zero
        # affinities keep padding out of every expert.
        padded_tokens = (
            (original_tokens + moe_block_size - 1) // moe_block_size
        ) * moe_block_size
        if padded_tokens != original_tokens:
            token_padding = padded_tokens - original_tokens
            hidden = F.pad(hidden, (0, 0, 0, token_padding))
            affinities = F.pad(affinities, (0, 0, 0, token_padding))

        local_affinities = affinities[
            :, self.local_start : self.local_start + self.num_local_experts
        ]
        # EP-specific groups are deliberately not constructed for EP1.  In
        # that geometry the ordinary TP group is the identical communication
        # domain (and at TP1 both are a single-rank no-op group).
        group = (
            get_neuron_ep_tp_group()
            if get_neuron_ep_degree() > 1
            else get_tp_group()
        )
        masked, token_ids, block_experts, conditions = NF.build_blockwise_mapping(
            expert_affinities=local_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.topk,
            block_size=moe_block_size,
            moe_group=group,
            tp_degree=self.expert_tp_degree,
        )
        output = NF.moe_cte(
            hidden_states=hidden,
            expert_affinities_masked=masked,
            gate_up_proj_weight=self.routed_gate_up,
            down_proj_weight=self.routed_down,
            token_position_to_id=token_ids,
            block_to_expert=block_experts,
            conditions=conditions,
            block_size=moe_block_size,
            implementation=MoECTEImplementation.shard_on_block,
            activation_function=ActFnType.SiLU,
            compute_dtype=nl.bfloat16,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            gate_clamp_upper_limit=10.0,
            gate_clamp_lower_limit=-10.0,
            up_clamp_upper_limit=10.0,
            up_clamp_lower_limit=-10.0,
            skip_token=True,
            is_tensor_update_accumulating=True,
        )
        return output[:original_tokens]


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
        config: DeepseekV4ModelConfig,
        layer,
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
        self.moe = DeepseekV4MoE(config, layer.mlp.value)
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)
        self.input_layernorm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        streams: torch.Tensor,
        token_id: torch.Tensor,
        positions: torch.Tensor,
        *,
        self_attn_name: str,
        attn_metadata: dict,
    ) -> torch.Tensor:
        post, comb, hidden = self.attn_hc(streams)
        attended = self.attention(
            self.input_layernorm(hidden),
            positions,
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

    def __init__(self, config: DeepseekV4ModelConfig, hf_config):
        super().__init__()
        self.config = config
        self.topology = resolve_parallel_topology()
        if self.topology.tp_degree > 1:
            from vllm.distributed.parallel_state import get_tp_group
            from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

            self.embed_tokens = VocabDimShardedEmbedding(
                vocab_size=config.vocab_size,
                embed_dim=config.hidden_size,
                dtype=config.torch_dtype,
                tp_group=get_tp_group().device_group,
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self._setup_weight_loaders()
        # One rotary_emb shared by every layer's attention and compressor,
        # matching the real architecture's single model.rotary_emb -- it
        # holds both the "main" (sliding-window layers) and "compress"
        # (compressed-attention layers and their compressors) inv_freq
        # tables internally, selected per call via layer_type.
        rotary_emb = NeuronDeepseekV4RotaryEmbedding(hf_config)
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
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _setup_weight_loaders(self) -> None:
        if self.topology.tp_degree == 1:
            return
        from vllm_neuron.utils.weight_loader import (
            set_weight_loader,
            sharding_weight_loader,
            with_rank_override,
        )

        loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.embed_tokens.vocab_size_per_rank,
            num_shards=self.topology.tp_degree,
            pad_shard=True,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            with_rank_override(loader, rank=self.topology.tp_rank),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError("DeepSeek-V4 expects a flat [tokens] input_ids tensor")
        hidden = (
            self.embed_tokens(input_ids, scatter_tokens=False)
            if self.topology.tp_degree > 1
            else self.embed_tokens(input_ids)
        )
        if inputs_embeds is not None:
            if is_token_ids is None:
                raise ValueError("is_token_ids is required with inputs_embeds")
            if inputs_embeds.shape != hidden.shape:
                raise ValueError("inputs_embeds shape must match embedded token shape")
            hidden = torch.where(is_token_ids.view(-1, 1), hidden, inputs_embeds)
        streams = hidden.unsqueeze(-2).expand(-1, self.config.hc_mult, -1)
        for index, layer in enumerate(self.layers):
            self_attn_name = f"model.layers.{index}.self_attn"
            streams = layer(
                streams,
                input_ids,
                positions,
                self_attn_name=self_attn_name,
                attn_metadata=attn_metadata,
            )
        return self.norm(self.hc_head(streams))


@torch.no_grad()
def _initialize_dummy_parameters(module: nn.Module) -> None:
    """Materialize stable, finite dummy weights without changing buffers.

    The Neuron runner bypasses vLLM's generic ``DummyModelLoader`` and invokes
    this model's loader directly. Config-only depth checkpoints therefore need
    the equivalent per-parameter initialization here. Buffers are left alone
    because routing tables and RoPE state were just reconstructed by their
    owning modules.
    """
    from vllm.model_executor.model_loader.weight_utils import (
        initialize_single_dummy_weight,
    )

    for parameter in module.parameters():
        if parameter.device.type != "neuron":
            initialize_single_dummy_weight(parameter)
            continue
        if not torch.is_floating_point(parameter):
            continue
        # PrivateUse1 does not provide torch.Generator, which the generic vLLM
        # initializer assumes. Match its stable per-parameter policy on CPU,
        # then use the normal host-to-Neuron weight copy path.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1234)
        value = torch.empty(
            parameter.shape,
            dtype=parameter.dtype,
            device="cpu",
        ).uniform_(-1e-3, 1e-3, generator=generator)
        parameter.copy_(value)


class DeepseekV4ForCausalLM(nn.Module):
    """Device-shaped DeepSeek-V4: batched, ``attn_metadata``-driven forward."""

    is_text_generation_model = True
    # Functional cache updates return whole tensor views from captured graphs.
    # Those outputs cannot safely alias the heterogeneous BF16/FP32 byte pools
    # that vLLM normally shares across cache groups: Neuron's alias rewrite
    # would let the final whole-view output overwrite the preceding groups.
    requires_independent_kv_cache_tensors = True

    def __init__(self, config: DeepseekV4ModelConfig, hf_config):
        super().__init__()
        self.config = config
        self.topology = resolve_parallel_topology()
        self.topology.validate(
            num_heads=int(hf_config.num_attention_heads),
            output_groups=int(getattr(hf_config, "o_groups", 8)),
            num_experts=config.num_experts,
            expert_intermediate_size=config.expert_intermediate_size,
        )
        expert_start, expert_end = self.topology.local_expert_interval(
            config.num_experts
        )
        logger.info(
            "DeepSeek-V4 topology: TP=%d DP=%d EP=%d expert-TP=%d "
            "local_experts=[%d,%d) logical_cores=%s",
            self.topology.tp_degree,
            self.topology.dp_degree,
            self.topology.ep_degree,
            self.topology.expert_tp_degree,
            expert_start,
            expert_end,
            os.environ.get("NEURON_RT_VISIBLE_CORES", "runtime-assigned"),
        )
        # Every parameter below `DeepseekV4Model` is created by a bare
        # `nn.Linear` / `nn.Embedding` / `nn.Parameter(torch.randn(...))`,
        # none of which take a dtype, so they were all built at
        # `torch.get_default_dtype()` -- FP32 -- regardless of the configured
        # BF16. `load_weights` then upcast the BF16 checkpoint into them via
        # `copy_`, so the model silently occupied ~2x its intended footprint
        # (measured on the real Flash slice: 12.5 GiB of FP32 parameters
        # against a 7.3 GiB BF16 checkpoint) and `dtype` was inert.
        #
        # `lm_head` below already passes `dtype=` explicitly, which is why it
        # was the single BF16 tensor in that measurement; it is unaffected by
        # this context. Scoped to the module construction rather than fixed at
        # each site so a newly added layer cannot silently reintroduce it --
        # `llama3/model.py` threads `dtype=` by hand and is the other valid
        # pattern here.
        #
        # RoPE is unaffected: `inv_freq` is built through an explicit
        # `.float()` in the Transformers rope-init functions, and
        # `NeuronDeepseekV4RotaryEmbedding.forward` upcasts to FP32 again
        # before the matmul.
        with set_default_torch_dtype(config.torch_dtype):
            self.model = DeepseekV4Model(config, hf_config)
        from vllm_neuron import nn as neuron_nn

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config is not None
            else None
        )
        debug_logits_enabled = bool(
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = bool(
            config.neuron_config is not None
            and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled
        try:
            from vllm.distributed.parallel_state import get_tp_group

            self.lm_head_tp_group = get_tp_group()
            device_group = self.lm_head_tp_group.device_group
        except AssertionError:
            self.lm_head_tp_group = None
            device_group = None
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=self.on_device_sampling_config is None,
            tp_group=device_group,
        )
        self._setup_lm_head_weight_loader()
        if self.on_device_sampling_config is not None:
            self.sampler = neuron_nn.Sampler(
                self.on_device_sampling_config,
                process_group=device_group,
                vocab_size=config.vocab_size,
            )

    def _setup_lm_head_weight_loader(self) -> None:
        if self.topology.tp_degree == 1:
            return
        from vllm_neuron.utils.weight_loader import (
            set_weight_loader,
            sharding_weight_loader,
            with_rank_override,
        )

        loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.lm_head.out_features_per_rank,
            num_shards=self.topology.tp_degree,
            pad_shard=True,
        )
        set_weight_loader(
            self.lm_head.weight,
            with_rank_override(loader, rank=self.topology.tp_rank),
        )

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None):
        config = DeepseekV4ModelConfig.from_configs(hf_config, neuron_config)
        return cls(config, hf_config)

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
        requested_block_size = (
            getattr(self.config.neuron_config, "kv_cache_block_size", None) or 32
        )
        # vLLM 0.24 requires every sliding-MLA page to fit within the largest
        # full-MLA page before it unifies heterogeneous groups. The ratio-4
        # cache is the largest full group; four raw-token slots per compressed
        # entry makes 4 * the public block size give it exactly the SWA page's
        # byte width. Keep 128 as the established floor for block_size=32.
        compressed_raw_block_size = max(128, 4 * requested_block_size)
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
                    # Compressed-entry addressing requires this raw-token page
                    # width to be divisible by every supported ratio. It also
                    # scales with the public block size so vLLM can group it
                    # with the SWA page at block sizes above the default 32.
                    block_size=compressed_raw_block_size,
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
            if ratio != 4:
                continue
            # The lightning indexer's own compressor: the same two layouts
            # again at index_head_dim instead of latent_size. Separate groups
            # rather than a wider shared one -- the reference keeps the two
            # compressors' states side by side and they are read at different
            # widths by different consumers.
            specs.append(
                LayerSpec(
                    name=f"{prefix}.indexer",
                    num_kv_heads=1,
                    head_size=self.config.index_head_dim,
                    dtype=torch.bfloat16,
                    cache_kind=CacheKind.MLA,
                    compress_ratio=ratio,
                    block_size=compressed_raw_block_size,
                    alignment=128,
                )
            )
            specs.append(
                LayerSpec(
                    name=f"{prefix}.indexer.compressor.state_cache",
                    num_kv_heads=1,
                    head_size=4 * self.config.index_head_dim,
                    dtype=torch.float32,
                    block_size=4,
                    cache_kind=CacheKind.COMPRESSOR_STATE,
                    sliding_window_size=2 * ratio,
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
                layer.attention.mla_raw_block_size = next(
                    s.block_size for s in self.get_kv_spec().layers if s.name == prefix
                )
                logical_width = (
                    layer.attention.mla_raw_block_size // layer.attention.ratio
                )
                if layer.attention.mla_cache.shape[2] < logical_width:
                    raise ValueError(
                        f"{prefix} physical stride is smaller than logical compressed width"
                    )
                layer.attention.compressor.state_cache = kv_caches[
                    f"{prefix}.compressor.state_cache"
                ][0]
            if layer.attention.indexer is not None:
                indexer = layer.attention.indexer
                indexer.mla_cache = kv_caches[f"{prefix}.indexer"][0]
                indexer.mla_raw_block_size = next(
                    s.block_size
                    for s in self.get_kv_spec().layers
                    if s.name == f"{prefix}.indexer"
                )
                if indexer.mla_cache.shape[2] < (
                    indexer.mla_raw_block_size // indexer.ratio
                ):
                    raise ValueError(
                        f"{prefix}.indexer physical stride is smaller than "
                        f"logical compressed width"
                    )
                indexer.compressor.state_cache = kv_caches[
                    f"{prefix}.indexer.compressor.state_cache"
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
        if spec_decode_metadata is not None:
            raise ValueError("DeepSeek-V4 does not support speculative/MTP decoding")
        hidden = self.model(
            input_ids,
            positions,
            attn_metadata,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )
        selected = torch.index_select(hidden, 0, sampling_positions)
        logits = self.compute_logits(selected.to(self.config.torch_dtype))
        if self.on_device_sampling_config is None:
            return logits
        gathered_logits = None
        if self._gather_logits:
            gathered_logits = (
                self.lm_head_tp_group.all_gather(logits, dim=1)
                if self.lm_head_tp_group is not None
                else logits
            )
        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        return sampled_tokens, gathered_logits

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
        the checkpoint directory has no ``.safetensors`` files, this initializes
        every parameter with vLLM's deterministic dummy-weight policy. The
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
            search_root = os.path.dirname(
                source.get_file_path(source.get_file_names()[0])
            )

        files = sorted(glob.glob(os.path.join(search_root, "*.safetensors")))

        # vLLM constructs the model on the meta device before load_weights
        # runs, and the runner moves it onto `device` right afterward
        # regardless of whether a checkpoint was found -- to_empty() must
        # run unconditionally, or that later .to(device) hits the same
        # "cannot copy out of meta tensor" error load_weights exists to
        # avoid. Deterministic buffers (moe routing tables, never provided
        # by a checkpoint) are recomputed explicitly since nothing else
        # touches them. If no checkpoint is present, the parameter-only dummy
        # initializer below fills the storage allocated by to_empty().
        self.to_empty(device=device)
        self.model._setup_weight_loaders()
        self._setup_lm_head_weight_loader()
        for module in self.modules():
            if isinstance(
                module,
                (
                    DeepseekV4MoE,
                    DeepseekV4Expert,
                    DeepseekV4Attention,
                    NeuronDeepseekV4RotaryEmbedding,
                ),
            ):
                if hasattr(module, "reinitialize_deterministic_buffers"):
                    module.reinitialize_deterministic_buffers()
                if hasattr(module, "_setup_weight_loaders"):
                    module._setup_weight_loaders()
        if not files:
            _initialize_dummy_parameters(self)
            parameter_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.parameters()
            )
            logger.info(
                "DeepSeek-V4 initialized deterministic dummy parameters; "
                "rank footprint: %.3f GiB",
                parameter_bytes / (1024**3),
            )
            return

        def _weights():
            local_start, local_end = self.topology.local_expert_interval(
                self.config.num_experts
            )
            for path in files:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        routed = re.search(r"\.ffn\.experts\.(\d+)\.", name)
                        if routed and not local_start <= int(routed.group(1)) < local_end:
                            continue
                        yield name, handle.get_tensor(name)

        load_checkpoint_weights(
            self, _weights(), expert_dtype=expert_dtype, strict=True
        )
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters()
        )
        logger.info(
            "DeepSeek-V4 rank parameter footprint: %.3f GiB",
            parameter_bytes / (1024**3),
        )
