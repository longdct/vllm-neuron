# SPDX-License-Identifier: Apache-2.0
"""Lightning-indexer scoring and top-k selection.

The indexer is what makes Compressed Sparse Attention sparse. It scores every
compressed entry against the current query with a small dedicated head and keeps
only the best ``index_topk`` of them; core attention then sees those and nothing
else. Critically it **only selects, it never weights** -- a selected entry enters
attention with exactly the weight it would have had under dense attention. That
is why omitting the indexer is *exact* wherever the eligible set is no larger
than ``index_topk``, and it is the property ``dense_csa`` turns into an admission
bound.

Two things in here are easy to get subtly wrong, so both are spelled out.

**The causal threshold is not this module's to invent.** A query may only see
compressed entries that already exist, and "already exists" is decided by
:func:`~vllm_neuron.model.deepseek_v4.attention.visible_compressed_entries` --
the same function the dense read path uses. Selection and visibility disagreeing
by one entry is not a rounding error: it is a different model, and it is
invisible at every position except ``position % ratio == ratio - 1``. That exact
off-by-one already shipped here once. The threshold is therefore taken as an
argument, never recomputed.

**Masking before the top-k is load-bearing, not defensive.** Entries past the
threshold are zeroed by the reader, and a zeroed key scores exactly
``relu(q·0) == 0``. A real entry whose gate weight is negative scores *below*
zero, so an unmasked top-k would spend its budget on padding and drop real
content. The ``-inf`` fill is what keeps the selection honest.

Free of ``vllm`` imports and of any config object: these are tensor functions
with their geometry read off the tensors themselves, which is what lets them be
compared against the Transformers reference directly.
"""

from __future__ import annotations

import torch

__all__ = [
    "lightning_index_scores",
    "select_compressed_entries",
    "selection_mask_from_indices",
]


def lightning_index_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Score compressed entries: ``∑_h w_{t,h} · ReLU(q_{t,h} · k_s)``.

    Paper §2.3.1 eqs. 14-16, matching
    ``transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4IndexerScorer``.

    ``query`` is ``[batch, tokens, heads, head_dim]``, ``keys`` is
    ``[batch, entries, head_dim]`` (one shared key per entry -- the indexer is
    MQA), and ``gate`` is the raw ``weights_proj`` output,
    ``[batch, tokens, heads]``. Returns ``[batch, tokens, entries]``.

    Both scale factors are derived from the tensors rather than passed in:
    ``head_dim**-0.5`` on the scores and ``heads**-0.5`` on the gate. There is
    no third scale to get wrong.

    Computed in fp32 regardless of input dtype. The scores feed a top-k, so
    ties and near-ties decide *which* entries attention sees; bf16 collapses
    neighbouring scores into exact ties and makes the selection depend on the
    tie-break rather than on the model.
    """
    if query.ndim != 4:
        raise ValueError("query must be [batch, tokens, heads, head_dim]")
    if keys.ndim != 3:
        raise ValueError("keys must be [batch, entries, head_dim]")
    if gate.ndim != 3:
        raise ValueError("gate must be [batch, tokens, heads]")
    if query.shape[-1] != keys.shape[-1]:
        raise ValueError("query and keys must share head_dim")
    if gate.shape[:2] != query.shape[:2] or gate.shape[2] != query.shape[2]:
        raise ValueError("gate must be [batch, tokens, heads] matching query")

    head_dim = query.shape[-1]
    heads = query.shape[2]
    # [b,t,h,d] @ [b,1,d,e] -> [b,t,h,e]
    scores = torch.matmul(query.float(), keys.float().transpose(-1, -2).unsqueeze(1))
    scores = torch.relu(scores) * (head_dim**-0.5)
    weights = gate.float() * (heads**-0.5)
    return (scores * weights.unsqueeze(-1)).sum(dim=2)


def select_compressed_entries(
    scores: torch.Tensor,
    visible: torch.Tensor,
    index_topk: int,
) -> torch.Tensor:
    """Top-``index_topk`` entry indices per token, ``-1`` where nothing qualifies.

    ``scores`` is ``[batch, tokens, entries]``, ``visible`` is ``[batch, tokens]``
    holding each token's entry count from ``visible_compressed_entries``. Returns
    ``[batch, tokens, k]`` int64 with ``k = min(index_topk, entries)``.

    Entries at or past a token's own threshold are ``-inf``-masked before the
    top-k so they cannot displace real content. A token with fewer than ``k``
    visible entries still gets ``k`` picks back -- top-k has a fixed width, and
    a data-dependent one would not compile -- so the surplus picks are returned
    as ``-1``. Callers must treat ``-1`` as "no entry", which
    :func:`selection_mask_from_indices` does.

    ``k`` is a Python int, derived from ``index_topk`` and the entry axis, so the
    output shape is a compile-time constant.
    """
    if scores.ndim != 3:
        raise ValueError("scores must be [batch, tokens, entries]")
    if visible.ndim != 2 or visible.shape != scores.shape[:2]:
        raise ValueError("visible must be [batch, tokens] matching scores")
    if index_topk < 1:
        raise ValueError("index_topk must be >= 1")

    entries = scores.shape[-1]
    k = min(index_topk, entries)
    threshold = visible.unsqueeze(-1)
    positions = torch.arange(entries, device=scores.device).view(1, 1, -1)
    future = positions >= threshold
    masked = scores.float().masked_fill(future, float("-inf"))
    # Positional [1], never ``.indices``: torch CPU returns a named tuple while
    # the Torch-XLA bridge returns a plain two-element list (see moe.py).
    #
    # ``.long()`` is not cosmetic. Neuron's top-k returns **uint32** indices
    # where CPU returns int64, and on an unsigned type the ``-1`` sentinel
    # below wraps to 4294967295 while ``>= 0`` is vacuously true -- so the
    # sentinel is never recognised and that value is handed to the scatter in
    # :func:`selection_mask_from_indices` as a real index. That reaches the
    # device as "indirect memory copy via vector DGE out-of-bound access",
    # reported against the whole NEFF with no hint of which op caused it.
    #
    # The range test covers the other half of the same problem: a row that is
    # entirely ``-inf`` (a token with no visible entries yet) leaves top-k free
    # to return whatever it likes, and only CPU promises 0..k-1. Anything
    # outside the entry axis becomes the sentinel rather than being clamped
    # into it -- clamping would turn a meaningless pick into a confident
    # selection of the last entry, which is worse than dropping it.
    chosen = torch.topk(masked, k, dim=-1)[1].long()
    invalid = (chosen >= threshold) | (chosen < 0) | (chosen >= entries)
    return torch.where(invalid, torch.full_like(chosen, -1), chosen)


def selection_mask_from_indices(
    indices: torch.Tensor,
    entries: int,
) -> torch.Tensor:
    """Turn ``[batch, tokens, k]`` picks into a ``[batch, tokens, entries]`` bool mask.

    The reference builds an additive ``-inf``/``0`` bias the same way
    (``DeepseekV4CSACompressor.forward``): scatter into a buffer one column wider
    than the entry axis so the ``-1`` sentinels have somewhere to land, then drop
    that column. Keeping the sentinel column is what avoids a conditional
    scatter, and the extra column is a compile-time constant.

    A bool mask rather than an additive bias because this plugin's attention
    takes a ``key_valid`` mask; the two are the same statement.
    """
    if indices.ndim != 3:
        raise ValueError("indices must be [batch, tokens, k]")
    if entries < 0:
        raise ValueError("entries must be non-negative")

    # Cast and clamp before anything indexes with these, the same discipline
    # ``gather_paged_latent`` and ``scatter_paged_latent`` apply: an index that
    # reaches an indirect memory op must be in range by construction, not by
    # assumption about who produced it. An unsigned index would make ``>= 0``
    # vacuously true and turn the sentinel into a very large positive offset
    # -- see :func:`select_compressed_entries`.
    indices = indices.long()
    in_range = (indices >= 0) & (indices < entries)
    safe = torch.where(in_range, indices, torch.full_like(indices, entries))
    padded = torch.zeros(
        (*indices.shape[:2], entries + 1), dtype=torch.bool, device=indices.device
    )
    padded.scatter_(-1, safe, torch.ones_like(safe, dtype=torch.bool))
    return padded[..., :entries]
