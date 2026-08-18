# DeepSeek-V4 device path: compressed-cache addressing and the carry-cache design

How `vllm_neuron/model/deepseek_v4/model.py`'s attention and compressor
modules address the paged compressed-MLA cache and the compressor
carry-cache, and why. This is the "no precedent in this plugin" piece
`docs/model-dev/deepseek-v4-serving-roadmap.md` flags for Step 2 — the design
here is new, not carried over from an existing pattern, so it is written down
the way `dense_csa.py`'s module docstring writes down its own derivation.

## Compressed-entry slot addressing

`vllm_neuron/vllm/worker/neuron_model_runner.py`'s `_build_attention_metadata`
computes `slot_mapping` the same way for every KV cache group — generic,
uniform, unaware of `compress_ratio`. That generic slot already encodes the
right physical block via `block_table` and a raw-token-scale offset
(`blk_idx = slot // block_size`, `pos_idx = slot % block_size`, exactly
`llama3/model.py`'s `_write_kv_cache` convention). Since a compressed MLA
group's page always covers `block_size` *raw* tokens in
`storage_block_size = block_size // compress_ratio` physical rows
(`kv_spec_conversion.py`, `mla_cache_shape` in `input_batch_params.py`), and
`layer_spec_to_vllm_spec` requires `block_size % compress_ratio == 0`, the
raw slot's low bits already equal the absolute token position's low bits mod
`compress_ratio`.

That means a raw token completes a compressed window exactly when
`(raw_slot + 1) % compress_ratio == 0`, and its physical storage row is
`raw_slot // compress_ratio` — ported directly from vLLM 0.24's own
first-party DeepSeek-V4 GPU backend,
`vllm/v1/attention/backends/mla/sparse_swa.py::_compressed_slot_mapping_kernel`,
which computes the identical thing from equivalent generic inputs
(`query_start_loc`, `seq_lens`, `block_table`). It is plain integer
arithmetic, not a GPU-specific optimization, so it ports to Neuron directly —
see `attention.py::compressed_entry_slot_mapping`.

`get_kv_spec()` sets `block_size=128` explicitly on every compressed MLA
group (`CacheKind.MLA`) rather than inheriting the Neuron platform's default
scheduler block size of 32 — 32 is not divisible by `compress_ratio=128`, and
`platform.py::register_custom_kv_cache_specs`'s own required
`MLAAttentionSpec` fixtures all use `block_size=128`, which is the
cross-validation this choice is grounded in.

## Compressor carry-cache

`compressor.py`'s `compress_hca_chunk`/`compress_csa_chunk` are already
correct, chunk-invariant, and oracle-verified (T0/T1) using an explicit
`GatedCompressorState` object threaded by the caller. The device path cannot
thread a Python object across scheduler-driven forward calls the way
`tiny_model.py` does — the carry has to live in the paged
`CacheKind.COMPRESSOR_STATE` cache instead.

Rather than serializing `GatedCompressorState`'s fields into cache rows, the
device path stores the **raw per-token `[kv, gate]` projection** (the
`fused_wkv_wgate` output, pre-norm, pre-window-reduction) under an ordinary
sliding-window lifecycle (`SlidingWindowMLASpec`, window = `coff * ratio`,
matching `get_kv_spec`'s declared `sliding_window_size`). This works because
`compress_hca_chunk`/`compress_csa_chunk`'s own state-based carry is nothing
more than raw unconsumed rows plus, for CSA, one extra window kept purely to
re-derive `overlap_kv`/`overlap_gate` — and CSA's own math derives those
*from* raw rows (`windows[:, -1, :, :head_dim]`), not from a separately
maintained reduction. So gathering the right number of trailing raw rows and
replaying them through the stateless function with `state=None` is
numerically identical to the incremental form, by `_join_gated_carry`'s own
`torch.cat((state.kv_carry, kv), dim=1)`.

`compressor.py::carry_gather_length(cached_seq_len, ratio, needs_overlap)`
computes exactly how many trailing rows to gather:

- HCA (`needs_overlap=False`): `cached_seq_len % ratio` — always
  `< ratio`, so it can never itself contain a complete window.
- CSA (`needs_overlap=True`) once at least one window exists
  (`cached_seq_len >= ratio`): the unconsumed suffix *plus* one full
  previous window, `min(cached_seq_len, cached_seq_len % ratio + ratio)`
  — bounded by the declared window (`coff * ratio` when `coff=2`), so the
  gather never walks into blocks the sliding-window lifecycle has already
  evicted/remapped to a null block (same concern
  `DeepseekV4Attention._swa_history` documents for the SWA cache).

Because the CSA case prepends one already-fully-emitted window purely to
reconstruct overlap, `compressor.py::carry_replay_already_emitted` says how
many leading output rows to drop from the replay before writing — 1 exactly
when that prepended window is present, 0 otherwise.

`test/unit/model/deepseek_v4/test_paged_cache_helpers.py` cross-checks this
replay-based design against incremental state-based chunking directly, for
arbitrary chunk boundaries including exact multiples of `ratio` (the edge
case most likely to hide an off-by-one).

### RoPE is deliberately not applied to compressed entries in this pass

`compressor.py::finalize_compressed_entries` (RMSNorm + partial RoPE) is
exercised in this pass's RoPE-degenerate mode (`rope_dim=0`, via an empty
trailing `cos`/`sin` dimension) rather than with a real rotary table. The
device path's attention keeps the same single-global-head, no-RoPE structure
`tiny_model.py` already established as the T0 oracle (see `model.py`'s module
docstring); RoPE-encoding only the compressed cache without a matching
RoPE-aware query would be a real numerical inconsistency, not a partial
improvement. Wiring the real multi-head q_lora/kv_lora/partial-RoPE MLA
attention is follow-up work.

## Mid-chunk compression boundaries force a per-token attention loop

A single multi-token prefill chunk can span a compression-window boundary:
tokens after the boundary need to see the compressed entry a token *earlier
in the same chunk* just completed. A flat batched history built once for the
whole chunk cannot express that — compression boundaries are positional, not
chunk-aligned, and there is no single causal cutoff that is correct for both
the "compressed" and "still local" portions simultaneously once part of the
chunk is on each side of a boundary.

`DeepseekV4Attention.forward` therefore loops over tokens one at a time (see
`_forward_one_token`) even during prefill, reusing exactly the same
read-then-write-then-attend path decode already needs. This was found by a
real chunk-invariance failure (see `test_deepseek_v4_model_assembly.py`'s
`test_batched_forward_is_chunk_invariant_against_single_shot`): a
whole-chunk-batched first draft of this code diverged from single-token
decode by up to 76% relative error at a mid-sequence position, not
floating-point noise. It is a real throughput cost, not a performance nuance
— the mHC/MoE stages remain batched across a chunk; only attention's
cache-dependent portion is per-token. Fine for this pass's correctness goal;
a real batched/graph-capturable version needs more than swapping the loop
back, since the boundary-crossing case requires attending each query against
a *different* prefix of "what's compressed so far."
