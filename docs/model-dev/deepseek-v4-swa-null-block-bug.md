# DeepSeek-V4: SWA/carry-state history reads stale (null) blocks past one window

> **Status: correctness fixed. `_swa_history` AND `_carry_rows` (the
> compressor's carry state) are now both Dynamo-shape-static and confirmed
> past their blockers on real Trn2 hardware. Tracing now advances one call
> further, to a new, separate, simpler blocker in `_compressed_history` —
> not part of this bug's scope (that cache group has no correctness issue),
> documented at the bottom of item 3 below as the next open item.**
>
> **Correctness (`gather_paged_latent`'s `start_token`).**
> `gather_paged_latent` (`vllm_neuron/model/deepseek_v4/attention.py`) now
> takes a `start_token` parameter; `_swa_history`/`_carry_rows` (`model.py`)
> passed `start_token=max(0, cached_seq_len - window)` so the read starts at
> the live window's true column instead of always column 0. Verification:
> `python tools/deepseek_v4/check_swa_null_block_bug.py` prints `PASS`; the
> regression test
> (`test/unit/model/deepseek_v4/test_paged_cache_helpers.py::
> test_gather_paged_latent_reads_stale_columns_once_swa_window_has_advanced`)
> is no longer `xfail`; and a new integration test,
> `test/vllm_neuron/test_deepseek_v4_matches_real_architecture.py::
> test_attention_matches_real_module_after_swa_eviction_past_one_window`,
> drives the device path past one window (using vLLM's real
> `KVCacheManager` for a genuinely null-remapped block table) against the
> real HF module — confirmed to fail on the pre-fix code (94/96 elements
> wrong) and pass on the fix.
>
> **`_swa_history` made Dynamo-shape-static (closes its specific Step 5d
> blocker).** `_swa_history` was redesigned around a new
> `gather_recent_window` helper (`attention.py`): always gathers exactly
> `sliding_window` rows ending at (and including) the query's own absolute
> position — a tensor-derived dynamic offset feeding fixed-size advanced
> indexing, never a Python-int-length slice — plus a `[sliding_window]`
> validity mask for the leading rows that don't exist yet in the first
> `sliding_window` tokens of a request. `mla_attention_reference` gained a
> matching `key_valid` parameter to apply that mask. `cached_seq_len`'s
> Python-int form is no longer used here at all; `position_ids` (already a
> real tensor end to end, from the earlier position_ids Dynamo fix) is
> reused directly as the window's end position. Confirmed on real Trn2
> hardware: re-running Step 5d's `vllm.LLM()` recipe
> (`enforce_eager=False`) against this fix advances cleanly past the
> `_swa_history` guard failure that blocked every previous attempt, onto a
> new, later blocker (below) — a CPU `torch.compile(fullgraph=True,
> dynamic=True)` proxy on an isolated sliding-window-only config also
> confirms this directly (no graph break at all, across multiple steps
> spanning past one window).
>
> **`_carry_rows` made Dynamo-shape-static too, closing this document's
> remaining item — confirmed on real Trn2 hardware.** The blocker
> (`_carry_rows`'s `if gather_n == 0:`, `model.py`) looked entangled with
> the compressor's internal chunking math, and it was — but the cascade
> turned out smaller than feared once a key fact was found:
> `DeepseekV4Compressor.forward` has exactly **one call site** in the whole
> tree, inside `_forward_one_token`'s per-token loop, so it always
> compresses exactly **one new raw token**, never a multi-token chunk. That
> collapses the write-side risk this item originally flagged (see below).
>
> The fix: `_carry_rows` now gathers a fixed `coff*ratio - 1` row window via
> `gather_recent_window` (the same helper item 2 built), ending one token
> before the new one, plus a `carry_valid` mask combining
> `gather_recent_window`'s own "doesn't exist yet" mask with a new tensor-
> valued `carry_gather_length_tensor` (`compressor.py`) marking rows already
> consumed by an earlier call. `compress_hca_chunk`/`compress_csa_chunk`
> gained an optional `carry_valid` parameter (default `None`, unchanged
> behavior for every existing caller) that neutralizes invalid rows via
> their existing gate-softmax (`-inf`) *before* the windowing reshape —
> masking pre-reshape, not post, is what makes CSA's overlap-half copy
> (`combined_gate[:, 1:, :ratio] = logits[:, :-1, ...]`) propagate
> invalidity correctly with no CSA-specific handling needed. Because the
> carry window is now always exactly `coff*ratio` rows total (fixed), the
> two compress functions always produce exactly `coff` candidate rows for
> this caller (a plain Python int, not a traced value) and the
> currently-completing window (if any) is unconditionally the *last* one —
> so the original "write-side `valid_slots[:write_count]` slot filtering" and
> "which computed entry is genuine" risk this item warned about doesn't
> arise: `DeepseekV4Compressor.forward` now just takes `compressed[:, -1:]`
> unconditionally and lets `scatter_paged_latent`'s existing `slot_mapping
> == -1` filtering (already Dynamo-safe) decide whether to write it. The
> compressed-entry RoPE position math also switched from the Python-int
> `cached_seq_len // self.ratio` to the tensor `position_ids`, the same
> swap item 2 made for `_swa_history`.
>
> Verified in stages, per this item's own original caution against rushing:
> new CPU-eager oracle tests first (bit-exact cross-checks of the masked
> fixed-window replay against the already-validated slice-based path across
> full token-by-token walks, including CSA's all-masked-first-window case; a
> new eviction-past-carry-window regression test driving `_carry_rows`
> itself against a real paged cache with real null-block eviction, mirroring
> `_swa_history`'s coverage; and a full real-`transformers`-module
> comparison through real paged cache I/O, past eviction, for both HCA and
> CSA) — all green — then a CPU `torch.compile(fullgraph=True, dynamic=True)`
> proxy confirming `_carry_rows`/`gather_n`/`compress_hca_chunk`/
> `compress_csa_chunk` no longer appear anywhere in the trace, then the same
> real-hardware `vllm.LLM()`/`enforce_eager=False` recipe as every attempt
> above, re-run on the same `trn2.3xlarge`: tracing reaches the real compile
> backend and advances cleanly past `_carry_rows`, landing on a new,
> separate, simpler blocker in `_compressed_history` (`model.py:552`,
> `cached_seq_len // self.ratio` — that cache group never evicts, so it has
> no correctness bug like this document's; it just needs the same
> fixed-length-gather-and-mask shape fix, not entangled with any windowing
> or write-side complexity) — not part of this document's scope, the next
> open item. Artifacts:
> `artifacts/deepseek-v4/20260819T022008Z-8df62fb/
> p6-null-block-fix-step5d-retry/` (correctness-only fix, confirms
> `_swa_history`'s *old* blocker unchanged),
> `artifacts/deepseek-v4/<run-id>/p6-dynamo-shape-static-swa/` (`_swa_history`
> redesign: one log confirming its blocker is gone and landing on
> `_carry_rows`; a second confirming the `carry_gather_length` guard fix and
> landing on the same `_carry_rows` line), and
> `artifacts/deepseek-v4/20260819T035235Z-2f88686-wip/p6-carry-rows-dynamo-static-fix/`
> (this fix's CPU proxy and real-hardware confirmation, both landing
> cleanly on `_compressed_history`, past `_carry_rows` — directory name
> notes this ran against an uncommitted working tree, see its
> `git-revision.txt`).

## One-paragraph summary

`gather_paged_latent` (`vllm_neuron/model/deepseek_v4/attention.py`) always
reads a request's block table starting from column 0. That reconstructs the
correct recent history only while nothing has been evicted yet from a
sliding-window cache group. Real vLLM's `SlidingWindowManager` never
compacts a block table when it evicts — it remaps old, low-index columns to
a shared null block *in place*, while the live window's real data keeps
living at ever-higher column indices as generation continues. So once a
sequence has run for more than one `sliding_window`'s worth of tokens,
`gather_paged_latent`'s "read the first N columns" strategy silently returns
null-block content instead of the true recent window. This is the common
case for any real generation, not an edge case.

## Reproduction

```bash
python tools/deepseek_v4/check_swa_null_block_bug.py
```

```
cached_seq_len=10 sliding_window=4
block_table_row: [0, 0, 0, 4, 5, 0, 0, 0]
gathered values: [-999.0, -999.0, -999.0, -999.0]
expected values: [6.0, 7.0, 8.0, 9.0]
FAIL -- gather_paged_latent returned null-block content instead of the true
recent window.
```

`block_table_row` here is built to look exactly like what real vLLM produces
at `cached_seq_len=10` with `sliding_window=4`, `block_size=2`: columns 0-2
(tokens 0-5, entirely before the live window `[6,10)`) are remapped to the
null block (physical id `0`, matching this plugin's own
`test_sliding_window_remapping_uses_null_blocks_but_latents_remain_stable`,
which asserts evicted columns become `(0, 0)`); columns 3-4 (tokens 6-9,
still live) hold real distinct physical blocks. `gather_paged_latent` is
asked for the last 4 tokens of history and returns four rows of the null
block's sentinel value (`-999.0`, planted there specifically to make a wrong
read unmistakable) instead of the real values `[6, 7, 8, 9]`.

The same scenario is captured as a proper (xfail, `strict=True`) regression
test in `test/unit/model/deepseek_v4/test_paged_cache_helpers.py::
test_gather_paged_latent_reads_stale_columns_once_swa_window_has_advanced` —
run `pytest test/unit/model/deepseek_v4/test_paged_cache_helpers.py -v` to
see it XFAIL. `strict=True` means the suite will fail loudly the moment a fix
makes it pass, as a forcing function to remember to delete the marker.

## Root cause, in detail

`gather_paged_latent(cache, block_table, sequence_length)`:

```python
slots_per_block = cache.shape[2]
required = math.ceil(sequence_length / slots_per_block) if sequence_length else 0
blocks = block_table[:required].long()
gathered = cache[blocks].permute(0, 2, 1, 3).reshape(-1, cache.shape[1], cache.shape[3])
return gathered[:sequence_length]
```

`block_table[:required]` — always the *first* `required` columns of the
row, regardless of where in the sequence the request currently is.

Two facts about real vLLM's block-table management make that wrong once
eviction has happened:

1. **The row never shrinks or compacts.** This plugin's own
   `max_num_blocks_per_req` derivation
   (`vllm_neuron/vllm/worker/input_batch_params.py:164-166`,
   `cdiv(max_model_len, spec.block_size * total_cp_world_size)`) sizes every
   group's row — SWA groups included — for the *entire* `max_model_len`, not
   just one window. It says explicitly this mirrors `MultiGroupBlockTable`'s
   own real-vLLM default. Column `i` always addresses logical block position
   `i` (tokens `[i*block_size, (i+1)*block_size)`); that mapping never moves.
2. **Eviction remaps old columns to a null block, in place.** Real vLLM's
   `SlidingWindowManager` (`vllm/v1/core/single_type_kv_cache_manager.py`):
   `get_num_skipped_tokens(num_computed_tokens)` returns
   `max(0, num_computed_tokens - sliding_window + 1)` — the count of tokens
   (from the start) that have fallen out of the window and get skipped. Its
   docstring's own example (`sliding_window=4, num_computed_tokens=7`):
   tokens 0-3 are skipped, tokens 4-7 are "the current window." Skipped
   tokens' *columns* get remapped to `block_pool.null_block`, but the
   columns for tokens 4-7 stay exactly where they always were — at their own
   `token // block_size` index, which keeps growing as generation continues.
   `find_longest_cache_hit`'s own comment shows the same shape directly:
   `[NULL, NULL, block 3]` — nulled low columns, real data at a higher one.

So `block_table[:required]` is correct only while `cached_seq_len` (before
this call) is small enough that the live window still starts at column 0 —
i.e., while `cached_seq_len <= sliding_window` (roughly). Past that, the
live window's real data has moved to columns starting around
`(cached_seq_len - sliding_window) // block_size`, and columns
`[0:required]` are entirely null.

## Affected call sites

Both call `gather_paged_latent` against a cache group that uses this same
null-block eviction lifecycle:

- **`DeepseekV4Attention._swa_history`** (`model.py:472`) — the raw
  sliding-window KV cache (`CacheKind.SLIDING_WINDOW` →
  `SlidingWindowSpec` in `kv_spec_conversion.py:55-58`). This is the bug
  reproduced above.
- **`DeepseekV4Compressor._carry_rows`** (`model.py:207`) — the compressor's
  `state_cache` (`CacheKind.COMPRESSOR_STATE` → `SlidingWindowMLASpec` in
  `kv_spec_conversion.py:45-54`, which is also an evicting sliding-window
  spec). The code's own comment already (incorrectly) assumed this was
  handled: *"the scheduler remaps blocks older than that to a null block
  once evicted, same as `DeepseekV4Attention._swa_history`"* — true about
  the eviction, but the gather was never actually made robust to it.

**Not affected: `DeepseekV4Attention._compressed_history`** (`model.py:493`),
which reads `self.mla_cache` — `CacheKind.MLA` → `MLAAttentionSpec`
(`kv_spec_conversion.py:33-44`), a **non-evicting** spec (compression is the
memory strategy for this group; nothing ever gets nulled). Its `num_entries
= cached_seq_len // self.ratio` grows without a window cap, and
`block_table[:required]` genuinely does hold the entire unevicted history
from column 0 onward for this group, so this call site is correct as
written.

## Why nothing caught this

Every existing test that exercises `_swa_history`/`_carry_rows` through a
real gather stays within `cached_seq_len <= sliding_window`:

- `test_attention_matches_real_module_through_paged_cache_io`
  (`test/vllm_neuron/test_deepseek_v4_matches_real_architecture.py`) uses
  `sliding_window=128` with only `tokens=5`, and its `block_table` is
  `torch.arange(8)` — every column a distinct, valid block; the null-block
  eviction path is never exercised in this test at all.
- `test_sliding_window_remapping_uses_null_blocks_but_latents_remain_stable`
  (`test/vllm_neuron/test_deepseek_cache_lifecycle.py`) tests that
  `KVCacheManager.remove_skipped_blocks` produces the right *remapping*
  (columns become `(0, 0)`) — it never calls `gather_paged_latent`/
  `_swa_history` afterward to check that a *consumer* of that remapped table
  still reads the right data.
- `test_deepseek_v4_device_e2e.py`'s `generate()` call uses `max_tokens=4`
  against a `sliding_window=16` config — again short of one window.

No test in the repository drives a decode past `sliding_window` tokens
through the actual gather path. That is the coverage gap this bug lived in.

## Relationship to the Dynamo/FakeTensor work

This was found while designing the fix for
[`deepseek-v4-024-device-validation.md`](deepseek-v4-024-device-validation.md)'s
Step 5d third-blocker-in-a-row: `_swa_history`'s `if cached_seq_len == 0:`
and its `gather_len = min(cached_seq_len, self.sliding_window)` are
data-dependent Python control flow/shape that Dynamo can't trace as-is
(`Could not guard on data-dependent expression`). Making that shape static
requires touching exactly the column-selection logic this bug lives in — so
**fixing the shape problem and the correctness problem are naturally one
piece of work**, not two independent ones. Do not paper over the shape issue
with a change that keeps reading `block_table[:required]` (that would make
the Dynamo blocker go away while leaving this bug in place, just harder to
find). Any redesign needs to compute a *dynamic starting column* — not just
a dynamic length — from `cached_seq_len`, and do so in a way Dynamo can
still trace (a tensor-derived offset feeding a *fixed-size* gather, rather
than a variable-length one).

## Suggested fix direction

1. **Correctness first, independent of Dynamo — done.** `gather_paged_latent`
   now takes a `start_token` keyword (absolute token offset to start
   reading at, default 0 = old behavior); `_swa_history`/`_carry_rows` pass
   `start_token=max(0, cached_seq_len - window)`. Implemented slightly more
   generally than the `start column // block_size` sketch originally here:
   `gather_paged_latent` computes `start_col`/`end_col` from
   `(start_token, sequence_length)` together and trims both ends, so it
   naturally reduces to "read from column 0" whenever nothing has been
   evicted yet, for any `(start_token, sequence_length)` pair, without a
   separate code path. `_compressed_history` is untouched (its default
   `start_token=0` is exactly its old behavior — correct, since its cache
   group never evicts). Verified: the xfail test is now a normal, passing
   assertion, `tools/deepseek_v4/check_swa_null_block_bug.py` prints PASS,
   and a new test,
   `test_attention_matches_real_module_after_swa_eviction_past_one_window`
   (`test_deepseek_v4_matches_real_architecture.py`), drives a
   decode-past-one-window case (using vLLM's real `KVCacheManager` for a
   genuinely null-remapped block table) against the real HF module — it was
   confirmed to fail against the pre-fix code and pass against the fix. The
   existing `sliding_window=128, tokens=5` test is left as-is (still valid,
   just not the eviction case).
2. **Making `_swa_history` Dynamo-shape-static, per Step 5d — done.** New
   helper `gather_recent_window` (`attention.py`): always gathers exactly
   `sliding_window` rows ending at (and including) a tensor `end_position`
   (fixed-size advanced indexing at a tensor-derived offset, never a
   Python-int-length slice), plus a `[sliding_window]` bool validity mask
   for rows before generation has produced them yet. `_swa_history` now
   returns `(window, valid)` built this way, using `position_ids` (already
   a real tensor, from the earlier position_ids Dynamo fix) as
   `end_position` instead of the Python-int `cached_seq_len`. Since the
   query's own token is already scattered into `swa_cache` by the time
   `_swa_history` runs, the returned window directly *is* the attention
   history -- no separate `cat` of the fresh token afterward, unlike
   before. `mla_attention_reference` gained a `key_valid` parameter
   (applied as an extra AND into its existing causal `allowed` mask) to
   consume the validity mask. Verified two ways: a CPU
   `torch.compile(fullgraph=True, dynamic=True)` proxy on an
   isolated sliding-only config traces cleanly across multiple steps
   spanning past one window (no graph break at all); and, on real Trn2
   hardware, re-running Step 5d's actual `vllm.LLM()` recipe against this
   fix advances cleanly past the exact guard failure that blocked every
   previous attempt, onto a new, later blocker (item 4).
3. **Making `_carry_rows` Dynamo-shape-static, per Step 5d — done, confirmed
   on real Trn2 hardware.** The cascade this item originally warned about
   turned out smaller than feared: `DeepseekV4Compressor.forward` has
   exactly one call site (`_forward_one_token`'s per-token loop), so it
   always compresses exactly one new raw token, never a multi-token chunk.
   `_carry_rows` now gathers a fixed `coff*ratio - 1` row window via
   `gather_recent_window` (the same helper item 2 built) plus a new tensor-
   valued `carry_gather_length_tensor` (`compressor.py`) to build a
   `carry_valid` mask; `compress_hca_chunk`/`compress_csa_chunk` gained an
   optional `carry_valid` parameter (default `None`, unchanged behavior for
   every existing caller) that neutralizes invalid rows via their existing
   gate-softmax (`-inf`) *before* the windowing reshape -- this ordering is
   what makes CSA's overlap-half copy propagate invalidity correctly with
   no CSA-specific handling needed. Because the carry window is now always
   exactly `coff*ratio` rows, both compress functions always produce
   exactly `coff` candidate rows for this caller (a plain Python int, not a
   traced value), and the currently-completing window (if any) is
   unconditionally the *last* one -- so the "write-side
   `valid_slots[:write_count]` slot filtering" / "which entry is genuine"
   risk this item warned about doesn't arise: `DeepseekV4Compressor.forward`
   just takes `compressed[:, -1:]` unconditionally and lets
   `scatter_paged_latent`'s existing `slot_mapping == -1` filtering decide
   whether to write it. Verified with dedicated oracle comparisons first
   (per this item's own original caution), then a CPU `torch.compile`
   proxy, then real hardware -- see the status block at the top of this
   document for the full account and artifact paths.
4. `tools/deepseek_v4/check_swa_null_block_bug.py` and the (now non-xfail)
   regression test are the correctness gate — green as of item 1.
   `tools/deepseek_v4/check_carry_rows_dynamo_trace.py` (new) is item 3's
   CPU-only Dynamo tracing gate. Step 5d's device attempt (see that doc for
   the exact `vllm.LLM()` invocation and required env vars) is the Dynamo
   gate: items 2 (`_swa_history`) and 3 (`_carry_rows`) are both now
   confirmed to pass their slice of that gate; the attempt now fails one
   call later, at `_compressed_history`'s own `cached_seq_len //
   self.ratio` (`model.py`) -- a separate, simpler item (that cache group
   never evicts, so it has no correctness bug like this document's), not
   tracked here. All of items 1-3 have landed; item 1 alone (or 1+2) was
   not sufficient for compiled-serving sign-off, but items 1-3 together are
   not sufficient either -- `_compressed_history` is the next, and (as far
   as this document's investigation found) last, blocker in this family.

## Severity

Any real serving session whose generation exceeds one `sliding_window`'s
length of tokens — the common case, not an edge case — would have gotten
silently wrong attention for every SWA layer (and wrong carry-state replay
for the compressor). Fixed as of item 1 above (correctness). The
compiled-serving Dynamo work is now fully done for the code this document
covers (item 2, `_swa_history`; item 3, `_carry_rows` and the compressor's
internal chunking) — see "Relationship to the Dynamo work" above. Tracing
now advances past all of it, to a separate, simpler, not-yet-attempted item
in `_compressed_history` (`model.py`) outside this document's scope.
