# DeepSeek-V4: SWA/carry-state history reads stale (null) blocks past one window

> **Status: confirmed, unfixed.** A real correctness bug, found while working
> the Dynamo/FakeTensor blockers in
> [`deepseek-v4-024-device-validation.md`](deepseek-v4-024-device-validation.md)'s
> Step 5d — independent of that work, but touches the same code, so read that
> doc's Step 5d before starting a fix (see "Relationship to the Dynamo work"
> below). Not caught by any existing test; a regression test now documents it
> as `xfail` (`test/unit/model/deepseek_v4/test_paged_cache_helpers.py::
> test_gather_paged_latent_reads_stale_columns_once_swa_window_has_advanced`),
> and a standalone repro lives at
> `tools/deepseek_v4/check_swa_null_block_bug.py`.

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

## Suggested fix direction (not attempted)

1. **Correctness first, independent of Dynamo.** Change
   `gather_paged_latent` (or add a variant) to read the columns that
   actually cover the live window: start column
   `max(0, cached_seq_len - sliding_window) // block_size`, for `required =
   ceil(sliding_window / block_size)` columns — not `block_table[:required]`
   from column 0. Verify against the new xfail test (flip it to a normal
   assertion once it passes) and extend
   `test_attention_matches_real_module_through_paged_cache_io` with a
   decode-past-one-window case compared against the real HF module, since
   the current test's parameters (`sliding_window=128`, `tokens=5`) would
   not have caught this and won't catch a regression either.
2. **Then (or alongside), make it Dynamo-shape-static**, per Step 5d: gather
   a fixed, compile-time-constant number of columns (`required`, computed
   from `self.sliding_window`, a config constant — never from
   `cached_seq_len`) starting at the *dynamic* offset above, computed as a
   tensor (not a Python int fed through `int()`), and rely on masking
   (`mla_attention_reference`'s existing `kpos`/`qpos` position-based
   masking, extended to also exclude rows before the true live start
   position — today it assumes the supplied tensor's length *is* the valid
   length, which stops holding once the gather is always a fixed size)
   rather than a variable-length slice to represent "how much of this
   window is real." `_carry_rows` needs the same treatment for
   `self.state_cache`.
3. Re-run `tools/deepseek_v4/check_swa_null_block_bug.py` and the new xfail
   test as the correctness gate; re-run Step 5d's device attempt (see that
   doc for the exact `vllm.LLM()` invocation and required env vars) as the
   Dynamo gate. Both need to pass; neither alone is sufficient.

## Severity

Any real serving session whose generation exceeds one `sliding_window`'s
length of tokens — the common case, not an edge case — would get silently
wrong attention for every SWA layer (and wrong carry-state replay for the
compressor) once this ships. This should block any correctness sign-off for
SWA/compressor-carry layers, independent of the compiled-serving Dynamo
work, which is still separately blocked on this same code (see above).
