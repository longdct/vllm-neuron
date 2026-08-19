# DeepSeek-V4: SWA/carry-state history reads stale (null) blocks past one window

> **Status: correctness fixed; Dynamo-shape-static redesign still open.**
> `gather_paged_latent` (`vllm_neuron/model/deepseek_v4/attention.py`) now
> takes a `start_token` parameter, and `_swa_history`/`_carry_rows`
> (`model.py`) pass `start_token=max(0, cached_seq_len - window)` so the read
> starts at the live window's true column instead of always column 0. This
> was a pure correctness fix (still a Python-int `cached_seq_len`, same
> dynamic-shape behavior as before) — it does **not** attempt the
> Dynamo-shape-static redesign sketched in "Suggested fix direction" item 2
> below (fixed-size gather + extended `mla_attention_reference` masking).
> Step 5d in
> [`deepseek-v4-024-device-validation.md`](deepseek-v4-024-device-validation.md)
> (the compiled-serving Dynamo blocker that shares this code) is therefore
> **still open** — this fix only closes the correctness half of that
> section's coupled work, not the tracing half.
>
> Verification: `python tools/deepseek_v4/check_swa_null_block_bug.py` now
> prints `PASS`; the regression test
> (`test/unit/model/deepseek_v4/test_paged_cache_helpers.py::
> test_gather_paged_latent_reads_stale_columns_once_swa_window_has_advanced`)
> is no longer `xfail`; and a new integration test,
> `test/vllm_neuron/test_deepseek_v4_matches_real_architecture.py::
> test_attention_matches_real_module_after_swa_eviction_past_one_window`,
> drives the device path past one window (using vLLM's real
> `KVCacheManager` to produce a genuinely null-remapped block table) and
> checks the post-eviction step against the real HF module — it fails
> against the pre-fix code (94/96 elements wrong) and passes against the
> fix.
>
> **Confirmed the Dynamo blocker is genuinely unchanged, on real Trn2
> hardware.** First a CPU-only proxy (`torch.compile(fullgraph=True,
> dynamic=True)` on the default `eager` backend) reproduced the identical
> attempt-5 error at the identical line. Then, on an actual `trn2.3xlarge`
> instance, Step 5d's own `vllm.LLM()` recipe was re-run against this fix
> (`enforce_eager=False`, real `libtorch_neuronx_lite` parallel-trace-fork
> compile, `VLLM_NEURON_ENABLE_DEEPSEEK_V4=1`,
> `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, `NEURON_LOGICAL_NC_CONFIG=2`,
> `NEURON_SKIP_EFA_AFFINITY=1`) and hit the exact same
> `Could not guard on data-dependent expression Eq(u0, 0)` at
> `_swa_history`'s `if cached_seq_len == 0:` (`model.py:498`), this time from
> `RuntimeError: Parallel trace fork failed (rank=0): ... status=ERROR`, the
> real compile backend's own error surface, not just Dynamo's. This is now
> the authoritative confirmation, not a proxy: the correctness fix does not
> touch Step 5d's Dynamo-shape blocker at all, on real silicon. Artifact:
> `artifacts/deepseek-v4/20260819T022008Z-8df62fb/
> p6-null-block-fix-step5d-retry/attempt-post-null-block-fix.log`.

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
2. **Making it Dynamo-shape-static, per Step 5d — still open, not
   attempted.** This fix deliberately kept `cached_seq_len` a Python int and
   the gather variable-length (same shape behavior as before), so Step 5d's
   `Could not guard on data-dependent expression Eq(u0, 0)` blocker on
   `_swa_history`'s `if cached_seq_len == 0:` remains. That redesign still
   needs to gather a fixed, compile-time-constant number of columns (`required`, computed
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
3. `tools/deepseek_v4/check_swa_null_block_bug.py` and the (now non-xfail)
   regression test are the correctness gate — both green as of item 1.
   Step 5d's device attempt (see that doc for the exact `vllm.LLM()`
   invocation and required env vars) remains the separate Dynamo gate for
   item 2, still open. Both need to pass before compiled-serving sign-off;
   correctness alone is not sufficient for that.

## Severity

Any real serving session whose generation exceeds one `sliding_window`'s
length of tokens — the common case, not an edge case — would have gotten
silently wrong attention for every SWA layer (and wrong carry-state replay
for the compressor). Fixed as of item 1 above (correctness). The
compiled-serving Dynamo work is still separately blocked on this same code
(see "Relationship to the Dynamo work" above) — item 2 remains open.
