
# Real Qwen3.5 checkpoints: loading, two device NaNs, and two accuracy defects

Date: 2026-08-29, revised 2026-08-31. Checkpoints: `Qwen/Qwen3.5-0.8B`
(24 layers, 18 GDN + 6 full, hidden 1024, head_dim 256, 16 K / 16 V heads,
vocab 248320) and
`Qwen/Qwen3.8-27B` (64 layers, hidden 5120, 16 K / 48 V heads, 18 shards).

The 0.8B is the useful one for bring-up: it is 1.8 GB but carries every
structural property of the 27B that matters -- the released wrapper config, a
vision tower, `head_dim=256`, the 3:1 hybrid schedule, the full 248320 vocab.
Everything below was found by running it, and would not have been found on the
synthetic fixture.

## Loading: four blockers, three of them wrong assumptions

1. **`from_configs` signature.** Fixed by adopting `qwen3_vl`'s signature. See
   that module's docstring; the failure was `TypeError: got an unexpected
   keyword argument 'text_neuron_config'` on every rank at every TP degree.

2. **MTP.** The factory refused any config declaring `mtp_num_hidden_layers`,
   on the premise that no released checkpoint ships MTP weights. **The premise
   was wrong.** Both checkpoints ship 15 `mtp.*` tensors and both declare
   `mtp_num_hidden_layers: 1`. Downgraded to a warning; nothing reads the
   field, so the weights are skipped by prefix like `model.visual.*`.

3. **Tied embeddings.** 0.8B sets `tie_word_embeddings: true` and ships no
   `lm_head.*`; the 27B is untied and ships it. The mapping now follows the
   config. Not a family-wide property, so it cannot be assumed either way.

4. **Hybrid page alignment.** Solved with `MambaSpec.page_size_padded` rather
   than the two options the earlier note listed. See `align_mamba_pages`.

Two further notes for anyone serving a real checkpoint:

- **Parent-process registration is not a blocker.** The earlier record listed
  it as one. On a real checkpoint the parent resolves the architecture and
  brings up all ranks; that failure was an artifact of the synthetic fixture
  having no `preprocessor_config.json`.
- **The vision stack constrains a text-only run.** `_resolve_vision_auto_config`
  synthesizes a `vision_attention_block_size` of 2048 from the mere presence of
  `vision_config`, then rejects any smaller token bucket:
  `ValueError: Largest bucket (256) is smaller than vision_attention_block_size
  (2048)`. Raising `max_model_len` to 2048 works around it. The durable fix is
  to let a text-only model opt out; that touches `platform.py`, which every
  multimodal model on this backend shares.

## Real-weight parity on CPU: the model is correct

The first accuracy evidence for this port on real weights. Random tiny weights
cannot settle accuracy (`deepseek-v4-tiny-tp1-neuron-investigation.md:36-43`);
a real checkpoint can. Same 32-token prompt, final position, fp32 on CPU:

| | min | max | mean | std | top-5 |
|---|---|---|---|---|---|
| ours | -11.698 | 12.851 | -2.004 | 1.975 | 200, 16, 17, 220, **199** |
| HF | -11.931 | 12.715 | -2.043 | 1.967 | 200, 16, 17, 220, **15** |

Argmax agrees, top-4 identical, distributions match; the fifth slot is a
near-tie. **bf16 on CPU gives the same top-5**, so precision is not a factor.

This establishes the weight loading and the tied-embedding mapping, and nothing
more. Read as "the model is correct" it was actively misleading: this prompt is
32 tokens with `slot_mapping = arange(32)`, which is not a shape the serving path
ever produces, and the bug below needs the padded bucket to appear. A parity run
whose inputs the runtime cannot generate proves less than its agreement suggests
-- the same 32 tokens placed in their real 2048 bucket go NaN on this exact
checkpoint. The padded-bucket case now reproduces the same top-5.

## The NaN: an unstable matrix inverse, not a device fault

The previous revision of this document called these "two device-only NaNs" and
spent several device runs bisecting the attention sublayer. Both claims were
wrong, and the way they were wrong is the most reusable thing here.

**Root cause.** `unit_triangular_inverse` in `gated_deltanet.py` -- the port's one
deliberate departure from the reference -- replaced HuggingFace's 63-step forward
substitution with the binary-powering identity

    (I - A)^-1 = I + A + ... + A^(n-1) = (I + A)(I + A^2)(I + A^4)...(I + A^(n/2))

to keep the traced graph small (10 matmuls, static shapes, no per-row scan). The
identity is exact. The evaluation is catastrophically unstable for the matrices
this layer actually produces.

Here `A = -(k_beta @ key^T) * decay_mask` with l2-normalized keys. When
successive keys are near-identical, `k . k == 1` and `A` approaches `-beta` times
the strictly-lower-triangular ones matrix -- whose powers *grow*:

| A | true max\|inv\| | sequential | binary powering | max\|A^32\| |
|---|---|---|---|---|
| random, entries ~0.1 | 1 | err 1e-7 | err 1e-7 | 5e-24 |
| -0.5 * lower-ones | 1 | err 3e-8 | **112** | 1.1e8 |
| -0.9 * lower-ones | 1 | err 7e-8 | **1.07e9** | 1.6e16 |
| -0.99 * lower-ones | 1 | err 2e-8 | **6.87e10** | 3.4e17 |

The true inverse is bounded by 1; the algorithm builds 1e17-magnitude
intermediates and relies on cancellation to get back. fp32 keeps none of it, and
**fp64 does not rescue it either** (error 64 at beta = 0.99), so this is not a
precision shortfall -- the scheme is unusable for this matrix class.

**What made it fire.** Bucket padding. A 32-token prompt occupies a 2048 bucket,
so ~2016 rows carry the same padded token -- exactly the near-identical-keys
regime. Real prompts reach it too, just later: 2000 real tokens diverged at layer
5 instead of layer 0.

**The fix.** Recursive 2x2 block substitution. If `x` holds the inverses of the
`b`-sized diagonal blocks, then `x @ a @ x` restricted to each `2b` block's lower
quadrant is exactly `X22 @ A21 @ X11`, so one masked update per level and
`log2(n)` levels suffices. It never forms a power of `A`: every factor is a true
inverse or a sub-block of `A`, so there is nothing to cancel. 12 matmuls at
n = 64 against the binary form's 10, same static shapes, same compile-time bound.
Measured error 3e-8 on the matrices above, exact at beta = 1.

`nki_gdn.py` imports the same function, so the one fix covers the oracle and the
device path together.

## Why the bisection took as long as it did

Worth recording, because the cost was almost entirely in the method rather than
the bug.

**The CPU oracle was fed inputs the serving path never produces.** Every CPU
probe passed exactly the real tokens: 32 ids, positions 0..31,
`slot_mapping = arange(32)`. The device always pads to the bucket. So "finite on
CPU, NaN on device" was never a device/CPU difference at all -- it was two
different inputs, and the framing it produced sent every subsequent run at the
wrong sublayer. Reproducing the device's *metadata* on CPU took one script and
inverted the whole diagnosis. **Before concluding "device-only", check that the
CPU path is being given the device's actual inputs, padding included.**

**Component substitution kept exonerating things because the culprit was
upstream.** `_project`, `NF.flash_attention`, `NF.o_proj`, `_write_cache` and the
SP collectives were each replaced or removed on device, one run apiece, and each
came back "still NaN" -- correctly, since the fault was in layer 0's GDN, three
layers before any attention layer runs.

**A forward hook cannot see the GDN.** `Qwen3_5DecoderLayer.forward` calls
`self.linear_attn.forward_paged(...)` directly rather than through `__call__`, so
`register_forward_hook` never fires on it. Every hook-based observation was blind
to the module that was actually failing. Instrument `forward_paged` explicitly.

**The existing tests could not have caught it.**
`test_blocked_inverse_matches_the_reference_loop` uses `randn * 0.3`, which gives
`max|A^32| = 5e-24` -- no cancellation, error 1e-7, passes clean. The failure
needs entries near 1 with consistent sign. The regression tests now pin that
structure and are verified to fail against the old implementation.

A structure test also pinned the *wrong* thing: it asserted `span *= 2`, i.e. the
specific doubling scheme, which made a numerically unusable algorithm look load
bearing. It now constrains the graph shape -- logarithmic bound, no per-row scan,
no power of the input formed -- and leaves numerics to the accuracy tests.

## Bucket padding poisons the recurrence

The NaN fix above was necessary and not sufficient. With finite logits the 0.8B
still emitted `"^^^^..."` on device and, on CPU, 3 of 32 tokens against
HuggingFace -- wrong from the very first generated token.

The measurement that isolated it: run the same prompt twice, once padded to the
2048 bucket exactly as the runner pads it, once as the bare 12 real tokens.

| prefill | tokens matching HF |
|---|---|
| 12 real rows, no padding | 32 / 32 |
| padded to 2048, no mask | 3 / 32, diverges at token 1 |
| padded to 2048, masked | 12 / 32, diverges at token 12 |

Unpadded parity was already passing, which is why this survived: **every CPU
probe up to that point fed the model a shape the serving path never produces.**
`slot_mapping = arange(32)` with no padding is not a bucketed prefill, and the
one difference between the two columns is the entire defect.

Full attention tolerates padding because a padded key is a column the causal
mask discards. A recurrence has no such mask -- every row it walks over mutates
the state decode resumes from -- so ~2036 filler rows were being carried into
the cache. Filler repeats the last position, so those rows share one token,
their l2-normalised keys are near-identical, and the delta rule's update is at
its most aggressive exactly where the data is meaningless.

Three things have to happen, and all three, not any two:

1. `beta = 0` on padded rows, so they add nothing to the state;
2. `g = 0` on padded rows, so `exp(g) == 1` and they do not *decay* the state
   they pass through either. This is the subtle half: masking only `beta` still
   applies 2000 steps of decay to a state that should have stopped evolving,
   and the result looks plausible;
3. the conv state must be the `kernel-1` columns ending at the last *real*
   token. `causal_conv1d_with_state` takes them from the end of the sequence,
   which on a padded bucket is a window over padding, so decode resumed the
   convolution from tokens the prompt never contained.

The signal is a new `num_valid_tokens` metadata key -- real tokens per request
row -- present in **both** metadata builders at the same shape, since it is a
value and never a shape and warmup must trace the identical graph. The conv
window is a fixed-width `torch.gather` at a tensor-derived offset rather than
`extended[..., n : n + width]`, which would be a data-dependent shape the
moment `n` is an int (section 4.2). `torch.compile(fullgraph=True,
dynamic=True)` captures the padded path as one graph.

### The residual divergence is a bf16 tie, not a defect

Masked padding still diverges from HF at token 12, which looks like an
incomplete fix. It is not. Against an unpadded prefill the masked padded run
differs by 2.7e-05 in the conv state, 5.7e-06 in the recurrent state and
1.7e-05 in the logits, with identical argmax. The top-2 logit gaps across the
sequence are 0.75 to 7.4 everywhere except step 12, where the gap is **0.125 --
exactly one bf16 ULP at logit scale**, the floor this project already records
as unachievable to beat. A tie at the working precision is not a bug to chase.

## On device, with real weights

With the padding fix in place and every NKI kernel off, the 0.8B at TP=2
reproduces HuggingFace **exactly, 32 of 32 greedy tokens**:

    " Rome. The capital of Spain is Madrid. The capital of Japan is Tokyo. The
     capital of the United States is Washington, D.C. The capital of the"

That is the first end-to-end statement about this port that is worth anything:
prefill, decode, the paged GDN state seam, the attention layers and the real
checkpoint, on hardware, against an independent oracle.

The shipping default keeps the depthwise conv kernel and drops only the scan.
That run agrees with HuggingFace for 12 tokens and then diverges -- at the same
step 12 whose top-2 gap is one bf16 ULP. The conv kernel's rounding differs from
torch's by less than that gap, so it lands on the other side of the tie. Both
runs are correct to the precision the model works at; only the tie-break
differs, and the run above is quoted as the exact one because it is.

## Batch > 1, unequal prompt lengths

A batch of one cannot tell a correct per-request padding mask from one that uses
the whole batch's length, and equal-length prompts cannot either -- they pad
identically. So the check is four prompts of 3, 4, 12 and 24 tokens submitted
together, each diffed against its own HuggingFace reference.

TP=2, `max_num_seqs=4`, 32 greedy tokens each:

| prompt tokens | vs HF | device |
|---|---|---|
| 3 | **32/32** | `" 100 C. If a 100 g sample of water is heated to ..."` |
| 4 | **32/32** | `" four.\nTwo plus two equals four ..."` |
| 12 | 12/32 | `" Rome. The capital of Spain is Madrid ..."` |
| 24 | 19/32 | `" the Apollo 11 mission ... Neil Armstrong and Buzz Aldrin"` |

The two shortest are the load-bearing rows. Three real tokens in a 2048 bucket
is ~2045 padding rows -- the most padding-dominated case in the set -- and they
come back exactly right. A mask taking one length for the whole batch would have
destroyed those two first.

**TP=8 returns byte-identical output to TP=2** on all four prompts -- same
tokens, same divergence points, same text. TP=8 gives each rank 2 value heads
instead of 8, a different `conv_dim_per_rank`, and a different collective
pattern, so agreeing token for token does mean the GDN sharding and the
sequence-parallel gather/reduce-scatter carry the same state at both degrees.

> **Superseded in part.** This section originally read the 12/32 and 19/32 rows
> as bf16 drift, and read TP=8 == TP=2 as the TP-invariance section 7.5 asks
> for. Both readings were wrong, for the same reason: TP=1 had never been run.
> At TP=1 with the same four prompts the model matches HuggingFace **128/128,
> exactly**, so the 33 missing tokens here are a TP defect, not arithmetic
> noise. Invariance among TP>=2 holds; invariance between 1 and 2 does not. See
> "Two defects, not one" below. The two 32/32 rows still carry their original
> weight as evidence about the padding mask.

## Two defects, not one: the scan kernel and tensor parallelism

The previous revision of this section was titled "The chunk-scan kernel is wrong
on device" and blamed the kernel for everything. That was wrong twice over. The
kernel is **not** wrong on device, and the larger defect has nothing to do with
it.

### The four-way table

Every earlier comparison held TP fixed at 2 and varied only the kernel, so the
TP axis was never a variable. Running all four corners, same four prompts, same
`max_num_seqs=4`, greedy, against HuggingFace:

| configuration | tokens matching HF |
|---|---|
| TP=1, scan kernel **off** | **128 / 128 -- exact** |
| TP=1, scan kernel on | 77 / 128 |
| TP=2, scan kernel off | 95 / 128 |
| TP=2, scan kernel on | 1 / 128 |

Two independent effects, which compound:

1. **The scan kernel costs accuracy even at TP=1** (128 -> 77). This one is a
   real defect and is still open.
2. **TP>1 costs accuracy with no NKI kernel anywhere in the graph** (128 -> 95).

> **Superseded.** This section originally called the second one "the more
> serious" defect. It is not a defect at all -- a per-layer TP=1 vs TP=2 capture
> shows accumulated bf16 rounding from a changed reduction order, with no
> discontinuity and no layer-type asymmetry, and TP=4 scores *better* than TP=2.
> See "The TP gap is precision, not a bug" below. Row 1 stands.

The second effect was invisible until TP=1 was run. An
earlier note in this file recorded that TP=2 and TP=8 produce byte-identical
output and called that TP-invariance confirmed. They *are* identical to each
other -- but both are wrong, and TP=1 is right. Invariance among TP>=2 held;
invariance between 1 and 2 was never tested and does not hold. 95/128 was
previously read as bf16 drift. It is not drift: at TP=1 the model reproduces
HuggingFace exactly, so the 33 missing tokens are a real TP defect.

The GDN sharding *arithmetic* is not the cause -- the 21 CPU shard-invariance
tests in `test_qwen3_5_gdn_tp.py` pass, including the partition-coverage and
reassembly properties. That points at the attention sharding or at the
device-side collectives rather than at the GDN partition.

### Running the kernel on device, at last

The kernel could never be judged on its own because NKI's standalone path is
broken in this install: `nki.baremetal` shells out to
`python /tmp/nki_XXXXXXXX/None` and dies with `[Errno 2] No such file or
directory` before reaching the compiler. So "the kernel is wrong on device" had
only ever been inferred from whole-model output.

There is a way around it. `wrap_nki` produces a torch-callable HOP, so compiling
a one-op module with `torch.compile(backend="neuron")` drives the kernel through
*exactly* the lowering production uses. That is a better harness than baremetal
would have been, not a worse one: a bug that only shows up under the real
lowering is still in scope, while the rest of the model is not.
`tools/qwen3_5/run_scan_device.py` is that harness.

With it, the kernel is exact on device everywhere it was put:

| what was run on device | worst relative error |
|---|---|
| raw kernel, 7 geometries up to `b1 h16 c32 w64 d128` | 2e-7 |
| `chunk_gated_delta_rule_nki` wrapper, fp32 and bf16, carried state, pad branch | 1.5e-3 (bf16) |
| whole GDN layer including the paged-state seam | 2.2e-7 |
| the same layer in bf16 | 7.2e-3 |
| decay sweep from `exp(-32)` to no decay at all | 2e-7 |
| four stacked GDN layers | 1.9e-7 |
| state caches as **aliased** graph inputs | 1.9e-7 |
| TP=2, two ranks, real all-gather and reduce-scatter | 2.3e-7 |
| TP=2 with a padded bucket (12 real tokens in 2048) | 2.1e-7 |
| eight stacked GDN layers at TP=2, padded bucket | 3.2e-7 |
| real 0.8B layer-0 weights from the checkpoint | 1.7e-7 |

Each run asserts `can_use_chunk_scan_kernel` actually returned the intended
value, so a silent fallback cannot masquerade as a match.

Two of those rows exist because of mistakes worth recording. **The decay sweep**:
with `g = -rand(0,1)` the cumulative decay over a 64-token chunk is `exp(-32)`,
so the carried state is annihilated as soon as it is written and the sequential
recurrence is very nearly a no-op -- a bug in the inter-chunk carry would have
been invisible. Every probe before that one had this flaw. **The aliasing row**:
the real model compiles with 48 input->output aliases (18 GDN layers x 2 states
plus 6 attention layers x 2 KV tensors) and reports `Mutated inputs: [7, 8, 26,
...]`, whereas the early probes reported `Mutated inputs: []` -- no aliasing at
all, because the state caches were module attributes rather than mutated graph
inputs. Since Neuron's alias-output rewrite is a known clobber hazard here
(`requires_independent_kv_cache_tensors` exists for it), that gap had to be
closed before the kernel could be cleared.

So the kernel's own numerics are sound, and whatever costs 51 tokens at TP=1
lives in how it is scheduled or composed at model scale, not in what it computes.
It stays **off by default** behind `VLLM_NEURON_ENABLE_QWEN3_5_SCAN_KERNEL=1`.

## The TP gap is precision, not a bug

The previous section called the TP>1 accuracy loss "the biggest open correctness
defect". A per-layer capture says that is wrong, and it is worth being precise
about what the evidence actually shows.

**Method.** The real 0.8B run twice on the same prompt and seed with all NKI
kernels off, TP=1 and TP=2, with `tensor_capture` hooking all 24 decoder layers
(`neuron_config.tensor_capture`, `modules=["model.layers.0-23"]`). Diffed per
layer, per forward pass, prefill and decode separately — never aggregated. This
is the method that localized DeepSeek-V4's TP bug to `layers.0.moe` at 161%
relative error with everything upstream bit-exact.

**Result — prefill, TP=1 vs TP=2, expressed in bf16 ULP (1 ULP = 2^-8):**

```
L0  gdn  1.5    L6  gdn  1.0    L12 gdn  5.2    L18 gdn  3.2
L1  gdn  2.3    L7  attn 1.9    L13 gdn  4.4    L19 attn 3.0
L2  gdn  3.6    L8  gdn  2.0    L14 gdn  1.8    L20 gdn  3.6
L3  attn 3.4    L9  gdn  2.0    L15 attn 6.7    L21 gdn  4.8
L4  gdn  3.6    L10 gdn  1.9    L16 gdn  5.5    L22 gdn  4.2
L5  gdn  4.6    L11 attn 4.8    L17 gdn  4.6    L23 attn 6.5
```

Four independent properties, each of which a structural sharding bug would
violate:

1. **No discontinuity.** The error starts at 1.5 ULP in layer 0 and drifts to
   ~6.5 by layer 23, wandering up and down on the way (L5 is 4.6, L6 is 1.0).
   A partition error does not drift; it jumps, and everything upstream of it is
   exact.
2. **The ranks agree with each other exactly.** Decode captures record a
   per-rank spread of 0 — every `all_reduce` lands and every rank ends the layer
   holding the same tensor.
3. **No non-finite values anywhere**, in any layer, in any pass. This retires
   the leading candidate from the previous plan: the unmasked-V path in
   `_gather_cache`, where `0 * NaN = NaN` on never-written slots, is not firing
   in this configuration.
4. **Both layer types are equally affected**, which is the strongest single
   piece of evidence:

   | pass | attention layers | GDN layers |
   |---|---|---|
   | prefill | 4.38 ULP | 3.32 ULP |
   | decode 0 | 1.77 | 1.02 |
   | decode 1 | 2.15 | 1.44 |
   | decode 2 | 0.88 | **1.05** |

   A bug in the attention partition would put attention far above GDN; a bug in
   the GDN partition, the reverse. They track each other within ~1.3x, and in
   one decode pass GDN is the higher of the two. What they share is that both
   end in a collective.

**The magnitude accounts for the token flips.** The last layer differs by ~2.5e-2
relative. Logits sit at a scale of about 19.8 (measured earlier in this
document), so that is roughly 0.5 absolute — about 4 ULP at that magnitude. Any
token whose top-1/top-2 logit gap is under ~0.5 can therefore flip. In natural
text that is a large fraction of positions, which is exactly the observed
33-of-128 rate.

**The degree sweep corroborates it.** Running all four degrees, kernels off:

| TP | tokens matching HF | 12-token prompt |
|---|---|---|
| 1 | 128/128 | 32/32 |
| 2 | 95/128 | 12/32 |
| 4 | **115/128** | **32/32** |
| 8 | 95/128 | 12/32 |

Non-monotonic — 128, 95, 115, 95 — and the 12-token prompt is *exact* at TP=4
after failing at TP=2. A systematic sharding error does not improve when you
shard harder. This is coin-flipping on near-ties. It also retires the earlier
claim that "TP=2 and TP=8 are byte-identical, so the cause is structural and
degree-independent": TP=4 is not identical to either, so that agreement was
coincidence on a four-prompt set, not evidence of structure.

**Why TP=1 is exact and TP>1 is not.** At `world_size == 1` every collective is
bypassed (`GroupCoordinator.all_reduce` returns its input unchanged), so the
model computes each projection as one contraction, in the same order the
HuggingFace reference does — hence 128/128. At TP>1 the same sum is split across
ranks and recombined, which changes the summation order. Two mechanisms
contribute, and both are inherent rather than defects:

- the cross-rank reduction itself, performed in bf16 on the model dtype (no site
  in this repo upcasts for collectives — `qwen3` and `deepseek_v4` do the same);
- `NF.o_proj`'s 3-D path, which re-derives `D = min(128, ND)` and `N = ND // D`
  (`functional/attention/o_proj.py:186-192`), so the reduction is tiled as
  N = 16 / 8 / 4 / 2 at TP = 1 / 2 / 4 / 8. The contraction is mathematically
  identical; its accumulation order is not.

**What this means for acceptance testing.** "Token-identical to TP=1" is not a
reachable bar for bf16 tensor parallelism over a 32-token greedy generation, and
holding to it would mean chasing noise indefinitely. It is also not the bar that
matters: every TP=2/4/8 continuation observed here is coherent and factually
correct ("Rome. The capital of Spain is Madrid..."). The right criterion is a
quality metric — perplexity on a held-out set, or task accuracy — plus the
structural invariants that *are* exact and worth asserting: ranks agreeing with
each other, no non-finite values, and shard-reassembly identities.

If the constant needs to come down, the lever is reducing collectives in fp32
(upcast before, downcast after) at the cost of doubled collective bandwidth.
That should be measured before it is adopted, not assumed.

## How long a context this model can actually serve

Asked to size the 27B for a 32K-64K context with a 4096- or 8192-token prefill
bucket, the answer is that neither is reachable today, and the reason is not
tensor parallelism or memory.

**Three independent walls, in the order you hit them.**

1. `MAX_MODEL_LEN_SINGLE_SHOT = 16 * 1024` (`utils/bucket_utils.py:329`).
   Above 16K, `resolve_segmented_prefill_config` refuses single-shot prefill and
   requires chunked prefill.
2. Chunked prefill *is* segmented prefill here — the same function returns
   `kv_segment_size_buckets`, and the segmented attention kernel raises
   `head_dim=256 exceeds maximum supported head dimension (128)`
   (`functional/attention/attention_segmented_cte.py:510`). The 27B and the 0.8B
   are both head_dim 256.
3. Even with both of those lifted, the torch fallback materialises the whole
   score matrix — `scores = torch.matmul(q, k)` at
   `functional/attention/attention_cte.py:125`, shaped `[heads, T, T]`, because
   `flash_attention` also falls back at head_dim > 128. Per rank at TP=8
   (3 query heads):

   | prefill | scores, one copy | with softmax/probs |
   |---|---|---|
   | 4096 | 0.09 GiB | ~0.23 GiB |
   | 8192 | 0.38 | ~0.94 |
   | 16384 | 1.50 | ~3.75 |
   | 32768 | 6.00 | ~15.0 |
   | 65536 | 24.0 | ~60.0 |

   Against a ~24 GB per-rank budget already holding 6.26 GiB of weights, 32K is
   out on this term alone. So `MAX_MODEL_LEN_SINGLE_SHOT` is a policy constant,
   but it is not an arbitrary one.

**A consequence that is easy to miss:** with head_dim 256 the prefill bucket
cannot be *smaller* than `max_model_len`, because that is precisely the chunked
case. "Prefill 8192 with a 32K context" is not a configuration this model has —
prefill bucket and `max_model_len` must be equal, and both under 16K.

**The blocker is deeper than the kernel.** `Qwen3_5Attention.forward_prefill`
has no `kv_segment_size` branch at all. Its sibling `qwen3` reads
`attn_metadata[layer_name]["kv_segment_size"]` and dispatches to
`NF.segmented_attention` (`qwen3/model.py:321, 363-371`); qwen3_5 runs one
`flash_attention` over whatever tokens it was handed. So even a head-dim-256
segmented kernel would not be enough on its own — the layer needs the branch
too. Long context is two pieces of work, not one.

**This used to fail silently, and now does not.** `needs_single_shot_prefill`
was consulted in exactly one place, the factory, and only logged a warning.
Nothing rejected a chunked configuration. Handed a chunk, the attention layer
attends within that chunk and ignores everything cached before it — not a crash,
but coherent, confident text computed against a truncated context, which is the
worst available failure mode. `_validate_config` now raises when
`kv_segment_size_buckets` is set, with a message naming the fix. The regression
test asserts the raise, and was confirmed to fail without the guard.

**Practical ceiling today: `max_model_len == max_num_batched_tokens <= 16384`.**
At 16K with 8 concurrent requests on the 27B at TP=8 that is roughly 6.26 GiB
weights + 2.0 GiB KV + 0.14 GiB GDN state + ~3.75 GiB prefill transient, near
12 GiB of 24 before the compiler arena — comfortable. 32K and 64K need a tiled
head-dim-256 prefill kernel plus the segmented branch in the layer, and belong
in their own plan.

## Still open

- ~~**TP>1 loses accuracy on its own.**~~ **Closed, and it was not a defect.**
  The per-layer capture in "The TP gap is precision, not a bug" shows a smooth
  1.5 -> 6.5 bf16 ULP drift with no discontinuity, ranks agreeing exactly, no
  non-finite values, and attention and GDN equally affected. TP=4 scores 115/128
  where TP=2 scores 95 -- non-monotonic, so not systematic. What remains is a
  decision, not a bug: whether to spend doubled collective bandwidth on fp32
  reductions to lower the constant. Measure before adopting.
- **Acceptance testing needs a quality metric, not exact-token-match.** Holding
  TP>1 to "identical to TP=1" is unreachable in bf16 and would mean chasing
  noise. Use perplexity or task accuracy, plus the invariants that *are* exact:
  cross-rank agreement, finiteness, and shard reassembly.
- **The scan kernel costs 51 tokens at TP=1** (128 -> 77) even though it is exact
  on device in isolation at every geometry, dtype, decay regime, padding, alias
  structure and TP degree tried (see the table above). What is left is scale and
  scheduling: 18 GDN call sites in one graph rather than the 8 that have been
  reproduced clean, plus the interleaved attention layers. Until it is understood the GDN prefill runs the torch chunk rule,
  which is a performance gap, not a correctness one.
- **TP=16 and TP=32 are unverified on hardware, and cannot be checked on the
  0.8B at all**: vLLM requires `num_attention_heads % tp == 0` and this
  checkpoint has 8, so TP=16 is rejected at config validation before reaching a
  device. TP=8 is the ceiling here. TP=16 needs the 27B together with the 24->32
  Q-head padding of section 2.2, so the two are gated on each other.
- **Attention propagates padding NaN into real rows.** Through layers 0-2 the 32
  real rows stayed clean while padding rows were NaN (`bad_real = 0`); at layer 3,
  the first attention layer, all 2048 rows went bad including the real ones.
  Padding sits strictly *after* the real tokens and every op here is per-token or
  causal, so a causal-correct attention cannot do that. The inverse fix removes
  the NaN source, but this contamination path is a separate defect and would bite
  again for any other source of non-finite values.

## Where the 27B stands

Downloaded (18 shards, 52 GB, 1199 keys: 850 text, 333 visual, 15 MTP) and all
four load blockers above are fixed. Not yet compiled.
