
# Real Qwen3.5 checkpoints: what it took to load, and two device NaNs

Date: 2026-08-29. Checkpoints: `Qwen/Qwen3.5-0.8B` (24 layers, 18 GDN + 6 full,
hidden 1024, head_dim 256, 16 K / 16 V heads, vocab 248320) and
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

## Still open

- **The NKI chunk-scan kernel is wrong.** With kernels enabled the device emits
  token 61 thirty-two times; with `VLLM_NEURON_DISABLE_NKI_KERNELS=1` it tracks
  the CPU path. It is the one component CPU never exercises, which is why it
  survived the oracle and simulator suites -- those check the kernel against the
  torch form on shapes the serving path does not produce.
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
