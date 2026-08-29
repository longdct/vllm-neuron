
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
Prefill *and* decode are finite on CPU. The model, the weight loading and the
tied-embedding mapping are therefore all correct.

## Two device-only NaNs

On device the same checkpoint emits token 0 forever. Logprobs show why -- it is
NaN, and argmax over an all-NaN vector ties to index 0:

```
TOKENS [0, 0, 0]
STEP0 0:nan 3:nan 4:nan 1:nan 2:nan
```

Bisection, one device run per row, all on the real checkpoint:

| variable | verdict |
|---|---|
| model + weights (CPU fp32 and bf16) | correct -- matches HF |
| `head_dim=256` | exonerated -- a fixture at 256 generates normally |
| `max_model_len` 2048 | exonerated -- fixture generates normally at 2048 |
| NKI conv + scan kernels | exonerated -- NaN persists with both disabled |
| TP degree (8 vs 2) | exonerated -- NaN at both |
| TP=1 | **cannot compile**: SBUF exhaustion. This model has a minimum |
| | viable TP degree, so TP=1 is not available as a debugging fallback. |

Skipping the attention sublayer entirely (`torch.zeros_like`, not multiplying
its output by zero -- `NaN * 0` is still NaN) splits the failure in two:

```
TOKENS [106384, 0, 0]
STEP0 106384:-4.6152 27614:-5.2402 228671:-5.3027   <- finite, real distribution
STEP1 0:nan ...                                      <- still NaN
```

**NaN 1 -- prefill attention.** Step 0 is NaN with attention on and finite with
it skipped. `forward_prefill` calls `NF.flash_attention` with no explicit mask;
at `head_dim=256` that falls back to torch inside `NF`. A fully-masked softmax
row is the classic way to produce NaN here, and a 32-token prompt in a 2048
bucket leaves ~2000 padding rows. Not yet confirmed -- the mask behaviour
inside the fallback has not been read.

**NaN 2 -- GDN decode.** Steps 1+ stay NaN with attention skipped *and* both
NKI kernels disabled, so pure-torch GDN decode goes NaN on device while the
identical code is finite on CPU. Untouched.

These are independent: fixing either alone leaves the other.

### Reproducer

A 4-layer prefix of the real checkpoint reproduces both, so iteration is
minutes rather than tens of minutes. Symlink the weight files into a new
directory and truncate `num_hidden_layers` and `layer_types` to 4 in
`config.json` -- the loader only requests the layers the config declares. The
prefix must keep the 3:1 schedule's own indices, because layer index decides
which weights exist; 4 layers is the shortest prefix containing an attention
layer.

A GDN-only prefix (3 layers) is **not** a usable control. The config guard
rejects it, and with the guard relaxed the runner fails anyway at
`model.py:483`, `next(k for k in attn_metadata if k.endswith("self_attn"))`,
with `StopIteration`. A purely linear stack is unsupported end to end.

## Where the 27B stands

Downloaded (18 shards, 52 GB, 1199 keys: 850 text, 333 visual, 15 MTP) and all
four load blockers above are fixed. It was deliberately **not** compiled: a
64-layer cold compile that reproduces these NaNs would cost hours and teach
nothing the 0.8B has not already shown more cheaply.
