# DeepSeek-V4 real-weight validation: CPU and device both exact

Status as of 2026-08-23, on `trn2.3xlarge`, branch `ds-v4-tiny-official-weights`.

This records a validation pass that drove the plugin and
`transformers.models.deepseek_v4` from the *same* real DeepSeek-V4-Flash
weights and compared them numerically. It found and fixed one model-correctness
bug and four device-only bugs, and it ends with one unresolved device
divergence, documented here with the hypotheses already eliminated.

Scope: a three-layer slice, 32 of 256 routed experts, single request, TP1. This
is not a claim about full DeepSeek-V4 at production shape.

## Where things stand

| Path | Result |
| --- | --- |
| CPU vs `transformers` reference, sliding layer | matches, 0 logits out of tolerance |
| CPU vs reference, sliding + CSA + HCA | matches, 0 logits out of tolerance |
| CPU vs reference, CSA alone | matches, 0 logits out of tolerance |
| Device gate, synthetic checkpoint | passes (3 NEFFs cold, 3 hits / 0 HLOs warm) |
| Device vs CPU, real weights | matches, 0 logits out of tolerance |

The tolerance is 0.125, the BF16 ULP at the observed logit scale (~23). The
plugin's `lm_head` emits BF16, so nothing below half that is achievable by any
implementation; a tighter bound measures the output dtype rather than the model.

## Part 1: the compressed-entry off-by-one

### Symptom

On the three-layer slice, prefill step-0 logits differed from the reference by
`max|diff| = 1.084`, with 68,303 logits past tolerance. All four generated
tokens still matched, so aggregate token agreement did not reveal it.

### What made it findable

Comparing *per layer and per token* rather than as one scalar. Layer outputs at
prefill, plugin against reference:

| Layer | ratio | per-token max abs diff (8-token prompt) |
| --- | --- | --- |
| 0 sliding | 0 | `0 0 0 0 0 0 0 0` |
| 1 CSA | 4 | `0 0 0 `**`0.2027`**` 0 0 0 `**`0.1186`** |
| 2 HCA | 128 | `0 0 0 0.2219 0.0108 0.0323 0.0132 0.1304` |

Divergence appeared only at tokens 3 and 7 -- `position % 4 == 3`, the
compression block boundaries -- and was exactly zero elsewhere. Layer 2
(ratio 128, which completes no window inside 8 tokens) only inherited layer 1's
error and spread it through attention.

That sparsity is the whole reason this survived: six of eight positions were
bit-exact, so any aggregate metric reads the result as floating-point noise.

### Root cause

`DeepseekV4Attention._compressed_history` counted visible compressed entries as
`position // ratio`. But a token completes a window when
`(position + 1) % ratio == 0` -- the write side's own rule, in
`compressed_entry_slot_mapping`, ported from vLLM's GPU backend -- and the
compressor writes that entry *before* attention reads the history. The entry was
already sitting in the cache and the mask hid it from the very query that
produced it.

The reference gates visibility the same way the write side does:

```python
causal_threshold = (position_ids + 1) // self.compress_rate  # DeepseekV4CSACompressor
```

Two halves of one convention, drifting apart in two different files.

### Fix

The rule now lives beside its write-side counterpart as
`attention.visible_compressed_entries`, so the two are read together.

| Measurement | before | after |
| --- | --- | --- |
| layer 1 (CSA/4) | 0.202666 | **0.000000** |
| layer 2 (HCA/128) | 0.221915 | 0.000101 |
| step-0 logit max abs diff | 1.084224 | 0.077663 |
| logits past tolerance | 68,303 | **0** |

A slice whose only layer is CSA went from every token wrong (`max|diff|` 12-28)
to matching -- with no sliding layer, the compressed entries carry all of the
information, so the same off-by-one is no longer diluted.

`test_compressed_entry_is_visible_to_the_query_that_completes_it` pins the
read/write agreement directly; it fails against the old behaviour.

## Part 2: four device-only bugs

None of these are reachable from the CPU oracle. Each was found by running the
real-weight slice on Trn2 and fixing whatever stopped it.

1. **`hash_experts` bound check was untraceable.** It branched on tensor
   *values* (`if input_ids.min() < 0 or ...`). Dynamo raises "Data-dependent
   branching" on that rather than graph-breaking, so tracing failed outright.
   Now eager-only, matching the `sinkhorn_positive` guard already beside it.

2. **Identity K/V init used `expand` on the device.** Neuron rejects a
   non-contiguous `copy_` source *and* cannot run the `.contiguous()` that would
   fix it. Built on CPU and copied across.

3. **RoPE buffer reinit passed the device to `ROPE_INIT_FUNCTIONS`.** Those do
   `torch.arange(...).to(device=device, dtype=torch.float)` -- a device move and
   a dtype cast in one step, which Neuron rejects. Only the class's own
   `compute_default_rope_parameters` is device-safe, so a config selecting yarn
   (every real DeepSeek-V4 config) failed where a default-RoPE one did not.

4. **Weight loading copied BF16 into FP32 across devices.** CPU upcasts
   silently; Neuron rejects a copy that changes device and dtype together.
   `_copy_into` casts on the host first, which is numerically what CPU was
   already doing implicitly.

### Why the device gate never caught the first one

Before the config normalization added in `c4aa67d`, vLLM's `get_config` skipped
`DeepseekV4Config.__post_init__`, so `mlp_layer_types` stayed absent and the
plugin silently fell back to routed-MoE. The device gate had therefore **never
exercised the hash router at all**, which is what real DeepSeek-V4 layers use.
Normalizing the config made the synthetic checkpoint faithful, and the
untraceable guard surfaced immediately.

This is worth generalizing: a synthetic checkpoint validates only the code paths
its config actually selects, and a config that quietly defaults is a config that
quietly narrows the gate.

## Part 3: the device divergence, and its cause

### What was observed

With the four fixes above the real-weight slice ran on device cleanly -- 3
NEFFs, no PJRT faults -- and produced **different tokens** from CPU:
`[85, 92132, 109502, 98751]` against CPU's `[85, 85276, 50955, 125488]`.
Step-0 logits differed by `max|diff| = 20.69`, with 96% past tolerance, though
mean and std matched: a structurally-similar *wrong* computation, not garbage.

Per-module capture put the first divergence at `model.layers.0.attn_hc`, the
mHC gate, with `model.embed_tokens` bit-exact. Prompt token 0 was exact at every
module; tokens 1-7 were wrong from `attn_hc` onward.

### Root cause: `Tensor.split` with a list of sizes

**`Tensor.split(sizes, dim)` returns the wrong data on Neuron whenever `sizes`
is a *list* and `dim` is not 0.** Not one chunk -- every chunk, silently.

`hyperconnection_reference` did:

```python
pre_w, post_w, comb_w = F.linear(flat, fn.float()).split([hc, hc, hc * hc], dim=-1)
```

Compiling that module alone made the contradiction visible: the *un-split*
`F.linear` result was correct to `2.4e-4`, while `post_w` -- which is literally
`projected[:, 4:8]` of that same correct tensor, in the same graph -- was off by
**717**. The dumped FX graph is correct (`split` -> `getitem`), so the defect
lies below FX, in the torch-xla / neuronx-cc lowering.

Measured behaviour, from `check_mhc_device.py --check-split`:

| Form | Result |
| --- | --- |
| `split([4,4,16], dim=-1)` | **all chunks wrong** |
| `split([8,8,8], dim=-1)` (uniform, but a list) | **all chunks wrong** |
| `split([4,20], dim=1)` | **both chunks wrong** |
| `split(8, dim=-1)` (int size) | correct |
| `chunk(3, dim=-1)` | correct |
| `split([2,6], dim=0)` | correct |

So it is the *list-of-sizes* form on a non-zero dim. This is a general defect,
not a DeepSeek-V4 one -- it is catalogued with the stack's other lowering
pitfalls in
[`neuron-lowering-pitfalls.md`](neuron-lowering-pitfalls.md). An int size,
`chunk`, and dim 0 are all fine -- which is why nothing else in the plugin was affected:
`mhc.py` held the only such call inside a compiled graph. (`eagle3_model.py`
and `topk.py` use `torch.split`, but with int sizes and/or dim 0, and outside
the graph.)

### Fix

Slice explicitly instead. Both decompositions in `hyperconnection_reference`
were converted, including the `base` one that splits on dim 0 and measures
correct -- two adjacent decompositions in the same function should not be able
to drift apart again.

| Measurement | before | after |
| --- | --- | --- |
| mHC gate `post`, device vs CPU | 0.167178 | **2.2e-08** |
| mHC gate `collapsed` | -- | **0.000000** |
| full model, step-0 logits device vs CPU | 20.69 | **0.0625** |
| logits past 0.125 | 124,450 of 129,280 | **0** |
| device tokens | `[85, 92132, 109502, 98751]` | `[85, 85276, 50955, 125488]` |

The residual 0.0625 is half a BF16 ULP at logit scale 23 -- the floor.

Per-layer and per-token at prefill, device against CPU, after the fix (the
aggregate-only view is what hid the Part 1 bug, so this is checked at every
position, not just in total):

| Module | max abs diff | worst position |
| --- | --- | --- |
| `embed_tokens` | 0.000000 | -- |
| `layers.{0,1,2}.attn_hc` | 0.000000 | -- |
| `layers.{0,1,2}.ffn_hc` | <= 0.000034 | -- |
| `layers.0.attention` | 0.000016 | -- |
| `layers.2.attention` | 0.001805 | token 4 |
| `lm_head` | 0.062500 | -- |

Worst across every module and every position: **0.0625**, which is the BF16
output floor. Before the fix the same table read `attn_hc` 0.167,
`layers.0.attention` 7.10, `lm_head` 20.69.

The synthetic gate is unaffected: 3 NEFFs cold, 3 hits / 0 submitted HLOs warm,
no faults, and `t28t26t48t27` both cold and warm, identical to before the fix.

### How it was found, and what that cost

The productive move was abandoning full-model compiles (7-13 min each) for a
**standalone module compile**, following
`examples/vllm_neuron/basics/helloworld.py`. `torch.compile(module,
backend=get_compile_backend_name())` on the gate alone compiles in about **1.5
seconds**, and because the harness owns the module it can simply *return* the
intermediates -- no `tensor_capture`, which keeps only element `[0]` of a tuple
return. That harness is committed as `tools/deepseek_v4/check_mhc_device.py`.

Hypotheses eliminated along the way, all by experiment. Recorded so they are not
retried:

- *BF16 precision.* BF16 emulation of the gate on CPU reproduced FP32 to three
  digits; the device was off by 0.167 from **both**.
- *`expand` lowering.* `expand` -> `repeat` gave byte-identical device output.
- *A uniformly wrong RMS denominator.* `post_w = W @ flat` is linear in `flat`,
  so a scalar error scales all four mHC components equally; inverting the
  sigmoid gave per-component ratios disagreeing from -2.4 to 1.6.
- *Wrong input selection.* Brute-forcing all 8^4 assignments of prompt
  embeddings to the four stream slots explained rows 0 and 6 but no others.
- *FX pass interference.* `device_rewriter` only rewrites `device` kwargs;
  `nki_kernel_backend_config_pass` only touches NKI nodes; `inplace_to_outofplace`
  finds no in-place op here.
- *RMS-norm kernel substitution.* `ir_op_priority` is call-site dispatch
  reachable only through the `RMSNorm` module and resolves to `native`; the
  Neuron path never reaches Inductor, and there are zero pattern matchers on it.
- *The reduction at the 16384 free-dimension limit.* Plausible on shape grounds
  -- 16384 is exactly `2**14` -- but the standalone harness showed `variance`
  accurate to `9.3e-10`. The reduction was never the problem.

The last two entries are the point: the suspicion carried into this
investigation (a per-token RMS reduction) was **wrong**, and one 1.5-second
compile that returned the intermediates settled it. Localize by measuring
intermediates, not by reasoning about which op looks riskiest.

## Reproducing

Build the slice from official shards, then compare CPU against the reference:

```bash
.venv-neuron/bin/python tools/deepseek_v4/fetch_official_shards.py <shard-dir>
.venv-neuron/bin/python tools/deepseek_v4/build_tiny_from_official.py \
  <shard-dir> <slice-dir> --layers 0,2,3

VLLM_NEURON_ENABLE_DEEPSEEK_V4=1 VLLM_NEURON_CPU_MODE=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_NEURON_TINY_VALIDATION_DIR=<capture> \
.venv-neuron/bin/python tools/deepseek_v4/generate_tiny_tp1.py <slice-dir> \
  --output <json> --enforce-eager --load-format auto \
  --prompt 671,6102,294,8760,344,1024,2048,4096 --max-model-len 16

cd tools/deepseek_v4 && PYTHONPATH=$PWD ../../.venv-neuron/bin/python \
  compare_against_reference.py <slice-dir> <capture> \
  --prompt 671,6102,294,8760,344,1024,2048,4096
```

Use a prompt that exactly fills a bucket (8 tokens here). A shorter prompt is
padded, and the padded positions are compared against real reference positions.

For the device run, drop `VLLM_NEURON_CPU_MODE` and `--enforce-eager`, and set
`NEURON_VISIBLE_DEVICES`, `NEURON_SKIP_EFA_AFFINITY=1`, a private
`VLLM_CACHE_ROOT`, and `PATH` including `.venv-neuron/bin` so Lite finds its
matching `neuronx-cc`.

### The fast loop

Before reaching for a full-model compile, use the standalone harness -- about
1.5 seconds per iteration instead of 7-13 minutes:

```bash
PATH="$PWD/.venv-neuron/bin:$PATH" \
NEURON_VISIBLE_DEVICES=0 NEURON_RT_VISIBLE_CORES=0 \
NEURON_SKIP_EFA_AFFINITY=1 VLLM_CACHE_ROOT=/tmp/mhc-cache \
.venv-neuron/bin/python tools/deepseek_v4/check_mhc_device.py

# and to re-characterise the split lowering defect itself
... tools/deepseek_v4/check_mhc_device.py --check-split
```

`NEURON_RT_VISIBLE_CORES` must be set alongside `NEURON_VISIBLE_DEVICES` --
the runtime rejects one without the other, and unlike the vLLM worker this
harness has nothing to set it for you. `--check-split` exits non-zero while the
compiler defect is present; it will start passing if AWS fixes the lowering, at
which point the slicing workaround could be revisited (but need not be).

### Slice shapes that will not load

vLLM asserts `max(sm_page_sizes) <= max(all_page_sizes)` when grouping KV
caches. Any slice carrying a sliding layer but no ratio-4 layer -- `[0,3]`,
`[3]` -- fails at engine start. Slice surgery is therefore a poor isolation tool
for the HCA layer; per-layer capture on a valid slice is the working approach.

## Still open

- The full 256-expert set, for final correctness sign-off. The 32-expert subset
  is an iteration-speed compromise; running the complete MoE also removes router
  remapping entirely (no `gate.weight` slice, no `tid2eid` remap), eliminating a
  whole class of extraction bug.
- Layers beyond 0/2/3, which a three-layer slice cannot bound for
  depth-dependent error accumulation.
- An FP8/MXFP4-preserving variant, to exercise the quantized load path. MXFP4
  needs Trn3.
- `Tensor.split` with a list of sizes is still wrong on this Neuron stack for
  any non-zero dim. Worth reporting to AWS; `check_mhc_device.py --check-split`
  is a self-contained reproducer. Until it is fixed, avoid that form anywhere
  inside a compiled graph.
- The plugin never honours the configured model dtype: parameters are created
  FP32 by `torch.randn` and `copy_` upcasts the BF16 checkpoint, so `--dtype` is
  inert for this model and weights occupy twice their intended footprint. Device
  bug 4 above is a symptom of this; the underlying gap is unfixed.
