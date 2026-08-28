# The DeepSeek-V4 lightning indexer

What the indexer is, why it needs a second compressor, and where the two places
it can silently go wrong are. Companion to
[deepseek-v4-carry-cache-design.md](deepseek-v4-carry-cache-design.md), which
covers the compressor addressing this builds on.

## What it does

Compressed Sparse Attention compresses every 4 source tokens into one entry.
Without an indexer, a query attends to *all* of them — which is what this plugin
did until now. The lightning indexer scores each compressed entry against the
current query and keeps only the best `index_topk` (512 in V4-Flash).

The property that makes the staged bring-up possible is that **it only selects;
it never weights**. A selected entry enters attention with exactly the weight it
would have had under dense attention. So wherever the eligible set is no larger
than `index_topk`, selecting the top-k *is* selecting everything, and omitting
the indexer is not an approximation — it is the same computation. That is what
`dense_csa.py` turns into an admission bound, and it is why the indexer could be
deferred this long.

It is also why the indexer is unusually easy to fake. Below the bound, an
implementation that selects everything, and one that is correct, produce
identical logits. Every test here therefore runs *past* the bound and asserts
that entries were actually discarded.

## The math

Per paper §2.3.1 eqs. 14–17, and bit-exact against Transformers 5.15's
`DeepseekV4IndexerScorer`:

```
scores[t,s] = ∑_h w[t,h] · ReLU(q[t,h] · k[s])
```

with `q = q_b_proj(q_residual)` reshaped to `index_n_heads × index_head_dim` and
RoPE'd at the token's position, `k` the indexer's own compressed entries, and
`w = weights_proj(hidden)`. Two scale factors, both derived from the tensors
rather than configured: `index_head_dim**-0.5` on the scores and
`index_n_heads**-0.5` on the gate.

`q_residual` is the post-`q_a_norm` residual the attention layer already
computes for its own query projection. It is reused, not recomputed.

Everything runs in fp32. The scores feed a top-k, so ties and near-ties decide
*which* entries attention sees; bf16 collapses neighbouring scores into exact
ties and makes selection depend on the tie-break rather than on the model.

## The second compressor

The part that surprises people reading the reference: **the indexer runs a
complete compressor of its own**, at `index_head_dim` (128) instead of the
model's `head_dim` (512), over the same windows, with the same two-series
Ca/Cb overlap layout and the same `compress_rope_theta`.

It is *not* a projection of the outer compressor's output. It has its own
`wkv`/`wgate`/`ape`/`norm` weights in the checkpoint and its own cache state;
the reference keeps the two side by side under the keys `"compressor"` and
`"indexer"` (`DeepseekV4CSACache`). Sharing the theta is what keeps query and key
rotations consistent — with different thetas, `q · k` would carry a residual
position-dependent skew.

This plugin models it the same way: `DeepseekV4Indexer` owns a
`DeepseekV4Compressor`, the same class the attention layer uses, constructed at
the indexer's width. Hence **two additional cache groups per c4 layer**:

| group | kind | dtype | head_size |
|---|---|---|---|
| `…self_attn.indexer` | MLA | bf16 | `index_head_dim` |
| `…self_attn.indexer.compressor.state_cache` | COMPRESSOR_STATE | fp32 | `4 × index_head_dim` |

Only `compress_ratio == 4` layers have one. c128/HCA layers attend to their whole
compressed history, and the checkpoint ships indexer tensors for the c4 layers
alone.

## Where it plugs in

`DeepseekV4Attention._forward_one_token` runs one token at a time in prefill and
decode alike, so there is exactly one query per call. The indexer returns a
boolean mask over compressed entries, ANDed into `compressed_valid` before the
existing concatenation into `key_valid`:

```python
compressed_history, compressed_valid = self._compressed_history(...)
if self.indexer is not None:
    compressed_valid = compressed_valid & self.indexer(...)
```

The reference spells the same thing as an additive `-inf`/`0` `block_bias`
scattered into a buffer one column wider than the entry axis, the extra column
absorbing `-1` sentinels. With one query token the two are the same statement,
and a mask is what `mla_attention_reference` already takes — so no tensor shape
changes. This is correctness only; the gather that turns selection into a
throughput win is separate, and still to come.

## Two ways this goes silently wrong

**The causal threshold.** Selection and the dense read must agree on which
entries exist. They disagree only at `position % ratio == ratio - 1`, which is
1 position in 4 for c4 layers and reads as floating-point noise in any aggregate
metric. That exact off-by-one already shipped here once
(`11ef0a9`). The rule lives in one place,
`attention.visible_compressed_entries`, and both readers call it —
`read_compressed_history` for the dense path, and the indexer for its own
parallel cache. It is passed *into* the selection functions rather than
recomputed inside them.

**Masking before the top-k.** Entries past the threshold are zeroed by the
reader, and a zeroed key scores exactly `relu(q·0) == 0`. A real entry whose
gate weight is negative scores *below* zero. An unmasked top-k therefore spends
its budget on padding and discards real content — while looking entirely
healthy. The `-inf` fill is load-bearing, not defensive, and
`test_indexer.py::test_negative_gate_produces_scores_below_a_zeroed_row` pins the
ordering.

## Weight names

The real checkpoint nests the indexer as a **sibling of the compressor**, which
is vLLM's layout, not Transformers' `self_attn.compressor.indexer.*`:

```
layers.N.attn.indexer.wq_b.weight            -> …attention.indexer.q_b_proj.weight
layers.N.attn.indexer.weights_proj.weight    -> …attention.indexer.weights_proj.weight
layers.N.attn.indexer.compressor.wkv.weight  -> …indexer.compressor.fused_wkv_wgate (shard 0)
layers.N.attn.indexer.compressor.wgate.weight-> …indexer.compressor.fused_wkv_wgate (shard 1)
layers.N.attn.indexer.compressor.ape         -> …indexer.compressor.ape
layers.N.attn.indexer.compressor.norm.weight -> …indexer.compressor.norm_weight
```

`tools/deepseek_v4/hf_reference.py` inverts the same contract for the reference
side. Note the collision risk: the indexer's compressor and the layer's own are
the same class with identically-named parameters, distinguished only by the
`indexer.` segment, and `resolve_stacked_shard` matches on a bare substring.

## Testing

| where | what |
|---|---|
| `test/unit/model/deepseek_v4/test_indexer.py` | Weight-independent properties: never selects the future, never exceeds the budget, never keeps padding, degenerates to dense at budget ≥ entries. Plus a `torch.export` check for dynamic-shape ops. |
| `test/vllm_neuron/test_deepseek_v4_component_oracles.py` | Bit-exact against `DeepseekV4IndexerScorer` and against the selection extracted from `DeepseekV4Indexer.forward`. Runs 40 tokens against a budget of 3 and asserts pruning happened. Covers tied scores. |
| `test/vllm_neuron/test_deepseek_v4_model_assembly.py` | End to end: a small budget changes the output, selection is chunk-invariant across prefill/decode splits, and a budget ≥ the entry count reproduces dense attention exactly. |

The last two cross-check each other. Budgets of 10 and 4096 agreeing bit-exactly
is what proves the pruning test's two models share weights, so the difference it
measures can only come from selection.

### On device

`tools/deepseek_v4/check_indexer_device.py` compiles scoring, selection and the
mask alone (~1 min, against ~200 s for the whole model) and diffs each against
CPU. It runs three regimes and **the first is the least informative**: a row
with real values to rank, a row that is entirely `-inf`, and a row holding one
real pick beside a sentinel. Only the last two expose the uint32 top-k sentinel
bug; a probe that tests only the first reports MATCH while the model faults on
device.

`--stack` picks the backend — `xla` for vllm-neuron's torch-xla 2.11, `neuronx`
for the from-source torch-neuronx 2.12 in `~/.venv-torch-neuronx-dev`, `auto`
(the default) for whichever is importable. The two lower differently and
disagree here, so any claim about "Neuron gets this right" has to name one:

| stack | raw `topk` index dtype | shipped indexer | pre-fix indexer |
|---|---|---|---|
| torch 2.11 / torch-xla 2.11 | uint32 | MATCH | **DIVERGENCE** — `-1` arrives as 4294967295 |
| torch-neuronx 2.12.3 / torch 2.12.1 | int64 | MATCH | MATCH |

Both verified at the tiny shapes and at the real Flash config (`entries=1024,
topk=512, heads=64, head_dim=128`), all three regimes. The `.long()` cast in
`select_compressed_entries` therefore stays.

### Does a newer torch change any of this? Not on the stack vLLM runs on.

Worth stating carefully, because "torch 2.12 fixes it" and "torch 2.12 does not"
are both reported, and both are right — of different stacks. Full procedure and
the reconciliation in [neuron-lowering-stacks.md](neuron-lowering-stacks.md):

* **The release line** — `torch 2.12.1`, `torch-xla 2.12.0`,
  `libtorch-neuronx-lite 2.12.0.1.0.1284`. This plugin runs on it unchanged
  (vLLM 0.24's extensions are `abi3`, so they survive the torch bump; only
  `torchaudio` has no 2.12 build, and transformers guards that import). It fixes
  **neither** defect: `Tensor.split` still mis-lowers and top-k still returns
  uint32.
* **The from-source `torch-neuronx` 2.12.3** (`~/.venv-torch-neuronx-dev`) —
  gets both right, because Dynamo decomposes `split` into `aten.slice` and
  top-k returns signed indices there. It **cannot run this plugin**:
  `envs.get_compile_backend_name`
  returns `neuron_libtorch` or lite's native backend and reserves the name
  `"neuron"` for torch_neuronx as a separate install, and 49 files under
  `vllm_neuron/` import lite APIs. Driving it is a port, not an install.

Measured end to end on the real 3-layer slice (8-token prompt,
`--max-model-len 16`), release line vs release line:

| comparison | worst `max|diff|` | argmax |
|---|---|---|
| **device 2.11 vs device 2.12** | **0** — bit-identical | identical |
| device vs CPU, on 2.11 | 0.25 | identical |
| device vs CPU, on 2.12 | 0.25 | identical |
| CPU 2.11 vs CPU 2.12 | 0.125 — one bf16 ULP at these magnitudes | identical |

The first row is the one that matters: **the torch version changes nothing on
device, bit for bit.** The device-vs-CPU gap is the same size on both stacks, so
it is pre-existing and numerical, not a lowering fault — and generated tokens
are `85, 85276, 50955, 125488` in all four runs.

To rebuild the release-line 2.12 environment, clone the working venv rather than
upgrading it in place — nothing here is worth losing a 12 GB install to:

```bash
cp -a ~/.venv-vllm-neuron ~/.venv-vllm-neuron-212
~/.venv-vllm-neuron-212/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 torchvision
~/.venv-vllm-neuron-212/bin/python -m pip install --no-deps \
  --extra-index-url https://pip.repos.neuron.amazonaws.com \
  torch-xla==2.12.0 libtorch-neuronx-lite==2.12.0.1.0.1284+f49d8626
~/.venv-vllm-neuron-212/bin/python -m pip uninstall -y torchaudio
```

The tiny model's own device gate is the usual CPU-vs-Neuron logit comparison,
but note the default 8-token prompt with 4 outputs never leaves the
dense-equivalent regime at `index_topk=2` — at most one entry is ever dropped,
on the final token. Pass a longer prompt to make the gate mean something:

```
--prompt "$(seq -s, 1 40)" --max-model-len 64 --num-gpu-blocks-override 256
```

At 40 prompt tokens the c4 layer reaches 11 compressed entries against a budget
of 2: 33 of 44 tokens prune, 153 entry-selections are dropped across the
sequence, and 9 are dropped on the last token alone.

Measured on trn2, that run: no out-of-bounds, 4/4 captures, worst per-logit
`max|diff| = 0.0098` against CPU (tolerance 0.025), argmax identical at every
step, same generated tokens `[10, 2, 43, 2]`. The default 8-token prompt at
`max-model-len 16` also passes (`max|diff| = 0.0078`) but, as above, barely
exercises selection.

### Against the reference, on real weights

The 3-layer slice from official shards (`--layers 0,2,3`; its layer 1 is the c4
CSA layer) carries all six indexer tensors, and the loader now consumes them
rather than skipping them. Compared with
`tools/deepseek_v4/compare_against_reference.py`, 8-token prompt,
`--max-model-len 16`:

| `index_topk` | regime | generated tokens | vs reference |
|---|---|---|---|
| 512 (stock) | dense-equivalent — at most 4 entries exist | `85, 85276, 50955, 125488` | argmax matches at every step |
| 1 | selection actually prunes | `85, 63624, 10964, 28953` | argmax matches at every step |

Both rows matter, and the second is the one that proves something. At the stock
budget the plugin reproduces the pre-indexer output **bit for bit** — verified
against the fork point, which printed identical `max|diff|` figures — because
with ≤4 entries and a budget of 512, selecting the top-k is selecting
everything. Dropping the budget to 1 changes three of the four generated
tokens, and the reference changes to *the same three tokens*.

The residual per-logit `max|diff|` of 0.14–0.21 (7–61 logits past the 0.125
tolerance, out of a 129k vocab) is **pre-existing and unrelated to the
indexer**: the fork point produces the identical numbers on the identical slice
and prompt. Do not read it as an indexer regression; the comparison script
still reports DIVERGENCE on it.

## Tensor parallelism

Replicated. Upstream head-shards `q_b_proj` and `weights_proj` colwise and
all-reduces the scorer output so every rank picks the same entries
(`configuration_deepseek_v4.py`). That is a valid optimization, but the ranks
must agree on the selection *exactly* — a rank that picks a different entry set
computes different attention — and replication makes that true by construction
rather than by a collective.

## Still to do

- **The gather.** Masking is correctness; it buys nothing. Attention still
  materializes the whole compressed capacity per token
  (`mla_attention_reference`'s K/V einsums), and `_compressed_history` still
  gathers the full block-table capacity. Replacing the mask with an
  `index_select` down to `index_topk + sliding_window` rows is the actual point
  of sparse attention. `index_topk` is a compile-time constant, so it stays
  Dynamo-static.
- **NKI.** The kernels have no token-level gather — only contiguous
  `bound_min/bound_max`, SWA, and a block-granular `active_blocks_table`. The
  gather stays in the torch path for now.
- **Compile cost.** `DeepseekV4Attention.forward` already unrolls one attention
  body per token at trace time. The indexer adds a second compressor and a top-k
  to every unrolled c4 body; watch this before scaling `max_model_len`.
