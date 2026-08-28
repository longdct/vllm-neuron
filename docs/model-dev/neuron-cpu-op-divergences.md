# Torch ops that behave differently on Neuron than on CPU

An inventory of every case this project has actually found where the same torch
code produces a different result — or no result — on a Neuron device than it
does on CPU.

**Read the scope honestly.** This is a list of what has been *found*, mostly by
debugging a wrong answer backwards, not the output of a systematic audit. The
real DeepSeek-V4 graph alone contains **368 distinct op targets**; the entries
below cover roughly a dozen. **Absence from this list is not evidence of
correctness** — every entry here spent time as an unknown-unknown, and several
looked healthy for weeks. Treat it as a list of known landmines in a field that
has not been swept.

**Which stack.** The historical inventory below was observed on the retired
lite/XLA route. The plugin now uses TorchNeuron Native from the external
`torch-neuronx` package with `backend="neuron"`. The workarounds remain because
several divergences only reproduced in production-sized graphs.

On `torch-neuronx==2.12.3.0.0+aa8779f4.dev` with torch 2.12.1, the full harness
was re-characterized on Trn2. Entries 1 (`split`), 6 (`topk` index dtype), and 9
(stride-2 partial store) now match CPU. Entry 11 remains rejected, entries 2,
4, 5, 7, and 10 match in isolation but retain their workarounds, and the
kernel-level wide-gather case remains outside this torch-op harness.

## Quick reference

Ordered by how hard they are to notice, worst first.

| # | Op / form | What happens | Safe form |
|---|---|---|---|
| 1 | `Tensor.split(list, dim≠0)` | **wrong data, silently** | slice, `chunk`, or int-size `split` |
| 2 | `torch.cat` of a small rotary suffix, rank-4 | suffix lowered as **dead/zero** operand | `torch.index_copy` |
| 3 | wide NKI gather above the DVE group size | **corrupts results** (NKILIB-1592) | tile to `max_free_dim` (2¹⁴) |
| 4 | `int32`→`uint32` cast that is not the final op | dtype **silently lost** | cast as the last operation |
| 5 | XLA scatter with mask width `< K` | wrong / rejected scatter | pad mask width to `≥ K` |
| 6 | `torch.topk(...)[1]` | returns **uint32**; CPU returns int64 | `.long()` immediately |
| 7 | non-contiguous source into `copy_` / `index_put_` | rejected at graph capture | `.contiguous()` on the host |
| 8 | `.to(device=…, dtype=…)` in one call | rejected | cast on host, then transfer |
| 9 | stride-2 partial store (`x[…, p % 2] = …`) | does not lower | write whole pairs |
| 10 | `[:, 1:].contiguous()` on device | unresolvable view | build the tensor pre-sliced |
| 11 | data-dependent branching (`if t.any():`) | Dynamo raises, no graph | guard with `torch.compiler.is_compiling()` |

Rows 1–5 are the dangerous class: **they produce plausible numbers.** Rows 6–11
fail loudly or produce an obviously wrong type, which is a gift by comparison.

## Reproducing, and what actually reproduces

`tools/check_neuron_op_divergences.py` drives the unfixed form of each entry and
diffs it against CPU. It is a characterization harness, not a test suite: a
documented divergence is the *expected* result, so it exits non-zero only when
reality stops matching this document — a tripwire in both directions.

```bash
PATH="$VENV/bin:$PATH" PYTHONPATH=$PWD \
NEURON_VISIBLE_DEVICES=0 NEURON_RT_VISIBLE_CORES=0 \
VLLM_CACHE_ROOT=$(mktemp -d) \
  $VENV/bin/python tools/check_neuron_op_divergences.py
```

Measured on torch 2.11/torch-xla 2.11/lite 2.11 **and** on torch 2.12/torch-xla
2.12/lite 2.12 — byte-for-byte the same table on both:

| # | op form | result |
|---|---|---|
| 1 | `split(list, dim=-1)` | **DIVERGES** — 56 of 192 elements differ, max\|diff\| 112 |
| 1 | `split(int, dim=-1)` *(control)* | MATCHES |
| 6 | `topk(...)[1]` | **DIVERGES** — int64 on CPU, **uint32** on device |
| 9 | stride-2 partial store | **REJECTED** — `aten::index_put` fails to lower |
| 11 | data-dependent branching | **REJECTED** — `Data-dependent branching` |
| 2, 4, 5, 7, 10 | — | **not reproduced standalone** (see below) |
| 3 | wide NKI gather | skipped — kernel-level, not a torch op |

**Entries 2, 4, 5, 7 and 10 do not reproduce in isolation.** They were observed
inside a real model graph, and a minimal form of each matches CPU here. That
does **not** clear them, and the workarounds stay: for the entries whose
mechanism is understood, rank, surrounding ops and graph size all mattered — the
rank-4 `cat` was correct at rank 3 with identical code. Read those rows as "not
reproduced", never as "fixed".

Two things the harness had to learn the hard way, both worth copying into any
similar tool:

* **Each check runs in its own subprocess.** This isolates compiler/runtime
  failures and keeps one unsupported form from contaminating later checks.
* **Compile with `fullgraph=True`.** At the default `fullgraph=False`, Dynamo
  graph-breaks around a data-dependent branch and runs it in eager, so entry 11
  passes and hides the exact limitation under test. `fullgraph=True` is also
  what the model runner uses.

## The silent ones

### 1. `Tensor.split` with a list of sizes, on any dim but 0

The worst entry on this list, because nothing fails and the bad data is *real,
adjacent values* rather than garbage — so the model stays statistically healthy
while being wrong.

```python
pre, post, comb = projected.split([4, 4, 16], dim=-1)   # WRONG on Neuron
pre, post, comb = projected[..., :4], projected[..., 4:8], projected[..., 8:]
```

Affected: any *list*-of-sizes form on a non-zero dim. An int size, `chunk`,
`narrow`, `index_select` and plain slicing are all correct.

The cause is visible in the IR, not merely inferred. For
`arange(8*24).reshape(8,24).split([8,8,8], dim=-1)`, the first HLO — emitted by
torch-xla before any lite pass — flattens and takes *contiguous* runs:

```
parameter 8x24 ; reshape 192 ; slice 64 [start=0  limit=64]  ; reshape 8x8
                 reshape 192 ; slice 64 [start=8  limit=72]  ; reshape 8x8
                 reshape 192 ; slice 64 [start=16 limit=80]  ; reshape 8x8
```

Column offsets used as flat offsets, row stride 24 ignored. Row 0 coincides;
every later row is wrong. `neuronx-cc` is not at fault — it compiles that HLO
faithfully, and the same compiler version serves the stack that gets it right.

Reproducer: `tools/repro_neuron_split_lowering.py` (exits non-zero while
present). Cost when missed: device generated entirely different text, step-0
logits off by **20.69**. See
[deepseek-v4-real-weight-validation.md](deepseek-v4-real-weight-validation.md).

### 2. `torch.cat` of a small rotary suffix onto a rank-4 tensor

Neuron lowered the rotated suffix as a dead/zero concat operand, zeroing the two
rotary channels — while the otherwise identical rank-3 KV path was correct. So
rank matters, and a passing test at one rank proves nothing at another.

```python
return torch.index_copy(x, -1, rotary_indices, rotated)
```

`vllm_neuron/model/deepseek_v4/attention.py`, commit `15e548c`.

### 3. Wide gathers above the DVE free-dimension limit

A gather wider than the ISA group size splits into multiple internal groups, and
**the multi-group form corrupts results on hardware** (NKILIB-1592). Wide gathers
must be tiled to `max_free_dim` (2¹⁴).
`vllm_neuron/functional/vendored_kernels/rotational_topk/rotational_topk_utils.py`.

### 4. Dtypes that do not survive XLA lowering

Compute in `int32` and cast to `uint32` as the **final** op, or the dtype is
lost: `vllm_neuron/functional/moe/build_all_gatherv_metadata.py`.

### 5. Scatter mask narrower than K

XLA's scatter lowering needs the mask width padded to at least `K`:
`vllm_neuron/functional/moe/build_all2all_dispatch_metadata.py`.

## The loud ones

### 6. `torch.topk` returns unsigned indices

`torch.topk(...)[1]` is **uint32** on Neuron, int64 on CPU. Values are correct,
so anything that only *gathers* is fine. What breaks is the `-1` sentinel idiom:
`full_like(chosen, -1)` wraps to 4294967295 and `>= 0` is vacuously true, so the
sentinel is handed to the next scatter as a real index — surfacing as
`nrta status=1006`, reported against the whole NEFF with no op name.

```python
chosen = torch.topk(masked, k, dim=-1)[1].long()
invalid = (chosen >= threshold) | (chosen < 0) | (chosen >= entries)
```

Treat out-of-range as "no selection", never clamp — clamping turns a meaningless
pick into a confident selection of the last entry. Only appears in rows that are
entirely `-inf`, which is why a probe on representative inputs misses it.
`vllm_neuron/model/deepseek_v4/indexer.py`.

### 7. Non-contiguous tensors into `copy_` / `index_put_`

Rejected on device; passes on CPU. Worse, at load time Neuron can also refuse to
run the `.contiguous()` that would fix it — so build on the host and let `copy_`
cross the boundary. `vllm_neuron/functional/attention/attention_decode.py`.

### 8. `.to(device=…, dtype=…)` in one step

Rejected ("Expected self.dtype() == dst.dtype()"); CPU performs it silently.
Cast on the host first. Bites weight loading whenever checkpoint and parameter
dtypes differ, and stock `transformers.modeling_rope_utils` hits it too.
`vllm_neuron/model/deepseek_v4/weight_loaders.py`.

### 9. Stride-2 partial stores

`k_cache[..., :, p % 2] = ...` needs a tensor-valued inner index and a stride-2
partial store, which **does not lower**. Write whole pairs instead — reshape to
`[num_pairs, d_head, 2]` and `index_put_` on the leading dims, leaving a dense
contiguous payload. `vllm_neuron/functional/attention/attention_decode.py`.

### 10. `[:, 1:].contiguous()` on device

Slicing a stacked tensor produces a non-contiguous view that `.contiguous()`
cannot resolve on device. Build the already-contiguous tensor instead of slicing
one. `vllm_neuron/model/llama3/eagle3_model.py`.

### 11. Data-dependent branching

Reading tensor *values* for control flow (`if t.min() < 0:`) makes Dynamo raise
`Unsupported: Data-dependent branching` — it does not graph-break, it fails.
Not a divergence so much as a capability gap, but it changes how code must be
written:

```python
if not torch.compiler.is_compiling():
    if input_ids.max() >= table.shape[0]:
        raise ValueError("token id is outside the table")
```

Shape-based checks are safe and belong outside the guard.

## Testing an op yourself

Do not use the full model — 7–13 minutes per compile. Compile the suspect module
alone (~1.5 s) and diff against CPU:

```python
import vllm_neuron                     # registers the backends
from vllm_neuron.envs import get_compile_backend_name

compiled = torch.compile(module.to("neuron:0"), backend=get_compile_backend_name())
torch.testing.assert_close(expected_cpu, compiled(x.to("neuron:0")).to("cpu"))
```

`tools/deepseek_v4/check_mhc_device.py` and `check_indexer_device.py` are worked
examples. Set `NEURON_RT_VISIBLE_CORES` alongside `NEURON_VISIBLE_DEVICES`
outside vLLM, and a private `VLLM_CACHE_ROOT` so a stale NEFF cannot answer for
you.

Four ways such a test lies, all of which have happened here:

* **the compile cache** replays another stack's NEFF — check for `Local cache
  miss` then `Compiling...`;
* **a silent CPU fallback** passes and looks like correctness — keep a negative
  control that must fail;
* **testing the workaround** instead of the backend — the shipped code already
  casts/slices around these, so it passes everywhere;
* **one input regime** — the `topk` defect needs an all-`-inf` row; the
  compressed-entry off-by-one appeared only at `position % ratio == ratio - 1`.

And when comparing: **per element and per position, never in aggregate.** A
defect exact at most positions and wrong at a few reads as floating-point noise
in any summary statistic.

## Related

* [neuron-lowering-pitfalls.md](neuron-lowering-pitfalls.md) — the same defects
  with full mechanism write-ups and how to hunt a new one.
* [neuron-lowering-stacks.md](neuron-lowering-stacks.md) — which stack a claim
  applies to, and how to reproduce across all three.
