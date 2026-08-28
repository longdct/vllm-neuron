# DeepSeek-V4 MLA compile explosion: it moved, it did not go away

Status: mechanism identified and graph effect measured as of 2026-08-26; the
whole-model compile-time effect is NOT yet measured (no whole-model run has ever
run to completion).  Supersedes the diagnosis in
[deepseek-v4-q512-mla-compile-explosion.md](deepseek-v4-q512-mla-compile-explosion.md),
whose recommended fix is **not implementable** on this stack (see
"Why the previous recommendation cannot work").

## Summary

The streaming-MLA rewrite genuinely fixed the problem it targeted. The NKI
kernel artifacts shrank by more than an order of magnitude and every standalone
component probe now compiles in under a minute. Whole-model cold compilation
still ran for 2h52m without producing a NEFF.

The cost did not disappear; it moved from **inside** the NKI kernel to the
**outer graph**. The rewrite kept each compiled kernel body small by slicing the
query axis in Python, which emits one opaque custom call per tile. At Q512 with
an 8-query tile that is 64 custom calls per attention layer.

The three-layer tiny model therefore hands `neuronx-cc` a graph containing
**233 opaque NKI custom calls and 1108 `dynamic_slice` operations**, and
the working hypothesis is that compilation is superlinear in the number of
opaque call sites.  That is demonstrated on single-layer probes; it is not yet
demonstrated on the whole model.

This is a genuine dilemma rather than an oversight, and it is why "fixing the
kernels" did not help:

| Query tile | Custom calls per layer | Kernel body | What explodes |
| ---: | ---: | ---: | --- |
| 512 (no tiling) | 1 | 256 queries unrolled per program | inside NKI |
| 8 (current) | 64 | 4 queries per program | outer graph |

Both ends of the tradeoff explode. Only a *runtime* query loop escapes it.

## Evidence

### The kernel fix worked

Standalone probes (`tools/deepseek_v4/compile_runtime_mla.py`), before vs after
the streaming rewrite:

| Component | Before (artifact / records) | After (artifact / records) | After (time) |
| --- | ---: | ---: | ---: |
| Sliding MLA Q512 | 910218 B / 19474 | 12589 B / 300 | 20.3 s |
| CSA MLA Q512 | 1487197 B / 27674 | 40150 B / 1130 | 38.2 s |
| HCA MLA Q512 | 2037539 B / 35866 | 67697 B / 1962 | 51.1 s |

Every component passes. The whole model still does not.

### The whole-model graph

The failing run is `/tmp/dsv4-q512-tp2-streaming-eVt8YbhE` (started 14:09,
killed 17:01, `cold.time` empty). Its compile cache shows 29 graphs started and
only 17 finished — all 17 completed by 14:11:14, every one a small NKI
sub-compile of 9-125 KB. Nothing completed in the remaining 2h50m.

The in-flight graph was dumped as StableHLO MLIR bytecode (not XLA HLO proto —
torch-neuronx 2.12 native path):

```bash
/tmp/neuron_hlo_845579_11.pb   # 110673 bytes, MLIR22.0.0git bytecode
```

Decoding it and counting the base64 `backend_config` payloads gives the call
sites, for a **three-layer** model:

| NKI kernel | Custom-call sites |
| --- | ---: |
| `nki_mla._paged_shared_latent_mla_kernel` | 128 |
| `nki_mla._paged_sliding_latent_mla_kernel` | 64 |
| `nki_indexer._projected_bf16_indexer_kernel` | 32 |
| `moe.moe_cte._torch_compatible_moe_cte_kernel` | 3 |
| `nkilib` find_nonzero / indexed_flatten | 6 |
| **total** | **233** |

Op histogram of the same graph — only ~4700 instructions, but note what
dominates:

```
1108 stablehlo.dynamic_slice     <- operand slices feeding the call sites
 653 stablehlo.reshape
 494 stablehlo.broadcast_in_dim
 234 stablehlo.custom_call
  63 stablehlo.concatenate       <- torch.cat rejoining the tiles
```

The 1108 slices are arithmetic, not coincidence. Per shared-latent layer the
dispatcher slices five operands (query, sliding slots, sliding mask, compressed
slots, compressed mask) 64 times = 320; two such layers = 640; the sliding-only
layer slices three operands 64 times = 192; plus the indexer's 32 calls. The
graph is small in ops but pathological in opaque nodes.

### Source of the fan-out

`vllm_neuron/model/deepseek_v4/nki_mla.py`, in `paged_shared_latent_mla`:

```python
return torch.cat(
    [launch(start, start + _PREFILL_QUERY_TILE) for start in range(0, q_count, 8)],
    dim=0,
)
```

`launch()` slices every query-derived operand and invokes
`_wrapped_paged_*_latent_mla[2](...)`. With `_PREFILL_QUERY_TILE = 8` and
`q_count = 512` that is a Python-level loop emitting 64 separate custom calls.

### Call sites dominate kernel body

Sweeping the tile on the single-layer CSA probe (Q512, compressed 512), holding
everything else fixed:

| Tile | Call sites (1 layer) | Artifact | Alloc records | Compile wall |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 64 | 40343 B | 1138 | 39.6 s |
| 16 | 32 | 76522 B | 2258 | 29.0 s |
| 32 | 16 | 222862 B | 4498 | 26.3 s |
| 64 | 8 | 484704 B | 8978 | 23.6 s |

The kernel body grows exactly linearly with the tile (records double each step,
as static unrolling predicts) and yet **total compile time falls**. Within this
range the per-call-site cost dominates the per-instruction cost.  Whether this
extrapolates to the whole model is exactly what has not yet been measured -- see
the next section.

### Whole-model effect: graph structure confirmed, timing NOT yet established

Re-running the exact failing configuration (TP2/EP2, LNC2, BF16, Q512, real
weights, empty cache) with only the tile changed gives a much smaller graph:

| | NKI call sites | `dynamic_slice` |
| --- | ---: | ---: |
| tile 8 (baseline) | 234 | 1108 |
| tile 64 | 66 | 380 |

MLA call sites fall from 192 to 24. Both counts are read off the actual
`module.mlir` handed to `neuronx-cc`.

**A speedup has not been demonstrated.** Beware of two traps here, both of
which produced wrong conclusions earlier in this investigation:

*Trap 1 — capture is not compilation.* Neuron compiles asynchronously. The
"Capturing prefill graphs" -> "Capturing decode graphs" interval measures graph
*tracing and dispatch*, not compile time. Measured, that interval is
essentially identical in both configurations:

| | prefill capture interval |
| --- | --- |
| tile 8 | 14:10:23 -> 14:11:13 = 50 s |
| tile 64 | 19:33:10 -> 19:34:02 = 52 s |

Reading the tile-64 run's 52 s as "prefill now compiles in under a minute" is
wrong; the baseline does the same thing in the same time.

*Trap 2 — "Capturing decode" in the log does not mean prefill finished
compiling.* To find out what a `walrus_driver` process is actually working on,
read the `module.mlir` in its working directory:

```bash
pgrep -f "neuronx-cc compile module.mlir" | head -1 | xargs -I{} readlink /proc/{}/cwd
python tools/deepseek_v4/count_graph_callsites.py <that dir>/module.mlir
```

Doing that showed the graph grinding for 67 minutes was the **prefill** graph
(66 call sites), not a decode graph. There is no evidence of a separate decode
bottleneck; the earlier claim of one in this document was that
misattribution and has been removed.

Neither whole-model run has ever completed: the baseline was killed at 2h50m
and the tile-64 run at 67 minutes, both still compiling prefill. Until one runs
to completion there is no before/after compile time to compare.

What *is* established end to end is the single-layer probe sweep in the table
above, where the compiles genuinely finish and fewer call sites win.

Note what dominates the remaining prefill fan-out after the tile change: the
**CSA indexer at 32 call sites**, which does its own Q16 host-side tiling and is
untouched by it. It is the next target.

## The dominant cost: packed compressor state gathering

Call-site fan-out is real but it is **not** the main cost. Later DMA-profiler
and source attribution corrected the initial diagnosis: the large indirect
load is `attention.py::gather_recent_window_batched`, called by
`DeepseekV4Compressor.forward_packed` for every packed query. It is not the
compressed-history gather in `nki_mla._stream_paged_query`.

From `log-neuron-cc.txt` (the compiler writes it into the neuronx-cc working
directory, e.g. `/tmp/neuron_backend/neuronx-cc/<hash>/`):

```
nc00/sg04  before unroll:                      1,780 instructions
nc00/sg04  Total count after Unroll:         405,263
nc00/sg04  DMACopy:                          327,900
nc00/sg04  Unrolled DGE count with Dynamic AP: 327,808
nc00/sg06  Unrolled DGE count with Dynamic AP: 262,408
```

Every other subgraph has at most 24. The numerical resemblance to
`Q x MLA-history` led to the earlier `sg04 = CSA MLA` / `sg06 = HCA MLA`
claim, but controlled before/after captures disproved it: changing the HCA MLA
compressed span changed `module.mlir` while leaving every DGE count identical.

The attributed large gathers are instead the packed compressors' raw state:

* HCA contributes the 524,288-instance gather associated with `sg06`.
* CSA contributes separate 524,288- and 131,072-instance gathers for the
  512-dimensional outer compressor and 128-dimensional indexer compressor,
  associated with `sg04`.

The boundary-only compressor path removes those per-query gathers entirely.
It constructs only completion candidates in the outer graph and performs the
paged gather plus gated reduction inside one runtime-loop NKI body.

### Why that structure is fatal to the backend

Each of those 262k-328k descriptors reads the same shared DRAM cache tensor, so
the dependency graph gains a node with `Max Readers: 262,145`. The compiler then
wedges in `anti_dependency_analyzer_post_shared_dram`:

```
anti_dependency_analyzer_post_shared_dram: started=18 finished=16 pending=2
```

The two unfinished ones are `nc00/sg04` and `nc01/sg04`. A run left in that
state made no log output for over three hours while RSS climbed from 14 GiB to
23 GiB and CPU accumulated at ~2 threads. It was killed at 3h29m, still inside
that pass, having never emitted a model NEFF.

### Measured, and ruled out

- **`oob_mode=nisa.oob_mode.skip` on the gather.** Hypothesis was that per-row
  out-of-bounds predication forces the unroll. Tested by removing it (the slots
  are already clamped by `safe_slots()`, so it is semantically redundant): the
  unroll counts are **bit-identical** — 327,808 / 262,408 / DMACopy 327,900.
  Not the cause.
- **`--internal_dynamic_dma_scratch_size_per_partition`.** Cannot be raised from
  `NEURON_CC_FLAGS`; `neuronx-cc` rejects it as unrecognized (it is an internal
  Tensorizer option). Untested. Note `neuronx-cc compile --help` crashes in this
  build, so options cannot be enumerated that way.

### Where to go next

The DGE levels already include `vector_dynamic_offsets`, so the hardware gather
exists and is not being used for this shape. Two directions, in order:

1. **Reduce the rows gathered.** The union of rows needed by a 512-query chunk
   is far smaller than `Q x history`. The sliding window of 512 consecutive
   queries spans at most 640 distinct positions, not 512 x 128 = 65,536, so
   gathering the chunk's span once and letting each query slice its own window
   out of it is a ~100x reduction on that stream. Compressed/CSA entries are
   genuinely per-query, but they are selected at block granularity, so whole
   blocks can be gathered and shared.
2. **Get one descriptor per gather instead of per row.** Worth raising with the
   Neuron compiler team, with `module.mlir` plus these unroll counts: a
   128-row x 512-element vector-indirect gather is exactly what a DGE should
   express in one descriptor.

Note this is a **runtime** problem too, not only a compile-time one. ~590k DMA
descriptors per core per forward pass will not execute quickly either.

## Implemented: the tile span gather (2026-08-27)

Direction 1 above is now implemented for the sliding stream, in
`nki_mla.py::_build_sliding_span`.

### The mechanism

`nisa.dma_copy` routes to the SWDGE `dma_copy_indirect` path -- the one the
`unroll` pass expands per row -- **only when `vector_offset` is set**
(`nki/isa/_copy.py:157-221`). An affine `.ap()`, or a plain static slice, takes
the ordinary `dma_copy` path and costs **one descriptor** however many rows it
covers. So the quantity to minimise is not rows gathered but
*address-independent runs*.

A second fact makes the addressing trivial. `nl.program_id` is a **trace-time
Python int**, not a runtime register:

> Multi-trace approach: kernel is traced LNC times with different
> program_id_value (0, 1, ..., lnc-1), producing specialized functions.
> -- `nki/_backends/mlir_tracer/__init__.py:214`

So `q_idx` is a compile-time constant, each LNC program can build only the span
*it* needs, and the per-query window read is a static slice --
`source[row : row + width, :]` -- with no `scalar_offset` machinery at all.

### The construction

`recent_sliding_logical_indices` gives the query at position `p` the logicals
`p-127..p`, so consecutive queries overlap by 127/128. A run of `n_q` queries
therefore needs only `count + n_q - 1` distinct rows: all of `slots[first_q]`
followed by the tail of `slots[first_q + n_q - 1]`. Both are **static** indices
into the kernel's own `slots` argument, so no new kernel arguments and no host
changes were needed. The span lands in `nl.private_hbm` (195 KB at tile 64).

Cost per query run: `count + n_q - 1` descriptors for the span plus one affine
read per query, instead of `n_q x count`.

### Contract narrowing -- read this before reusing the kernel

The span assumes **row `q` is row `0` advanced by `q`**. That holds for the
consecutive positions of a prefill tile, and trivially at `q_count == 1`
(decode), where the span degenerates to `slots[0]`.

It does **not** hold for an arbitrary slot table. Two consequences:

* The prefill **padding tail** repeats the last real position
  (`neuron_model_runner.py:3385-3400`), so padded rows have identical rather
  than shifted windows and their outputs are wrong. They are discarded
  downstream and stay finite (a convex combination of real cache rows), which
  `test_paged_span_requires_a_shifted_window` pins.
* `test_paged_streaming_mla_q8_lnc2_matches_oracle` previously built
  `torch.arange(128).repeat(8, 1)` -- every query the same window -- which the
  span cannot reproduce. It now builds a shifted table, matching what the model
  actually passes.

If unconditional exactness for padded rows is ever wanted, plumb an explicit
`window_base [T]` from `model.py::_forward_packed`, where positions are known,
and feed it through `scalar_offset` instead of the static `offset`.

### Validated

* `test_paged_span_gather_is_bit_exact_against_the_per_row_gather` runs both
  paths under the simulator on identical inputs and asserts `rtol=0, atol=0`.
  This is the only assertion in the suite that can establish it: every oracle
  comparison is at `rtol=0.025, atol=0.025`, because the kernel is BF16 online
  softmax against an FP32 full-softmax reference, so a genuine reordering would
  hide inside the tolerance.
* `VLLM_NEURON_DSV4_MLA_SPAN_GATHER=0` restores the per-row gather; that is what
  the bit-exactness test compares against, so keep it working.

### Not fixed by this

The **CSA compressed stream** (layer 1, `index_topk=512`) is untouched: its rows
are an arbitrary per-query `torch.topk` selection, so no bit-exact reduction
exists. At Q=512 it remains `512 x 512 = 262,144` descriptors, the same order as
the count already observed to wedge the backend. Shrinking it needs
block-granular or query-group-shared selection, which changes *which* entries are
chosen and so changes model outputs -- it needs accuracy validation, not just a
bit-exactness check.

`_PREFILL_QUERY_TILE` moved 8 -> 64 because the span's `count - 1` overhead
amortises over the tile, and `_DIRECT_QUERY_BUCKETS` gained 128 and 256 so the
prefill bucket can be lowered to get under the CSA wall.

## Implemented: the uniform compressed span (HCA, 2026-08-27)

Direction 1 is now also implemented for the **HCA** compressed stream, in
`nki_mla.py::_build_uniform_span`. CSA is still untouched, and is now
essentially all of what remains.

### Why HCA's rows repeat

`recent_compressed_logical_indices` (`attention.py`) builds each query's suffix
as

```python
visible = visible_compressed_entries(positions.long(), compress_ratio)  # (pos+1)//ratio
used    = visible.clamp(min=0, max=count)
start   = (visible - used)[:, None]
logical = start + torch.arange(count)[None, :]
valid   = offsets < used[:, None]
```

When `count >= capacity_entries` no reachable position can push `visible` past
`count`, so `used == visible` and **`start == 0` identically**. Every query then
asks for the same logical entries `0 .. count-1`; only `valid` differs. And
`logical_to_physical_slots_batched` maps those through one request's block table,
which for a single request depends only on the entry index. So `compressed_slots`
is `Q` identical rows, and the kernel was gathering all `Q` of them.

That premise is now explicit rather than accidental. `model.py::_forward_packed`
sizes the count from the addressable capacity and rounds **up** to the nearest
compiled bucket:

```python
capacity_entries = block_table_tensor.shape[1] * (mla_raw_block_size // ratio)
compressed_count = min(b for b in _HCA_COUNT_BUCKETS if b >= capacity_entries)
```

with `_HCA_COUNT_BUCKETS = (32, 64, 128, 256, 512, 1024)`. This is vLLM's own
rule -- `sparse_swa.py:218-223` bounds HCA entries by
`cdiv(prefill_max_model_len, compress_ratio)` -- and replaces an ad-hoc
32/256/1024 ladder that over-provisioned 2x at 64K (1024 where 512 suffices).
Rounding down would break `start == 0` and with it the span; never do it.

### The construction

`_build_uniform_span` gathers `slots[last_q]` once into `nl.private_hbm` and
every query reads the affine slice `span[start : start + width]` -- one
descriptor, through `_stream_paged_query`'s existing `span_base` branch, with no
change to that function at all. `span_base` is the trace-time int `0`.

Two details are load-bearing:

* **`last_q`, not `first_q`.** `safe_slots()` zeroes the physical slot of any
  entry a query cannot see, so the rows are identical only up to each query's
  `used`. `used` is non-decreasing within a launch (positions increase, and
  prefill padding repeats the last real position rather than going backwards),
  so the run's last query holds the longest valid prefix and its row is a
  superset of every earlier query's. Building from the first query would feed a
  later query cache row 0 where it can see a real entry.
  `test_uniform_compressed_span_matches_the_per_query_oracle` and the
  `partial=True` bit-exactness cases both fail if this is flipped -- verified by
  making that exact edit.
* **128-row chunking.** `vector_offset` is per-partition and caps at 128 rows,
  so a count above 128 needs one `_gather_rows` per chunk. `_build_sliding_span`
  gets away with a single call only because its count is exactly 128.

### Why it is a separate kernel specialization

`compressed_uniform` is a trace-time Python bool parameter on
`_paged_shared_latent_mla_kernel`, threaded from
`SharedLatentMLAInputs.compressed_uniform` through the microchunk recursion and
`launch()`. It cannot be inferred from shapes: HCA's capacity-derived count and
CSA's `index_topk` are **both 512** at 64K context. Two specializations where
there was one is the intended cost.

### Contract narrowing

The span is one request's block table, so a launch may not mix requests. The CSA
path already refused that (`IndexerModule.forward_packed`, "supports one TP1
request"); HCA layers have no indexer, so `_forward_packed` now states the same
check at the HCA call site instead of inheriting it silently. Multi-request
prefill was already outside the NKI milestone's contract.

### Validated

* `test_uniform_compressed_span_is_bit_exact_against_the_per_row_gather`
  (counts 64 and 256, full and partial prefixes, scattered block tables) runs
  both gather paths under the simulator on identical inputs at `rtol=0, atol=0`.
* `test_uniform_compressed_span_matches_the_per_query_oracle` pins the span
  source against ground truth, which agreeing-with-itself cannot.
* `test_uniform_compressed_span_handles_the_q1_decode_launch` covers the
  degenerate decode launch, where `last_q == first_q == 0`.
* `test_capacity_sized_compressed_rows_are_identical_for_every_query`
  (`test/unit`) pins the `start == 0` invariant where the arithmetic lives.

### Measured -- and what the measurement did NOT establish

**The predicted descriptor reduction was not observed at the whole-model level.**
Read this before quoting any speedup from this change.

What *is* established:

* The HCA layer's entry count does drop. Instrumenting `_forward_packed` on the
  tiny model at TP2/EP2, `--max-model-len 65536`, Q512 prints
  `layer=model.layers.2.self_attn width=512 capacity_entries=512 count=512`,
  where the old ladder returned 1024 (raw capacity `512 x 128 = 65,536 > 32,768`).
  That is the 2x over-provisioning the vLLM rule removes.
* The compiled graph genuinely changes. The `module.mlir` handed to `neuronx-cc`
  goes from 92,449 / 92,454 bytes (sha `65c4261a` / `cb6c3c15`) to 92,114 /
  92,109 bytes (sha `7d01caad` / `b6f6def1`).

What is **not** established -- the plan predicted the HCA subgraph would fall
from ~524,288 unrolled DGEs to ~10,000 while CSA's ~262,144 stayed put. It did
not. Every subgraph's `Unrolled DGE count with Dynamic AP` is byte-identical
before and after:

| subgraph | baseline | with the fix |
| --- | ---: | ---: |
| nc00/sg04 | 328,400 | 328,400 |
| nc01/sg04 | 328,320 | 328,320 |
| nc00, nc01 / sg06 | 262,664 | 262,664 |
| sg00 / sg01 / sg02 / sg03 / sg05 / sg07 / sg08 | 2 / 260 / 24 / 0 / 24 / 12 / 1 | identical |

Both runs: pristine `worktree-mla-callsite-explosion` vs this branch, same host,
same config (`--max-model-len 65536 --max-num-batched-tokens 512
--prefill-segment-buckets 512 --num-gpu-blocks-override 4096`, TP2/EP2,
`NKI_ENABLE_TRACE_CACHE=0`, `VLLM_NEURON_DSV4_MLA_SPAN_GATHER=1`, fresh
`VLLM_CACHE_ROOT`), run sequentially in the foreground. The counts were read from
the logs of the processes whose working directory holds the 92 KB `module.mlir`
above, so they are the model graph's own numbers, not a sub-compile's.

The result is now understood: this optimization targeted the wrong gather.
The HCA capacity rule and uniform MLA span remain valid secondary
optimizations, but they cannot be credited with removing the compiler's large
dependency node. The hot gather was still present earlier in
`DeepseekV4Compressor.forward_packed`, after projection/state scatter and
before `compress_hca_chunk`/`compress_csa_chunk`.

A shorter context is not a substitute: at `--max-model-len 2048` the MLA block
table is 16 columns wide, so `capacity_entries` is 16 and both the old ladder and
the new rule pick 32. The change is inert there by construction, and indeed all
nine subgraph counts matched exactly.

Two measurement traps were hit while producing the numbers above; both silently
manufacture "identical before/after" results:

* **`walrus_driver` outlives its parent.** A capture loop that greps for it
  system-wide will copy the *previous* run's `log-neuron-cc.txt`. Refuse to start
  a run while any `walrus_driver` is alive, and hash the working directory's
  `module.mlir` to confirm which graph a log belongs to.
* **Two runs sharing one output directory.** Each does `rm -rf $OUT` at start, so
  the second deletes the first's captured logs mid-flight.

Also unchanged by this work, and pre-existing on the baseline branch: decode
graph extraction fails at this configuration with `Model graph extraction failed
for decode (batch=1, seq=1024)`.

### What to do next on the measurement

Measure the boundary-only compressor in two stages using retained whole-graph
captures. After HCA, require the `attention.py` 524,288-instance indirect load
to disappear and `sg06` DGE count to fall by at least 90%. After CSA outer and
indexer integration, require both its 524,288- and 131,072-instance loads to
disappear and `sg04` to fall by at least 90%. The `Max Readers: 262,145` node
must be absent. Keep the HCA uniform-span measurement separate.

## Implemented: boundary-only paged compressor reduction (2026-08-28)

`nki_compressor.py` now owns a generic runtime-loop kernel for HCA c128 and
CSA c4 (outer dimension 512 and indexer dimension 128). Projection and FP32 raw
state-cache scatter remain Torch operations. The wrapper derives a fixed set
of completion candidates from the first packed position:

* Q1 has one candidate, valid only on a compression boundary;
* larger Q has `ceil(Q / ratio)` candidates spaced by `ratio`;
* candidates are valid only while the local index exists, positions remain
  contiguous, ownership is unchanged, the output slot is valid, and the paged
  raw-state mapping is valid;
* HCA builds `[candidates, 128]` raw slots; CSA builds
  `[candidates, 8]`, selecting prior-Ca plus current-Cb exactly as the portable
  `compress_csa_chunk` oracle does.

The NKI body uses `nl.fori_loop` and register-offset address patterns. Candidate
count changes the loop bound, not the instruction body. One candidate launches
at LNC1; multiple candidates use LNC2. Invalid candidates execute on sanitized
finite inputs and write exact zero, while missing early-history rows are masked
before softmax. RMSNorm, RoPE, and cache mutation remain outside the kernel.

`VLLM_NEURON_DSV4_NKI_COMPRESSOR=0` restores the portable per-query path for
A/B comparison or emergency fallback. `compile_runtime_compressor.py` reports
wall time, process peak RSS, COLZ size, and allocation-record count for Q1,
Q512, and Q1024 specializations.

Simulator comparisons cover HCA, CSA-512, CSA-indexer-128, Q1 boundary and
non-boundary decode, Q512/LNC2, non-zero starts, early partial overlap, page
crossings, shuffled/null block tables, and padded tails at the existing
`rtol=0.025, atol=0.025` tolerance. Whole-graph DGE and final TP2/EP2 cold/warm
acceptance remain hardware measurements, not claims established by simulator
success.

### Measured boundary-compressor structure

Cold component compiles in the recorded TorchNeuron Native environment used
production FP32 compressor-state pages and disabled the NKI trace cache:

| geometry | query | candidates | wall time | process peak RSS | COLZ | allocation records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HCA-512 | 1 | 1 | 4.15 s | 664 MiB | 5,858 B | 65 |
| HCA-512 | 512 | 4 | 4.86 s | 679 MiB | 7,945 B | 130 |
| HCA-512 | 1024 | 8 | 4.89 s | 679 MiB | 7,954 B | 130 |
| CSA-512 | 512 | 128 | 5.02 s | 679 MiB | 8,750 B | 132 |
| CSA-indexer-128 | 512 | 128 | 5.02 s | 668 MiB | 5,068 B | 54 |

The one-time Q1-to-Q512 HCA change is the required LNC1-to-LNC2
specialization. Q512-to-Q1024 adds nine bytes and no allocation records. No
artifact contains a `[Q, window, state_width]` allocation.

### One-rank whole-graph diagnostic

A 2026-08-28 dummy-weight TP1 capture used Q512, maximum length 131,072,
decode buckets 4,096 and 131,072, a 4,096-block override, one visible Neuron
device, an isolated cache, and a 20-minute bound. The prefill graph extracted
successfully. Dynamic-AP DGE counts in the live compiler log were 2,338/2,339
for `sg02` and 1,969/1,971 for `sg00`; the other observed subgraph was zero.
This is over 99% below the prior 262,408/327,808-scale explosion. The largest
observed reader counts were 35,249 and 98,139, so the old
`Max Readers: 262,145` dependency node was absent. Compiler peak RSS reached
15.4 GiB, below the 22 GiB target.

This diagnostic did not finish within 20 minutes. It was still making progress
through dependency and anti-dependency passes on the remaining outer graph,
including a roughly 650,000-instruction subgraph. The process tree was stopped
cleanly at the bound. Therefore the packed-compressor explosion is structurally
removed, but the under-600-second whole-model acceptance target is not met.
Per the test ladder, no TP2/EP2 cold compile, official-token/logit check, or
warm-cache launch was attempted.

## Why the previous recommendation cannot work

The earlier handoff's first recommendation was:

> Put a small fixed query tile, such as Q2 or Q4 per LNC, on the program grid.

That is not available, because **in NKI v2 (0.6.0) there is no user-settable
SPMD launch grid separate from the LNC degree.** The bracket syntax is defined
as LNC, not as a grid — `nki/framework/kernel.py`:

```python
lnc: int = 1
"""LNC degree."""

def __getitem__(self, lnc):
    """Allows users to set LNC at the callsite using bracket syntax."""
```

and `torch_neuronx/nki_hop.py` forwards it as exactly that:

```python
lnc = grid[0] if grid else 1
dconfig = get_config(kernel[lnc], **meta_args)
```

`nl.program_id` / `nl.num_programs` / `nl.num_programs_axes` only *read* the
grid; nothing in the package *sets* one except that LNC bracket, and the
`Kernel` dataclass carries no grid field at all. So the launch grid **is** the
LNC dimension, and `nl.num_programs(0)` is 1 or 2 by construction.

Consequently `queries_per_program` is at best `Q/2`, and keeping the kernel body
small *forces* host-side slicing. The kernels already read `nl.program_id(0)` /
`nl.num_programs(0)` correctly; there is simply no wider grid to tile onto.

A caution on how not to test this: writing `_wrapped_paged_shared_latent_mla[128]`
does **not** request a 128-program grid — it requests LNC=128, which is a
meaningless configuration and fails with
`NkiValidationError('NKI only supports LNC 1 or 2, but got 128')`. That error is
a symptom of misusing the bracket, not independent evidence of a grid cap; the
evidence is the source above.

## The actual fix: a runtime query loop

The query loop must execute at runtime inside a single launch, so that the body
is emitted once and the outer graph sees one node per layer. `nl.affine_range`
and `nl.static_range` both unroll; the primitive that does not is
`nki.language.fori_loop`:

> Structured for loop with dynamic bounds. [...] The body receives the current
> iteration value `i` as a VirtualRegister, passed by value. Read it inside the
> body (for example, as a `scalar_offset` into a tensor).

Progress made and verified in this investigation:

1. Replacing the `nl.affine_range` query loop with
   `nl.fori_loop(base, base + queries_per_program, body)` **is accepted** —
   folding the program base into the loop bounds avoids `int + VirtualRegister`
   arithmetic, which is rejected:
   `error: 'add' expected ... got (int, object)`.
2. `assert q_count in <frozenset>` must become a tuple; the NKI tracer rejects
   frozensets: `error: 'in' expected (any, list) or (any, tuple) ... got (int, NoneType)`.
3. The remaining blocker is addressing. Inside the loop `q_idx` is a register,
   so `query[q_idx, 0, :, ...]` no longer narrows the leading dimension:
   `dma_transpose dst.shape must match transposed src.shape. With axes=(2,1,0),
   src.shape=(512, 64, 128) transposes to (128, 64, 512), but got dst.shape=(128, 64)`.

The mechanism to finish it is already in the NKI API — `.ap()` accepts runtime
offsets:

```python
def ap(self, pattern, offset: int | Reg = 0,
       register_offsets: tuple[Reg | None, ...] | None = None,
       vector_offset=None, indirect_dim=0, dtype=None)
```

So `_stream_paged_query` needs its four `q_idx`-dependent accesses rewritten to
register-offset form (`register_offsets=(q_idx, None, ...)` is the per-dimension
mechanism, and is preferable to computing `q_idx * stride`, since register
arithmetic in the tracer is restricted):

- `query[q_idx, 0, :, d*128:(d+1)*128]` (the `dma_transpose` source)
- `slots[q_idx, start:start+width]`
- the attention-mask row
- the `result[q_idx]` write

This rewrite was **not** landed here: it changes the numerics path, so it needs
validation against the BF16 oracle (sink normalization, padding, null blocks,
partial final history tiles) before it can be trusted.

## Interim mitigation

`_PREFILL_QUERY_TILE` is now settable via `VLLM_NEURON_DSV4_MLA_QUERY_TILE`
(default unchanged at 8). Raising it trades outer-graph call sites for kernel
body size along the curve measured above, with no correctness change — the
kernel already partitions queries by `program_id * queries_per_program`.

## What is not the cause

- **Model or checkpoint size.** Qwen3-8B BF16 compiles on the same host at
  TP2/LNC2, Q512, 131072 cache in 308.77 s and 11.15 GiB peak RSS. The tiny
  DeepSeek-V4 model is a fraction of its weight size.
- **Q512, TP2/LNC2, BF16, or 131K capacity** as such — ruled out by the same
  Qwen3-8B run.
- **The NKI kernels' internal structure**, as of the streaming rewrite. Every
  standalone probe passes well inside budget; see the table above.
- **The CSA indexer and routed MoE.** 32 and 3 call sites respectively, and both
  compile standalone in minutes.

## Reproducing the analysis

The graph dumps are MLIR bytecode; `xla.HloModuleProto` will not parse them.

```python
from torch_mlir.ir import Context, Module
ctx = Context(); ctx.allow_unregistered_dialects = True
m = Module.parse(open("/tmp/neuron_hlo_845579_11.pb", "rb").read(), ctx)
text = m.operation.get_asm(large_elements_limit=8, enable_debug_info=False)
```

Then count call sites by decoding each base64 `backend_config` and reading its
`"func_name"` field.

Note that running the tools from a worktree needs the main checkout on
`PYTHONPATH` **after** the worktree, or vllm finds no `vllm.platform_plugins`
entry point and dies with `RuntimeError: Device string must not be empty`:

```bash
PYTHONPATH=<worktree>:/home/ssm-user/vllm-neuron
```
