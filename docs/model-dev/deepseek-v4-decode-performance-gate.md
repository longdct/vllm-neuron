# DeepSeek-V4 decode-performance gate

Status: implementation and empirical-validation update from 2026-09-01. The
production performance gate is **not yet passed**. Read this with
`deepseek-v4-deployment-blockers.md` (which tracks the correctness and scale
blockers this gate sits behind),
`deepseek-v4-development-environment-handoff.md`, and
`deepseek-v4-q512-mla-compile-explosion.md` before any device compile.

## Performance measurement correction

The previously reported `0.600 tok/s` value is not a steady-decode rate. The
retained depth-43 TP8 report computed it as 4 output tokens divided by a
6.663-second end-to-end generation call (`0.600314 tok/s`), while its separate
first-token probe took 20.111 seconds. Neither measurement recorded streaming
token boundaries, so the result mixes request phases and cannot identify
decoder-layer latency. It must not be used as the deployment baseline.

Decode reports now separate these phases:

1. **TTFT:** request submission through the first generated token, including
   prefill.
2. **First decode interval:** first-token timestamp through second-token
   timestamp. This is the prefill-to-decode transition penalty.
3. **Steady ITL:** intervals beginning with the second-to-third generated-token
   transition. Median and p95 are computed only from these intervals.
4. **Request throughput:** output tokens divided by end-to-end request latency.
5. **Aggregate throughput:** all output tokens in a concurrent workload divided
   by the workload makespan.

`tools/deepseek_v4/benchmark_decode.py` records the raw per-token timestamps and
all five derived measurements as JSON. Its defaults use two discarded warmups,
three measured repetitions, generic on-device sampling, and async scheduling.
CPU sampling remains available with `--sampling-backend cpu` for compatibility
and debugging.

The fixed workloads are:

| Workload | Prompt tokens | Output tokens | Purpose |
| --- | ---: | ---: | --- |
| short | 508 | 4 | Q512 TTFT and transition probe |
| sustained | 384 | 128 | batch-1 steady decode |
| batch8 | 384, 320, 256, 192, 128, 96, 64, 32 | 128 each | ragged aggregate throughput |

Run all three with `--query-bucket 512`. Repeat sustained and batch8 with
`--query-bucket 8192`; the tool omits the Q512-only short probe by default at
that geometry. Use an isolated `VLLM_CACHE_ROOT` for each cold configuration
and the identical cache for its warm relaunch.

## Decoder collective count

The ordinary TP path now combines the routed-expert local output with the
shared-expert local TP partial and performs one TP all-reduce over their sum.
Together with the attention output reduction, this leaves two TP-wide
reductions per decoder layer instead of three. At the production depth of 43
layers, that removes 43 collectives per generated token.

Cross-DP expert parallelism is intentionally unchanged. Its routed output uses
the wide EP communication domain while its shared expert uses TP, so those
reductions cannot be fused.

## Sampling path

The performance configuration is one generic device graph:

```json
{"all_greedy": false, "max_top_k": 256}
```

It accepts greedy, top-k, top-p, temperature, and mixed per-request parameters.
The LM head keeps vocabulary logits sharded and sampling uses distributed
argmax/top-k collectives. A full-vocabulary all-gather is emitted only when
logprobs or debug-logit capture is explicitly enabled. Padded vocabulary rows
are masked before sampling, and greedy rows use distributed argmax so tied
logits retain CPU argmax semantics.

## Acceptance gate

Compare isolated-cache baseline and optimized JSON reports at 43 layers and 256
experts. The available-host gate is TP32/EP2; TP64/EP4 remains the final gate
when all 64 logical cores are explicitly free.

- Batch-1 median steady ITL improves by at least 20%.
- Batch-8 aggregate output throughput improves by at least 20%.
- Median TTFT regresses by no more than 5%.
- Greedy token output has no regression.
- Fixed deterministic top-k, top-p, and temperature outputs remain valid.
- Per-rank HBM and compiled-artifact totals grow by no more than 10%.

The four comparison points are current model with CPU sampling, current model
with device sampling, fused MoE with CPU sampling, and the final fused/device
configuration. Shape-accurate BF16 dummy weights are the production-geometry
performance vehicle until the incomplete official checkpoint is available.

## Empirical validation on 2026-09-01

The available-host run used logical cores 12--19 and a shape-accurate BF16
dummy checkpoint with 3 decoder layers, 256 experts, TP8/EP4, Q512, and generic
device sampling. The production service on cores 32--63 and the unrelated TP8
job on cores 4--11 were not interrupted. These reduced-geometry results are
diagnostic evidence only; they are not the TP32/TP64 acceptance run.

The normal CSA selection graph compiled and dispatched, but did not make
forward progress at the next device synchronization. Core dumps showed a
successful eighth all-reduce followed by stalled post-collective instructions;
the runtime explicitly did not classify the failure as a collective mismatch.
Replacing only lightning-indexer scoring/selection with the existing fixed-CSA
bisection path made the same end-to-end configuration execute. This localizes
the normal-graph blocker to the lightning-indexer scoring/selection region, but
the bypass changes model semantics and must not be used for correctness or
production acceptance.

With fixed CSA, the fused/device sustained workload completed two discarded
warmups and three measured repetitions. Its retained JSON is
`/tmp/dsv4-benchmark-fixed-csa-ZA8wi1GT/result-sustained-warm.json`:

| Metric | Median | p95 |
| --- | ---: | ---: |
| TTFT | 167.902 ms | 168.072 ms |
| First decode interval | 68.677 ms | 69.362 ms |
| Steady ITL | 69.199 ms | 72.356 ms |
| Request latency | 8.946 s | 8.984 s |
| Aggregate output throughput | 14.307 tok/s | 14.337 tok/s |

An isolated-cache legacy overlay restored the second MoE reduction while
leaving every other source path unchanged. Its retained JSON is
`/tmp/dsv4-benchmark-legacy-moe-sIKEJjST/result-sustained.json`. The direct A/B
comparison was:

| Metric | Legacy reductions | Fused reduction | Change |
| --- | ---: | ---: | ---: |
| Median steady ITL | 69.094 ms | 69.199 ms | -0.15% |
| Aggregate output throughput | 14.300 tok/s | 14.307 tok/s | +0.05% |
| Median TTFT | 162.732 ms | 167.902 ms | -3.18% |
| First decode interval | 69.589 ms | 68.677 ms | +1.31% |

Positive change denotes improvement. At three layers these differences are
measurement noise, so the fused reduction has **not** empirically demonstrated
the required 20% batch-1 gain. All three 128-token repetitions were identical
within each configuration and across the A/B pair. Per-rank HBM was unchanged
at 6,448,742,400 bytes. Fused compiled artifacts were 190,671,349 bytes versus
185,530,842 bytes for legacy, a 2.77% increase and within the 10% limit.

The ragged batch-8 gate could not produce a valid measurement. A fused/device
graph for sequence buckets `[1, 8]` extracted and warmed, but its first real
ragged request batch did not return. That first attempt also exposed an
insufficient 512-block diagnostic override, which admitted only six
maximum-length requests, so no number from it is reportable. A separate CPU
sampling control used 768 blocks (6,553 token capacity, 12.8 maximum-length
requests), compiled both graphs, then aborted during prefill warmup with
`encd_mesh_add_wr_barrier: event->evt_type == EVT_SYNC`. Its retained log is
`/tmp/dsv4-benchmark-cpu-batch8-Ycs4DtmT/run.time`. Batch-8 therefore remains a
general batched-runtime blocker; the evidence does not isolate it to device
sampling.

Strict cold, trace-cache-disabled component compiles do confirm that the Q512
MLA structural explosion is removed. Q1-to-Q512 compile time stayed nearly
flat for sliding attention (2.385 to 2.466 seconds), CSA (2.584 to 2.743
seconds), and HCA (2.658 to 2.842 seconds). The corresponding compiler RSS did
not grow with Q512. Unit and NKI-simulator structural coverage completed with
21 passed and 36 skipped tests before the end-to-end runs.

The next acceptance attempt must first fix both runtime blockers, then rerun
the four-point comparison at TP32/EP2. TP64/EP4 remains unavailable while the
production service owns cores 32--63. Do not extrapolate the three-layer TP8
result to 43 layers or claim the 20% gate from collective counts alone.

Do not run the known broken full Q512 cold compile described in the MLA defect
handoff until its structural checks confirm that static query expansion and
query-by-history materializations are absent.

## RETRACTED: the depth-8 A/B below did not test what it claimed

The fused-vs-legacy comparison recorded in the next section is **invalid** and
its conclusion must not be used.

Its two arms selected the reduction with
`VLLM_NEURON_DSV4_FUSED_MOE_REDUCTION`, which is implemented only on this
branch. But the arms did not set `PYTHONPATH`, and the editable install points
at the primary checkout -- so vLLM's worker processes imported the model code
from `/home/ubuntu/vllm-neuron` (then at `6c9c783`), which contains **zero**
occurrences of that variable. The flag was never read. Both arms ran identical
model code.

The 0.52% difference between them was therefore run-to-run and cold-compile
variation between two independent builds, not the fused reduction, and the
"regression grows with depth" reading (-0.15% at depth 3, -0.52% at depth 8)
has no support. The differing NEFF byte totals, offered at the time as proof
the flag reached the graph, are explained the same way.

What survives:

  - The measured ITLs are still real measurements *of the primary checkout's
    model* at those depths: 69.199 ms median steady ITL at depth 3 and
    131.5-132.2 ms at depth 8, TP8/EP4, Q512, 256 experts, dummy BF16 weights,
    fixed-CSA. They remain usable as BF16 baselines provided the code under
    comparison is stated.
  - The normal-CSA stall reproducing at depth 8 exactly as at depth 3. That is
    a directly observed failure, independent of which reduction ran.

Anything running a DeepSeek-V4 benchmark from a git worktree must export
`PYTHONPATH=<worktree>` or it silently measures a different tree. The same
mistake surfaced later as the factory rejecting `quantization="fp8"` that the
worktree supports.

## Depth-8 A/B on 2026-09-02: the fused reduction is a small regression

The 2026-09-01 comparison above could not separate the fused reduction from
noise at 3 layers, where it removes only 3 collectives per generated token.
This run repeats it at depth 8, where it removes 8, and adds the controls the
earlier pair lacked: both arms come from one build at `a192310`, selected by
`VLLM_NEURON_DSV4_FUSED_MOE_REDUCTION` rather than a source overlay, and each
report records the flag that produced it.

Shape-accurate BF16 dummy checkpoint, 8 layers, 256 experts, TP8/EP4, Q512,
`max_model_len` 512, sustained workload (384-token prompt, 128 output tokens),
device sampling, 2 discarded warmups and 3 measured repetitions, logical cores
12--19, isolated compile cache per arm. Reports are
`/tmp/dsv4-moe-ab-depth8/{fused,legacy}/result.json`.

| Metric | Legacy (2 reductions) | Fused (1) | Change |
| --- | ---: | ---: | ---: |
| Median steady ITL | 131.534 ms | 132.215 ms | -0.52% |
| p95 steady ITL | 133.381 ms | 134.677 ms | -0.97% |
| Median TTFT | 577.016 ms | 577.707 ms | -0.12% |
| First decode interval | 131.357 ms | 132.685 ms | -1.01% |
| Aggregate output throughput | 7.403 tok/s | 7.359 tok/s | -0.59% |

Positive change denotes improvement, so **every metric moved the wrong way**.
The result is small but it is not noise: per-repetition median ITL was 131.517
/ 131.538 / 131.561 ms for legacy against 131.955 / 132.293 / 132.373 ms for
fused, so the two arms do not overlap and the within-arm spread (0.03% and
0.32%) is below the 0.52% gap. Per-rank HBM was identical at 15,804,137,472
bytes, and async scheduling behaved identically (635 async steps, 5 sync
fallbacks on both), so neither is a confound. The two NEFF sets do differ
(281,804,931 vs 281,802,386 bytes), which confirms the flag reached the
compiled graph rather than being silently ignored.

Read with the 2026-09-01 depth-3 pair, the regression grows roughly with depth
-- -0.15% at 3 layers, -0.52% at 8 -- which is the opposite of the trend the
fusion was adopted for. Removing 8 all-reduces per token made decode slower,
so **the TP collective count is not the decode bottleneck at this geometry**,
and the 20% batch-1 gate cannot be reached by removing more of them.

A plausible mechanism, untested: the legacy path issues two independent
all-reduces that the runtime can overlap with expert compute, while the fused
path serialises routed -> shared -> add -> one reduce, putting the whole sum on
the critical path. If that is the cause, the fix is to overlap the single
reduction rather than to restore the second one. Attribute this before either
keeping or reverting the fusion; the flag now makes that a one-variable
experiment.

Two caveats bound the result. Both arms required
`VLLM_NEURON_DSV4_FIXED_CSA_SELECTION=1`, so neither is a correctness or
acceptance measurement. And the normal CSA selection graph still does not make
forward progress: at depth 8 it compiled, dispatched, and then failed prefill
warmup at bucket 512 with `NRT EXECUTION FAILED ... Operation timed out`,
exactly as at depth 3. That blocker is therefore **not depth-dependent**, which
removes model depth as a candidate cause and keeps the fault localized to the
lightning-indexer scoring/selection region.

For planning only, and not a claim about the real model: median steady ITL was
69.199 ms at depth 3 and 132.215 ms at depth 8 on the same cores and geometry,
which fits `31.4 ms + 12.6 ms x depth`. Extrapolated to 43 layers that is
roughly 573 ms per token at TP8. TP32/TP64 is the geometry that has to close
that gap, and it remains unavailable while the production service holds cores
32--63.


## FP8 on TRN2: the constraints that actually bind (2026-09-02)

Recorded from getting the FP8 path onto device, because none of these are
discoverable from the error text they produce.

### `shard_on_i` needs I_TP >= 256, which forces EP at high TP

FP8 experts must use `shard_on_i` (`shard_on_block` ignores the scales -- see
the milestone 0 notes). At TP16 with no expert parallelism,
`expert_tp_degree = 16` and `I_TP = 2048/16 = 128`, and decode graph extraction
fails inside the kernel:

```
bwmm_shard_on_I.py:1483 [INTERNAL_ERROR] [NCC_IBIR243]
Access pattern out of bounds. Pattern: [[256,128],[1,256]]
```

The 256 is the FP8 MoE block; the 128 is `I_TP`. TP8 gives `I_TP = 256` and
works. The fix at TP16 is `--enable-expert-parallel --ep-degree 2`, which makes
`expert_tp_degree = world/ep = 8` and restores `I_TP = 256` at *identical*
per-rank memory -- each rank then holds 128 experts sharded 8 ways instead of
256 sharded 16 ways. **Above TP8, FP8 requires EP.**

### The paged-MLA query bucket set is sparse

`(1, 2, 4, 8, 64, 512, 1024, 2048, 4096, 8192)` -- there is no 16, 32, 128 or
256. A `max_model_len` of 256 fails in the MLA guard, which until now reported
it as a head-count violation. Fixed to name the bucket.

### KV budget is computed after graph allocation

At depth 43 / TP16 the parameter footprint is 18.695 GiB per rank and HBM sits
at ~21.5 GiB once graphs are allocated, so the default
`gpu_memory_utilization=0.9` leaves ~0.11 GiB for KV and the engine refuses a
512-token context. Raising GMU alone does not rescue it; the practical lever at
production depth is a shorter context.

### Depth 43 does not exist in BF16 here

18.695 GiB/rank of FP8 weights becomes ~36 GiB in BF16, against a 24 GiB core.
No available geometry runs depth 43 in BF16 -- cores 32-63 would be needed for
TP32 and are not free. This is the clearest statement of what FP8 buys: not
speed, but a configuration that otherwise cannot be instantiated.


## FP8 results on device (2026-09-02)

### Depth 43 runs, and only in FP8

43 layers, 256 experts, TP16/EP2, Q64, dummy FP8 weights, logical cores
0-3 + 20-31. Generated 4 tokens (`[6690, 824, 4628, 5697]`), 2219 s cold
init, 11.84 s to first token, 1.609 tok/s end-to-end at that geometry.

Per-rank parameter footprint **18.653 GiB**, 19.02 GiB HBM used, 4.98 GiB
free of 24. The same weights in BF16 are ~36 GiB/rank, so **no available
geometry runs depth 43 in BF16** -- TP32 would need cores 32-63. This is the
concrete thing FP8 buys.

The run needed `--enable-expert-parallel --ep-degree 2`: see the `shard_on_i`
I_TP constraint above.

### Depth 3, BF16 vs FP8

TP8/EP4, Q512, 256 experts, sustained workload (384-token prompt, 128 output
tokens), CPU sampling on **both** arms, 2 warmups + 3 repetitions, cores 12-19,
same build, isolated compile cache per arm.

| Metric | BF16 | FP8 | Change |
| --- | ---: | ---: | ---: |
| Median steady ITL | 54.405 ms | 44.597 ms | **+18.03%** |
| p95 steady ITL | 55.694 ms | 53.723 ms | +3.54% |
| Median TTFT | 132.415 ms | 117.000 ms | +11.64% |
| Aggregate throughput | 18.227 tok/s | 21.721 tok/s | **+19.17%** |
| Per-rank HBM | 6.006 GiB | 3.758 GiB | **-37.43%** |

### Depth 8, BF16 vs FP8

Same harness and geometry, depth-8 checkpoint.

| Metric | BF16 | FP8 | Change |
| --- | ---: | ---: | ---: |
| Median steady ITL | 130.436 ms | 106.935 ms | **+18.02%** |
| Median TTFT | 568.994 ms | 529.790 ms | +6.89% |
| Aggregate throughput | 7.470 tok/s | 9.064 tok/s | **+21.34%** |
| Per-rank HBM | 14.719 GiB | 8.725 GiB | **-40.72%** |

Positive is better. **FP8 is faster, not merely smaller** -- which contradicts
the earlier expectation, itself resting on a retracted A/B, that batch-1 decode
is latency-bound and would not benefit.

The ITL gain is 18.03% at depth 3 and 18.02% at depth 8 -- essentially
identical, which is what a per-layer effect looks like and is not what noise
looks like. Per-repetition medians confirm it: within-arm spread is 0.91% /
4.52% / 0.09% / 0.26% against an 18% gap, and no arm's repetitions come near
the other arm's.

**What the delta actually contains.** These arms differ in more than dtype: FP8
routes the MoE to `shard_on_i` with a 256-token block, BF16 uses
`shard_on_block` with 128. Kernel, block size and weight dtype all move
together, so the 18% is the combined effect of the FP8 path, not an
attribution to quantization alone. Separating them would need `shard_on_i` at
BF16, which no current call site builds.

Both arms use **CPU sampling**, because device sampling on this branch emits
out-of-range token IDs and kills the request in `_validate_token_ids` -- see
below. That makes these numbers internally comparable but not comparable to
the earlier device-sampling figures (69.199 ms at depth 3), which were also
produced by a different tree.

### Resolved: device sampling emitted invalid token IDs

`--sampling-backend device` used to produce token IDs far outside
`[0, 129280)` -- observed `967439869` and `-1167771948` -- failing
`_validate_token_ids` on the next decode step. Fixed; see
`deepseek-v4-on-device-sampling.md` for the diagnosis.

Two things from it bear on the numbers here. **These runs used cores 12-19,
which is a broken TP8 placement**: devices 3 and 4 have no interconnect link,
and a narrow collective across them silently returns partial data. The model's
wide collectives are unaffected, and the host-sampled tokens were correct
throughout, so these timings stand -- but new measurements should use a
connected group (`16-23`, `20-27`, `16-31`).

And on-device sampling now costs nothing measurable against host sampling:
depth 3, TP8/EP4, BF16, median steady ITL 54.83 ms on-device against 54.32 ms
on host, with overlapping ranges. Depth 8 on-device is 7.63 tok/s, reproducing
the ~7 tok/s recorded while sampling was still emitting garbage -- the timing
was always sound, only the tokens were wrong.
