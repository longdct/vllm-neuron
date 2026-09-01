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
