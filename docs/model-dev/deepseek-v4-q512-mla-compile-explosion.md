# DeepSeek-V4 Q512 MLA compiler graph explosion

Status: open as of 2026-08-26. This is a handoff for the host-memory OOM seen
while cold-compiling the three-layer official-weight DeepSeek-V4 tiny model at
TP2/EP2, LNC2, BF16, Q512 prefill, and 131072-token cache capacity.

For the exact host, Python/Neuron stacks, paths, cache conventions, and test
workflow, see
[DeepSeek-V4 development and test environment handoff](deepseek-v4-development-environment-handoff.md).

## Summary

The failure is a finite compiler graph explosion in the paged MLA kernels. It
is not an infinite model loop, it is not proportional to checkpoint weight
size, and the capacity-sized CSA indexer loop is no longer the primary cause.

For Q512 with LNC2, each MLA program computes
`queries_per_program = 512 // 2 = 256` and uses `nl.affine_range` over that
value. NKI expands the query work into hundreds of instruction bodies in each
program. The paged MLA implementation also materializes both of the large
intermediates that the streaming redesign was intended to remove:

- a BF16 latent tensor shaped `[Q, history, 512]`;
- a BF16 probability tensor shaped `[Q, heads, padded_history]`.

The three tiny-model layers have histories 128 (sliding), 640 (CSA), and 1152
(HCA). Combining their expanded programs in the full graph produces roughly
350000 instructions per LNC subgraph, about 7.3 million dependency edges, and
more than one million memory intervals. Two TP-rank compiler processes run at
the same time and exhaust the 124 GiB host.

The observed cold run lasted 1:29:55, reached 125670464 KiB (119.85 GiB) peak
RSS, and ended with `neuronx-cc` exit 70 / forced kill. No NEFF or first token
was produced for the full model.

## Important error-attribution correction

The Python error is reported while `_build_decode_synthetic_inputs()` moves
metadata to the device for the first `(1, 4096)` decode target. That does not
mean the decode graph caused the OOM. Neuron compilation is asynchronous: the
Q512 prefill graph had already been dispatched, and this later device operation
was the synchronization point that observed its failure.

The temporary NKI artifacts from the failed run were all emitted between
04:43:44 and 04:44:17, while Q512 prefill was being captured. No decode MLA
artifact was emitted after decode capture began. Treat the earlier description
"the first decode target took more than 19 minutes" as incorrect.

## Reproduction

The failing run used branch `deepseek-v4-tp-ep` at merge commit `b2763b2`, the
real-weight slice `/home/ssm-user/ds-v4-tiny-real`, and an isolated empty local
cache:

```bash
mkdir -p /tmp/dsv4-q512-repro/cache

/usr/bin/time -v env \
  PATH=/home/ssm-user/.venv-torch-neuronx-dev/bin:/opt/aws/neuron/bin:/usr/bin:/bin \
  PYTHONPATH=/home/ssm-user/vllm-neuron \
  VLLM_CACHE_ROOT=/tmp/dsv4-q512-repro/cache \
  NEURON_VISIBLE_DEVICES=0,1 \
  VLLM_NEURON_ENABLE_DEEPSEEK_V4=1 \
  VLLM_NEURON_VALIDATE_CACHE_METADATA=1 \
  NEURON_SKIP_EFA_AFFINITY=1 \
  /home/ssm-user/.venv-torch-neuronx-dev/bin/python \
  tools/deepseek_v4/generate_tiny.py \
  /home/ssm-user/ds-v4-tiny-real \
  --output /tmp/dsv4-q512-repro/result.json \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --ep-degree 2 \
  --load-format auto \
  --max-model-len 131072 \
  --max-num-batched-tokens 512 \
  --prefill-segment-buckets 512 \
  --decode-context-buckets 4096 \
  --num-gpu-blocks-override 4096 \
  --prompt-length 512 \
  --max-tokens 1
```

Do not repeat this full cold compile merely to confirm the known failure on a
124 GiB host. Start with the component probes and structural checks below.

The preserved logs on the original host are:

```text
/tmp/dsv4-full-q512-tp2-lnc2-merged-20260826/cold.log
/tmp/dsv4-full-q512-tp2-lnc2-merged-20260826/cold.time
```

## Source-level defect

The relevant code is `vllm_neuron/model/deepseek_v4/nki_mla.py`.

### Query bodies are statically expanded

The same query-loop shape appears in all three phases:

```python
queries_per_program = q_count if q_count == 1 else q_count // n_programs
for local_q_idx in nl.affine_range(queries_per_program):
    ...
```

Current locations are approximately:

- QK/softmax: lines 45-49;
- paged latent gather: lines 262-266;
- PV: lines 185-189;
- sliding gather: lines 369-373.

At Q512/LNC2 this creates 256 query iterations per program in each phase. The
history loops nested inside those iterations add factors for four 128-wide
latent-dimension tiles and up to twelve 128-key PV tiles.

`nl.affine_range` here does not produce the desired one-body runtime query loop.
Changing only the outer scheduler bucket to 1024 makes the per-program
expansion 512. The existing 1024-token microchunk dispatcher prevents native
Q2048/Q4096 specializations, but it does not solve the Q-axis expansion inside
the Q512/Q1024 binaries.

### Query-by-history tensors are still materialized

`_materialize_paged_latent_stage()` allocates:

```python
latent = nl.ndarray(
    (q_count, history, latent_dim),
    dtype=sliding_cache.dtype,
    buffer=nl.shared_hbm,
)
```

`_manual_qk_softmax_stage()` allocates:

```python
probs_out = nl.ndarray(
    (q_count, heads, padded_history),
    dtype=nl.bfloat16,
    buffer=nl.shared_hbm,
)
```

`_paged_shared_latent_mla_kernel()` gathers the complete latent tensor, invokes
the complete QK/softmax stage, then invokes a separate PV stage. It is therefore
a materialized three-stage attention kernel despite being bounded and opaque to
the outer FX graph.

For Q512, the explicit large buffers are approximately:

| Layer geometry | Latent buffer | Probability buffer | Total |
| --- | ---: | ---: | ---: |
| Sliding: history 128, padded 512 | 64 MiB | 32 MiB | 96 MiB |
| CSA: history 640, padded 1024 | 320 MiB | 64 MiB | 384 MiB |
| HCA: history 1152, padded 1536 | 576 MiB | 96 MiB | 672 MiB |
| Combined | 960 MiB | 192 MiB | 1152 MiB |

These are device-visible intermediates, not the 119.85 GiB host allocation by
themselves. The compiler's instruction, dependency, and memory-interval data
structures amplify the expanded programs into the host OOM.

The materialized paged implementation originated in commit `bf12ce5` ("Gather
paged MLA history inside NKI"). Commit `e4267eb` added TP/EP compatibility and
the Q512/Q1024 direct specializations, but did not replace the materialized MLA
body with streaming attention.

## Artifact evidence

The failed Q512 run left compressed NKI COLZ artifacts under `/tmp/nki_*`.
COLZ has a 16-byte header followed by a zstd-compressed JSON payload. Comparing
Q1 and Q512 artifacts gives:

| Component | Q1 COLZ | Q512 COLZ | Q512 tensor-allocation records |
| --- | ---: | ---: | ---: |
| Sliding MLA | 5298 bytes | 910218 bytes | 19474 |
| CSA MLA | 6868 bytes | 1487197 bytes | 27674 |
| HCA MLA | 5744 bytes | 2037539 bytes | 35866 |
| CSA indexer | 12909 bytes | 181059 bytes | 1906 |

Older paged-MLA probe artifacts also scale almost exactly linearly with the
query bucket: about 1.48 MB, 2.91 MB, 5.78 MB, and 11.47 MB for successive
Q512/Q1024/Q2048/Q4096 geometries. That is direct evidence of finite static
query expansion rather than a non-terminating compiler process.

On the original host, an artifact can be inspected without writing a decoded
copy:

```bash
artifact=/tmp/nki_bxfhp8hg/vllm_neuron.model.deepseek_v4.nki_mla._paged_shared_latent_mla_kernel_lnc0_cec9b6fc87172aa89.colz

dd if="$artifact" bs=1 skip=16 status=none \
  | zstd -dc 2>/dev/null \
  | jq '{tensor_allocation_records: ([.. | objects | select(has("tensor_id"))] | length)}'
```

The JSON stores both LNC functions, so counts above include both programs.

## What is not the primary defect

### CSA indexer

`nki_indexer.py` now slices Q512 into Q16 calls, assigns eight queries per LNC,
and scans history with:

```python
nl.fori_loop(0, page_count_reg, scan_page)
```

Its standalone Q512/131072-entry compile succeeds. Measured cold time was
210.15 seconds, peak process RSS was 1162564 KiB, and changing runtime visible
history did not create a different capacity-sized instruction body. The
remaining 181 KB Q16 artifact is much smaller than the MLA artifacts.

### Routed MoE

The standalone Q512 `moe_cte` probe succeeds in 57.23 seconds with 1615880 KiB
peak process RSS. It remains worth retaining in a full-graph bisection, but the
component does not exhibit the MLA query-bucket scaling or standalone memory
failure.

### Model and cache size

Qwen3-8B BF16 compiled successfully on the same machine with TP2/LNC2, Q512,
and 131072 cache capacity. Its cold initialization took 308.77 seconds, total
process time was 5:33, peak RSS was 11.15 GiB, and first-token latency was 6.32
seconds. A same-cache run submitted no `neuronx-cc` process and produced its
first token in 0.145 seconds. This rules out Q512, TP2/LNC2, BF16 weight size,
or 131K cache capacity as sufficient causes.

## Recommended implementation

Replace the paged MLA family with a genuinely fused streaming kernel:

1. Put a small fixed query tile, such as Q2 or Q4 per LNC, on the program grid.
   Do not make one program contain `Q / LNC` affine query bodies.
2. Gather only one 128-key latent tile from the sliding/compressed paged caches
   into SBUF.
3. Compute QK for that tile and immediately update FP32 online-softmax state:
   `(running_max, running_sum, running_output)`.
4. Initialize that state with the per-head attention-sink contribution so sink
   semantics remain exact.
5. Accumulate PV immediately and discard the key tile and tile probabilities.
6. Emit only `[Q, 1, heads, 512]`.
7. Use fixed bounded histories: sliding 128, CSA 512+128, and HCA 1024+128.
8. Compile one fixed query-tile binary and launch it across Q512/Q1024. Continue
   using four Q1024 scheduler microchunks for Q4096.

Do not merely replace `nl.affine_range` with another loop while retaining the
global latent and probability tensors. Both the instruction-axis expansion and
the query-by-history allocations must be removed.

## Suggested validation sequence

Use fresh isolated caches for cold measurements.

1. Add structural tests that reject any NKI allocation shaped like
   `[Q, K, 512]` or `[Q, 64, K]` for Q greater than one.
2. Compile Q1 and the fixed query-tile MLA binaries for histories 128, 640, and
   1152. Artifact size and allocation-record count should remain effectively
   independent of the scheduler Q bucket.
3. Run the existing standalone probes:

   ```bash
   python tools/deepseek_v4/compile_runtime_mla.py --query 512 --compressed 0
   python tools/deepseek_v4/compile_runtime_mla.py --query 512 --compressed 512
   python tools/deepseek_v4/compile_runtime_mla.py --query 512 --compressed 1024
   python tools/deepseek_v4/compile_runtime_indexer.py \
     --query 512 --block-columns 1024 --logical-slots 128 --visible 131072
   python tools/deepseek_v4/compile_runtime_moe.py --query 512
   ```

4. Verify NKI simulator and device outputs against the portable BF16 oracle,
   including sink normalization, padding, null blocks, and partial final
   history tiles.
5. Run one-rank full-graph component bisections before TP2 to avoid two broken
   compiler processes consuming host RAM simultaneously.
6. Repeat the full Q512 TP2/EP2 cold compile. The acceptance target is less
   than 600 seconds and less than 22 GiB peak compiler RSS.
7. Relaunch with the same cache and verify that no model or NKI specialization
   is submitted for compilation.

## Profiling infrastructure gap

`tools/deepseek_v4/benchmark_prefill_components.py` sets
`VLLM_NEURON_DEEPSEEK_V4_DIAGNOSTIC_IDENTITY`, but no model code currently
reads that environment variable. Therefore its advertised full-model
"replace one component with an opaque identity" mode is not implemented.
Before relying on that harness, add explicit, graph-stable diagnostic identity
boundaries for sliding MLA, HCA MLA, CSA indexer, CSA MLA, MoE mapping, and
`moe_cte`, with the default path unchanged.

## Completion criteria

The defect is resolved only when all of the following hold:

- no `[Q, K, 512]` latent or `[Q, 64, K]` probability allocation exists;
- query-loop instruction bodies are fixed-size and independent of Q512/Q1024;
- standalone MLA specializations compile in under 60 seconds where feasible;
- full tiny-model Q512 TP2/LNC2 compilation finishes below 600 seconds and
  22 GiB peak compiler RSS;
- warm Q512 runtime is no slower than the equivalent fixed query-tile launches;
- official selected tokens remain `[2030, 32974, 63376, 76010]`, with selected
  logit absolute error at or below 0.125;
- a warm relaunch performs no recompilation.
