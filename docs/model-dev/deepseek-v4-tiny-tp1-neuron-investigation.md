# Tiny DeepSeek-V4 TP1 Neuron investigation handoff

Status as of 2026-08-22. This is the handoff for continuing the three-layer
deterministic DeepSeek-V4 investigation on another Neuron machine. It is not a
production-support claim for full DeepSeek-V4.

## Current result

The original PJRT crash and cache out-of-bounds failure mode no longer occurs.
All three target graphs compile, warm, execute, and shut down cleanly:

- prefill bucket 8;
- prefill bucket 64;
- decode (batch 1, sequence 64).

The device output is finite through all four generation steps, cache persistence
works, and all four greedy tokens now match the CPU oracle:

| Run | Generated tokens | Logits |
| --- | --- | --- |
| CPU eager oracle | [28, 26, 48, 27] | finite |
| Original Neuron graph | [53, 52, 12, 63] | finite, but stale-cache result |
| Functional cache before invalid-history fix | [0, 0, 0, 0] | non-finite at step 0 |
| Invalid-history fix with shared backing | [28, 0, 0, 0] | prefill finite; decode non-finite |
| Independent logical caches | [28, 26, 48, 45] | all four steps finite |
| Query partial-RoPE lowering fix | [28, 26, 48, 27] | all four steps finite |

For the latest maximum-length-64 run, maximum absolute CPU/Neuron logit
differences by step were 0.080078125, 0.0234375, 0.01727294921875, and
0.0087890625. All four argmax IDs match and every decode step is within the
0.025 acceptance tolerance. Prefill still has 15 of 64 logits outside tolerance,
with its maximum at logit index 18. Deployment therefore remains blocked on the
prefill numerical discrepancy even though compilation, runtime stability, and
greedy output are green.

## Scope and environment

The device runs used the repository virtual environment, the packaged runtime
under /opt/aws/neuron, TP1, BF16, maximum length 64, synchronous CPU sampling,
prefix caching disabled, and isolated compiler caches. The dependency family
observed during this campaign was Torch/Torch-XLA 2.11,
libtorch-neuronx-lite 2.11, neuronx-cc 2.27, and NKI 0.6. Run
tools/deepseek_v4/write_device_preflight.py on the destination machine; do not
assume its runtime, driver, compiler, or cache format matches this host.

This work targets only the deterministic three-layer checkpoint produced by
tools/deepseek_v4/build_tiny_checkpoint.py. Full Flash/Pro checkpoints, TP above
one, prefix caching, MTP/speculative decoding, disaggregated inference, and
production performance remain out of scope.

## Investigation chronology

### 1. Establish the CPU oracle

After restoring deterministic non-persistent buffers following meta-device
materialization, CPU eager generation produced [28, 26, 48, 27], with finite
logits. CPU graph capture matched CPU eager exactly, ruling out graph capture by
itself.

The restored buffers are RoPE frequencies, the identity K/V projection, and MoE
routing state. Hugging Face checkpoints do not contain these derived,
non-persistent buffers. The restoration is therefore required for real
checkpoint loads as well as the dummy tiny checkpoint.

### 2. Locate the first divergence

Tensor capture showed bit-exact embeddings, a layer-0 input-normalization
difference of only 2.38e-7, a layer-0 attention difference near 0.8025, and a
layer-0 MoE difference near 0.00225. That localized the original problem to
attention/cache handling.

A decisive counterfactual made scatter_paged_latent a no-op on CPU. It produced
exactly the original Neuron tokens [53, 52, 12, 63], and its first-step logits
differed from Neuron by only 0.001953. Neuron was gathering stale cache contents
after scatter.

### 3. Expose compiler functionalization

Returning an in-place-mutated cache was insufficient because the compiler
canonicalized the alias away. A clone followed by a view copy also failed. The
post-inplace FX graph reduced to this dataflow:

    updated_cache = clone(cache)
    modified_view = slice_scatter(view, updated_head)  # no users
    gather(updated_cache)                              # unchanged clone

The view mutation was dead. The scatter helper now constructs and returns a
fixed-shape updated cache. Attention threads that tensor through every token in
a prefill bucket, and the immediate gather consumes the returned tensor.
Captured FX confirmed token N+1 reads token N's functional cache result.

Fixed-shape forms tried during the investigation were index_add with counts, a
one-hot/matrix-multiply reduction, and a single-row compare/broadcast/where
specialization. Model call sites update exactly one row at a time, so the last
form is current. It avoids boolean indexing and data-dependent tensor shapes.
The multi-row helper path remains fixed-shape and deterministic.

### 4. Neutralize invalid compressed storage

The first functional-cache graphs compiled but returned NaNs. All-layer capture
showed prefill layers 0 and 1 were finite while layer 2 was non-finite beginning
at position 0, before the first ratio-4 compressed page was logically valid.

_compressed_history propagated a validity mask but left invalid gathered values
intact. Masking attention logits is insufficient because invalid NaN values
still enter the value projection and zero times NaN remains NaN. Invalid
compressed rows are now replaced with zeros before key/value projection while
the logical mask is retained. Prefill became finite and selected token 28.

### 5. Separate functional outputs from shared cache backing

The first decode step remained non-finite. Compiler metadata exposed four alias
outputs for seven logical DeepSeek caches. vLLM intentionally shares raw byte
pools among heterogeneous cache groups and uses different block tables to
address disjoint regions. That works for eager in-place writes. It does not work
when this model returns complete functional BF16 and FP32 cache views: Neuron's
alias-output rewrite lets a later whole-view output clobber preceding state.

DeepseekV4ForCausalLM now declares
requires_independent_kv_cache_tensors = True. The Neuron runner honors that
model capability by allocating one backing tensor per logical cache name. Other
models retain shared allocations. The corrected graph exposed seven independent
cache outputs, all decode logits became finite, and the first three tokens
matched the oracle.

This is a memory-for-correctness tradeoff for the tiny target. Do not extrapolate
its memory use to a full checkpoint. A production solution should update one
packed backing tensor functionally or teach the backend to preserve disjoint
view mutations instead of multiplying full cache-pool allocations.

### 6. Locate and fix the fourth-token query-RoPE error

Focused attention capture showed that layer-0 query and KV projections were
within 6e-7 of CPU. Gathered history was bit-exact, the key-valid mask was
identical, and the rotated KV tensor differed by at most 8e-7. The final two
channels of every rotated query, however, were exactly zero on Neuron while
they were nonzero on CPU. Those two channels are the complete
qk_rope_head_dim=2 rotary suffix.

Changing only the query layout from rank 4 to rank 3 did not change the bad
result. Replacing the partial-RoPE concatenation with a functional index_copy
produced a different compiled graph and preserved the suffix. Both the reduced
length-16 graph and the full length-64 graph then generated [28, 26, 48, 27].
This was a compiler-lowering-sensitive expression, not a cache-state error.

The investigation runner now exposes hookable attention boundaries for one
selected layer. CPU eager capture is also enabled before its warmup-free early
return, fixing previously empty CPU capture directories.

## Changed code

- vllm_neuron/model/deepseek_v4/attention.py implements fixed-shape functional
  latent scatter, returns the updated cache, and expresses partial RoPE with
  index_copy so Neuron retains the small rotated suffix.
- vllm_neuron/model/deepseek_v4/model.py threads SWA cache state token by token,
  neutralizes invalid compressed rows, declares independent cache backing, and
  provides hookable attention-internal capture boundaries.
- vllm_neuron/vllm/worker/neuron_model_runner.py allocates independent logical
  caches only for models that request them.
- vllm_neuron/vllm/worker/neuron_worker.py enables configured tensor capture in
  CPU eager mode even though that mode skips device warmup.
- tools/deepseek_v4/generate_tiny_tp1.py supports CPU eager mode and optional
  all-layer or focused attention capture, plus a reduced diagnostic maximum
  model length.
- Earlier commits in the patch series add the checkpoint builder, compilation
  diagnostics, metadata validation, bounds-safe gathers, deployment tooling,
  and deterministic-buffer restoration.

## Reproduce on another host

Create or select the deterministic checkpoint and record preflight data:

    source .venv/bin/activate
    python tools/deepseek_v4/build_tiny_checkpoint.py /tmp/deepseek-v4-tiny
    python tools/deepseek_v4/write_device_preflight.py \
      --checkpoint /tmp/deepseek-v4-tiny \
      --output /tmp/deepseek-v4-preflight.json

Run the focused tests:

    python -m pytest -q \
      test/unit/model/deepseek_v4/test_paged_cache_helpers.py \
      test/vllm_neuron/test_deepseek_v4_model_assembly.py \
      test/vllm_neuron/test_deepseek_cache_lifecycle.py \
      test/unit/vllm/worker/test_cache_metadata_validation.py

The latest run passed 55 tests.

## Fast debug workflow

Neuron must compile a static executable for each relevant graph shape. A large
server does not make one neuronx-cc invocation scale across all host CPUs; the
longest compiler passes used only a small fraction of this machine. The most
effective iteration speedup is therefore to shrink the static graph while
preserving the failing positions, and to reuse its compilation cache.

This test has an eight-token prompt and produces four tokens, so 12 is the
minimum valid model length. Length 16 reproduced the original fourth-token
mismatch exactly. It reduced graph extraction from about 218 seconds at length
64 to about 29 seconds, and a diagnostic compile/run took about three minutes
instead of roughly 16 minutes for the full graph.

Use this loop for graph-affecting changes:

1. Keep the checkpoint, maximum length, capture module set, and
   VLLM_CACHE_ROOT stable. Their graph hashes affect cache reuse.
2. Compile and diagnose with --max-model-len 16. This selects buckets 8 and 16.
3. Capture only the suspected layer's attention internals. Capturing a different
   module set changes the graph, so do not enable broad capture unnecessarily.
4. Repeat the identical command to confirm local cache hits before changing
   code. Source changes that alter the graph require a new compile.
5. Once the reduced graph passes, compile length 64 once for final validation,
   then repeat that exact command to verify all three graphs are cache hits.

Example focused device run:

    PATH="$PWD/.venv/bin:/opt/aws/neuron/bin:$PATH" \
    VLLM_NEURON_ENABLE_DEEPSEEK_V4=1 \
    VLLM_NEURON_VALIDATE_CACHE_METADATA=1 \
    VLLM_NEURON_TINY_VALIDATION_DIR=/tmp/deepseek-v4-neuron-logits-16 \
    VLLM_CACHE_ROOT=/tmp/deepseek-v4-neuron-cache-16 \
    NEURON_VISIBLE_DEVICES=16 \
    NEURON_SKIP_EFA_AFFINITY=1 \
    python tools/deepseek_v4/generate_tiny_tp1.py \
      /tmp/deepseek-v4-tiny \
      --max-model-len 16 \
      --capture-attention-internals-layer 0 \
      --capture-dir /tmp/deepseek-v4-neuron-captures-layer-0 \
      --output /tmp/deepseek-v4-neuron-16.json

Use the same --max-model-len and capture selection for the CPU oracle. Parallel
compilation of independent graph hashes may use more of the host, but it also
multiplies compiler memory use and is less useful than graph reduction for this
serial investigation. Never share a partially populated cache root between
concurrent compilers.

Capture a CPU eager oracle:

    VLLM_NEURON_ENABLE_DEEPSEEK_V4=1 \
    VLLM_NEURON_CPU_MODE=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_NEURON_TINY_VALIDATION_DIR=/tmp/deepseek-v4-cpu-logits \
    python tools/deepseek_v4/generate_tiny_tp1.py \
      /tmp/deepseek-v4-tiny \
      --output /tmp/deepseek-v4-cpu.json \
      --capture-dir /tmp/deepseek-v4-cpu-captures \
      --enforce-eager

Run Neuron on a free core, changing NEURON_VISIBLE_DEVICES as needed:

    PATH="$PWD/.venv/bin:/opt/aws/neuron/bin:$PATH" \
    VLLM_NEURON_ENABLE_DEEPSEEK_V4=1 \
    VLLM_NEURON_VALIDATE_CACHE_METADATA=1 \
    VLLM_NEURON_TINY_VALIDATION_DIR=/tmp/deepseek-v4-neuron-logits \
    VLLM_CACHE_ROOT=/tmp/deepseek-v4-neuron-cache \
    NEURON_VISIBLE_DEVICES=16 \
    NEURON_SKIP_EFA_AFFINITY=1 \
    python tools/deepseek_v4/generate_tiny_tp1.py \
      /tmp/deepseek-v4-tiny \
      --output /tmp/deepseek-v4-neuron.json \
      --capture-dir /tmp/deepseek-v4-neuron-captures

Compare logits:

    python tools/deepseek_v4/compare_tiny_logits.py \
      /tmp/deepseek-v4-cpu-logits \
      /tmp/deepseek-v4-neuron-logits \
      --tolerance 0.025

Repeat the Neuron command with the same cache root. Acceptance requires three
local cache hits, zero submitted HLOs, identical token IDs, no fault signatures,
and clean shutdown/device release.

## Remaining work

The next step is a post-index_copy, maximum-length-16 focused capture of layer 0
on CPU and Neuron. Compare q_roped first to confirm the corrected suffix, then
attended_roped, attended, the attention output, and the complete layer output.
If layer 0 remains within tolerance, repeat the focused capture for layers 1 and
2 until the first prefill tensor outside tolerance is found. Decode is already
within tolerance, so prioritize the selected prefill position rather than
expanding the capture scope.

After resolving or explaining the prefill difference, compile length 64 and
repeat from the same cache root. Acceptance requires three local cache hits,
zero submitted HLOs, identical token IDs and logits within 0.025, no fault
signatures, and clean shutdown/device release. Only then validate vllm serve,
the health endpoint, chat completions, clean logs, graceful shutdown, and device
release.

## Hugging Face checkpoint implications

Loading a Hugging Face checkpoint does not remove these fixes. Checkpoint weights
replace learned parameters, but they do not change compiled cache mutation,
cache aliasing, bounds masking, or derived non-persistent buffers. Real weights
may change how visible the remaining numerical error is; they are not a
substitute for closing the deterministic tiny-oracle mismatch.
