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

The device output is now finite through all four generation steps. Cache
persistence works well enough for the first three greedy tokens to match the CPU
oracle, but the fourth token still differs:

| Run | Generated tokens | Logits |
| --- | --- | --- |
| CPU eager oracle | [28, 26, 48, 27] | finite |
| Original Neuron graph | [53, 52, 12, 63] | finite, but stale-cache result |
| Functional cache before invalid-history fix | [0, 0, 0, 0] | non-finite at step 0 |
| Invalid-history fix with shared backing | [28, 0, 0, 0] | prefill finite; decode non-finite |
| Independent logical caches | [28, 26, 48, 45] | all four steps finite |

For the latest run, maximum absolute CPU/Neuron logit differences by step were
0.228515625, 0.234619140625, 0.2421875, and 0.16015625. The first three argmax
IDs matched. Step 3 selected CPU token 27 versus Neuron token 45. The acceptance
tolerance is 0.025, so deployment remains blocked on numerical correctness even
though compilation and runtime stability are green.

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

## Changed code

- vllm_neuron/model/deepseek_v4/attention.py implements fixed-shape functional
  latent scatter and returns the updated cache.
- vllm_neuron/model/deepseek_v4/model.py threads SWA cache state token by token,
  neutralizes invalid compressed rows, and declares independent cache backing.
- vllm_neuron/vllm/worker/neuron_model_runner.py allocates independent logical
  caches only for models that request them.
- tools/deepseek_v4/generate_tiny_tp1.py supports CPU eager mode and optional
  all-layer tensor capture.
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

The latest run passed 41 tests.

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

Compare corrected CPU and Neuron captures module by module for prefill and every
decode step. The capture tool records all three decoder layers' hyperconnection,
normalization, attention, MoE and complete layer outputs, plus the LM head. Find
the first tensor outside tolerance and then capture internal boundaries of that
module.

Likely categories, in priority order:

1. BF16 lowering differences in compressed sparse attention;
2. compressor carry/state values at the ratio-4 boundary;
3. a finite logical-versus-physical page-stride error reading the wrong row;
4. ordinary BF16 error amplified by the tiny random checkpoint.

Do not raise tolerance merely to pass token 4. Require four matching token IDs
before HTTP deployment. Then rerun from cache and validate vllm serve,
the health endpoint, chat completions, clean logs, graceful shutdown, and device
release.

## Hugging Face checkpoint implications

Loading a Hugging Face checkpoint does not remove these fixes. Checkpoint weights
replace learned parameters, but they do not change compiled cache mutation,
cache aliasing, bounds masking, or derived non-persistent buffers. Real weights
may change how visible the remaining numerical error is; they are not a
substitute for closing the deterministic tiny-oracle mismatch.
