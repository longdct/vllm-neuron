# Track upstream vLLM and add DeepSeek-V4 to vllm-neuron — Plan v2

## Goals and scope

Upgrade `vllm-neuron` from `vllm==0.21.0` to `vllm==0.26.0`, then add staged
support for `deepseek-ai/DeepSeek-V4-Flash` (284B total / 13B active). Defer
vLLM 0.27 until the required PyTorch/Neuron SDK combination is available, and
defer DeepSeek-V4-Pro until the Flash quantized implementation is working.

This revision keeps the correctness-first strategy, but makes four items
explicit prerequisites rather than implementation details discovered late:

1. dependency and API compatibility for the vLLM upgrade;
2. engine-managed heterogeneous cache lifecycle semantics;
3. compilation feasibility for 512-dimensional MLA;
4. measured peak memory feasibility for BF16 bring-up.

The first externally useful target is short-context DeepSeek-V4-Flash serving
without MTP or disaggregated inference. Production performance and one-million-
token context are later milestones.

---

## Architectural baseline

Pin the semantics of record before implementation:

- the exact `vllm==0.26.0` DeepSeek-V4 implementation and KV-cache interfaces;
- the exact DeepSeek-V4-Flash checkpoint revision used for validation;
- a bounded Transformers version range known to work with vLLM 0.26 and the
  checkpoint;
- the matching Neuron SDK, `torch-xla`, `libtorch-neuronx-lite`, NIXL, and
  compiler versions.

Record these revisions in the model README and compatibility tests. Do not use
upstream `main` as the implementation specification after development starts.

DeepSeek-V4-Flash requires:

- MLA with a 512-dimensional latent KV representation, inverse RoPE, grouped
  low-rank output projection, and attention sinks;
- per-layer SWA, c4, and c128 attention variants;
- an HCA compressor with state carried across scheduler chunks;
- CSA indexer-based selection for long contexts;
- mHC residual mixing with Sinkhorn normalization;
- routed, shared, and hash MoE paths with `sqrtsoftplus`, `noaux_tc`, top-k
  normalization, `routed_scaling_factor`, and the SwiGLU limit;
- FP8 block-scaled dense weights and FP4 experts;
- MTP, eventually, but not for initial correctness.

Normalize both known Transformers config representations:

- legacy `compress_ratios`;
- `layer_types` plus `compress_rates`.

Convert either representation into one internal per-layer configuration and
unit-test that the two forms produce identical layer types.

---

## Workstream A — upgrade vLLM 0.21.0 to 0.26.0

### A0. Establish a reproducible compatibility baseline

Before changing the pin:

1. Save the fully resolved dependency set for the current working environment.
2. Add import-smoke tests for every `vllm.*` module used by the plugin.
3. Inventory every monkeypatch, overridden upstream method, private attribute,
   imported class, and class-path string. Record:
   - application phase;
   - expected owner and signature;
   - failure mode if the patch is absent;
   - a test or startup assertion that proves it took effect.
4. Capture baseline logits for the supported GPT-OSS and Qwen3-VL recipes.
5. Capture a working 1P1D NIXL transaction if suitable hardware is available.

Treat the upgrade as a dependency and semantic port, not only a Python import
port. Compare resolved versions of Transformers, Pydantic, FastAPI, NIXL, and
other packages that affect initialization or runtime behavior.

### A1. Consolidate patches without collapsing lifecycle phases

Use `vllm_neuron/vllm/patches/` as the registry for patch definitions and
guards. Do not assume every patch can safely run from one hook. Preserve
explicit phases such as:

- module/process initialization;
- platform configuration;
- distributed initialization;
- model-runner initialization;
- worker startup and shutdown.

Each patch must be idempotent and guarded by symbol existence, owner identity,
and signature or schema-shape checks. A mismatch must produce a clear startup
error. Remove the duplicate `in_the_same_node_as` implementation and retain one
guarded patch at the correct distributed-init phase.

High-priority tripwires cover:

- scheduler and async-scheduler class paths;
- the `ParallelConfig` schema modification;
- worker shutdown and termination helpers;
- model-registry interception;
- distributed same-node detection;
- scheduler and KV-cache wrapper signatures.

### A2. Bump and lock dependencies

Set the package version to `0.26.0.1.0.0` and pin `vllm==0.26.0`. Select a
bounded Transformers range only after testing DeepSeek config normalization and
the existing supported models. Record the complete resolved environment used
by CI and release validation.

Do not declare the upgrade compatible merely because the PyTorch major/minor
version is unchanged. Run import, construction, scheduling, and model smoke
tests under the actual Neuron dependency set.

### A3. Port known upstream breaks

#### NIXL connector

Re-derive the connector against vLLM 0.26's base/pull/push split. Document why
Neuron's current 1P1D and xPyD topology maps to the selected direction. Minimize
access to private parent fields by adding a narrow Neuron-owned adapter where
possible.

Test:

- handshake and metadata validation;
- request registration and cleanup;
- block transfer and completion notification;
- error and disconnect cleanup;
- one end-to-end 1P1D request.

#### `InputBatch`

Update construction for the 0.26 signature, including required block counts
and slot-mapping modes. Add a construction test with one cache group and a
synthetic heterogeneous multi-group configuration.

#### Mirrored scheduler/model-runner logic

Re-diff and re-port:

- `NeuronModelRunner._update_states` against the pinned 0.26
  `GPUModelRunner._update_states`;
- `NeuronAsyncScheduler._update_after_schedule` against the pinned 0.26
  `AsyncScheduler`.

Add focused tests for new requests, cached requests, finished requests,
grammar state, async output, block-table changes, and batch compaction. Include
comments pointing to the pinned upstream revision and describing intentional
Neuron differences.

### A4. Workstream A exit gate

Workstream A is complete only when:

- dependency resolution and import-smoke tests pass;
- patch tripwires prove that all required patches are active;
- scheduler and `InputBatch` tests pass;
- GPT-OSS and Qwen3-VL serve successfully with acceptable logit drift;
- the NIXL 1P1D example passes, or is explicitly gated in CI when hardware is
  unavailable;
- no DeepSeek-specific code is required for these regressions to pass.

---

## Workstream B — cache infrastructure before the model

### B0. Define cache types and layouts

Implement or register the 0.26 cache specs needed by DeepSeek-V4:

- latent MLA cache with its actual single-tensor layout;
- uncompressed SWA cache;
- c4 and c128 compressed caches with their required alignment;
- compressor carry-state cache;
- indexer cache when CSA is introduced.

Do not represent the latent cache as a dummy K/V pair merely to satisfy the
current allocator. Cache byte accounting, allocation shape, binding, and
connector registration must agree on the real layout.

Teach `NeuronModelRunner.initialize_kv_cache` and model cache binding to handle
heterogeneous groups and per-layer specs. Explicitly define how cache groups
with different page sizes, alignments, and layouts are allocated.

### B1. Specify cache lifecycle semantics

For every cache and compressor-state type, define behavior for:

- new request allocation;
- chunked-prefill continuation;
- decode updates;
- batch reorder and compaction;
- request completion and abort;
- block remapping;
- prefix-cache hit and insertion;
- request fork/copy, if supported;
- disaggregated transfer, or an explicit unsupported error;
- speculative decoding, or an explicit unsupported error.

Prefer engine-managed cache specs such as `HiddenStateCacheSpec` over a
runner-local dictionary keyed by request ID. If prefix caching cannot preserve
compressor state initially, reject that configuration clearly rather than
silently producing incorrect output.

### B2. Synthetic heterogeneous-cache gate

Build a tiny synthetic model that uses SWA, latent MLA, and compressor-state
caches without DeepSeek weights. Test allocation, scheduling, chunk boundaries,
batch compaction, abort/cleanup, and repeated prefix reuse.

Workstream C cannot begin until this gate passes.

---

## Workstream C — feasibility spikes

Run these spikes before implementing the full 284B model.

### C0. MLA compilation spike

Implement the smallest correct BF16 MLA attention operation with
`head_dim=512`. It must compile and execute for both prefill and decode shapes.
The current generic contiguous-attention fallback may be used only where it is
semantically correct; the segmented path currently rejects dimensions above
128 and therefore needs a DeepSeek-specific reference path or an early NKI
kernel.

The reference implementation must cover:

- latent MQA math;
- inverse RoPE;
- attention sinks;
- SWA plus compressed-history composition;
- paged-cache reads for decode;
- fixed-shape metadata suitable for Neuron compilation.

Record compile time, graph size, numerical error, and execution time. Start an
NKI prototype here rather than postponing all 512-dimensional kernel work until
long-context support.

### C1. BF16 peak-memory spike

Create a per-rank memory model including:

- all model parameters, not only expert weights;
- quantized source tensors and BF16 destination tensors during conversion;
- sharding and alignment imbalance;
- loader and page-cache duplication;
- compiler and graph memory;
- activations and collective buffers;
- minimum useful KV and compressor cache capacity.

Prototype streaming, shard-by-shard dequantization directly into final sharded
tensors. Measure host and device peak memory on the target instance. The BF16
milestone is approved only if there is documented headroom; aggregate HBM alone
is not sufficient evidence.

If BF16 does not fit safely, move block-scaled FP8/FP4 loading ahead of the
full-model correctness milestone while retaining BF16 per-module validation.

### C2. Feasibility exit gate

Proceed only when:

- 512-dimensional MLA compiles and executes for representative prefill and
  decode buckets;
- the planned BF16 or quantized loading path has measured memory headroom;
- the cache layouts used by the spike match Workstream B's allocator.

---

## Workstream D — DeepSeek-V4 model implementation

Create `vllm_neuron/model/deepseek_v4/` with a shared model implementation and
per-layer component selection. Avoid separate near-duplicate model files for
each quantization scheme.

Suggested files:

- `config.py` — config normalization and validation;
- `factory.py` — per-layer attention/MoE selection;
- `model.py` — model, decoder layer, embeddings, and LM head;
- `attention.py` — MLA, SWA, compressed attention, sinks, and inverse RoPE;
- `compressor.py` — c4/c128 compression and carry state;
- `mhc.py` — reference mHC/Sinkhorn operations;
- `moe.py` — routed, shared, and hash experts;
- `weight_loaders.py` — checkpoint mapping and dequantization;
- `__init__.py`.

Register `DeepseekV4ForCausalLM` only after the tiny-model construction and
forward tests pass.

### D0. Component-by-component implementation

Implement and validate in this order:

1. config normalization and layer selection;
2. checkpoint name/shape mapping;
3. mHC and Sinkhorn normalization;
4. compressor output and carry state across arbitrary chunk boundaries;
5. MLA/SWA prefill;
6. MLA/SWA paged decode;
7. routed MoE scoring and selection;
8. shared expert for prefill and decode, including CPU fallback;
9. hash MoE with token IDs preserved through padding and TP/SP/EP;
10. full decoder layer;
11. tiny multi-layer model;
12. real model loading.

Use exact comparisons for discrete behavior:

- layer types;
- selected expert IDs;
- hash expert IDs;
- block and slot mappings;
- compressor-state positions;
- sparse indices when CSA is added.

Use dtype-appropriate tolerances only for floating-point tensors. A single
blanket `rtol=0.01` is not sufficient for routing and cache correctness.

### D1. Short-context dense-CSA mode

For the initial milestone, omit the indexer only when dense attention over all
eligible compressed entries is mathematically identical to top-k selection.
Derive the exact safe limit from the pinned compressor/indexer implementation,
including:

- compression kernel width and stride;
- initial offset and incomplete carry state;
- sink or reserved entries;
- the local sliding-window candidate set;
- generated tokens as well as prompt tokens.

Express the result as a checked function of total sequence length, layer type,
and compressor state. Reject requests that could cross the limit during
generation. Add tests immediately below, at, and above every boundary and
repeat them with different prefill chunk sizes.

Do not document a fixed `2048` limit until this derivation and tests prove it.

### D2. M1 exit gate — short-context correctness

Validate progressively:

1. real checkpoint tensors for one component;
2. one real decoder layer;
3. a random tiny model with multiple layer types;
4. selected real model layers;
5. full DeepSeek-V4-Flash short-context prefill and decode.

Use CPU/HF reference execution where practical, but avoid requiring two full
284B BF16 copies for every debugging iteration. Use tensor capture and
replacement to find the first diverging component.

M1 passes only when:

- multiple chunkings of the same prompt produce matching logits;
- prefill followed by decode matches unchunked reference behavior;
- batch reorder, completion, and abort do not leak compressor state;
- the dense-CSA runtime guard is enforced;
- TP/SP/EP variants preserve expert routing and acceptable logits;
- measured memory remains within the approved budget.

---

## Workstream E — native quantization

Add explicit quantization schemes and loaders for:

- FP8 128×128 block-scaled dense weights;
- `ue8m0` scale representation;
- FP4 expert weights;
- any distinct indexer/cache quantization introduced later.

Do not assume the GPT-OSS MXFP4 dequantizer has the same storage or scale
semantics. Validate checkpoint metadata, packed layout, scale broadcasting,
sharding boundaries, and tail blocks independently.

Compare each quantized component against its dequantized BF16 reference before
running end-to-end validation. Re-run the short-context gate and establish the
new per-rank memory budget.

---

## Workstream F — long context and production kernels

Implement:

- the CSA lightning indexer;
- token-level top-k compressed-KV selection;
- fixed-shape sparse metadata and gathers for Neuron;
- optimized 512-dimensional MLA prefill/decode kernels;
- quantized attention and indexer cache paths as required.

Validate indexer scores and selected indices separately from attention values.
Test duplicate/tied scores, partially filled caches, chunk boundaries, c4/c128
layers, and the union with local SWA entries.

Scale sequence lengths gradually rather than jumping directly to one million
tokens. Track accuracy, compile time, latency, memory, and cache capacity at
each step.

---

## Workstream G — MTP, performance, and disaggregated inference

After base-model correctness and native quantization:

1. add MTP/speculative decoding and its cache fork/rollback semantics;
2. tune mHC, compressor, MoE, and attention kernels;
3. enable prefix caching where compressor-state semantics are proven;
4. extend NIXL transfer to all required DeepSeek cache/state tensors;
5. benchmark prefill, decode, throughput, and long-context scaling;
6. add DeepSeek-V4-Pro as a config, sharding, and capacity exercise only after
   architectural parity is demonstrated.

Unsupported combinations must fail during configuration rather than silently
falling back to incorrect behavior.

---

## Test and CI plan

Create `test/unit/` and `test/vllm_neuron/` and add staged jobs:

### CPU/unit jobs

- config normalization for legacy and modern Transformers forms;
- weight-name and tensor-shape mapping;
- mHC/Sinkhorn numerical tests;
- compressor chunk-boundary and lifecycle tests;
- exact MoE routing tests;
- synthetic cache allocation and cleanup;
- dense-CSA safe-limit derivation and boundary checks;
- upstream API/signature tripwires.

### Neuron compile/simulator jobs

- tiny model compilation;
- MLA prefill/decode buckets with `head_dim=512`;
- heterogeneous cache binding;
- TP/SP/EP routing and collective shapes;
- quantized component kernels.

### Hardware integration jobs

- supported-model regressions after the vLLM upgrade;
- DeepSeek tiny-model prefill/decode;
- full-model short-context correctness;
- NIXL 1P1D transfer;
- long-context tests when CSA lands.

Store pinned baseline logits and report drift rather than relying only on
top-token agreement.

---

## Milestones and decision gates

| Milestone | Deliverable | Required exit evidence |
|---|---|---|
| A | vLLM 0.26 compatibility | locked dependencies, patch/API tests, supported-model and NIXL regressions |
| B | heterogeneous cache infrastructure | synthetic lifecycle tests across chunking, reorder, abort, and prefix policy |
| C | feasibility | compiled 512-d MLA spike and measured per-rank peak-memory headroom |
| D/M1 | BF16 or fallback short-context correctness | component, tiny-model, and real-model prefill/decode validation with enforced safe limit |
| E/M2 | native FP8/FP4 | component accuracy, full-model regression, and measured memory reduction |
| F/M3 | long context | correct CSA indices/attention plus scaled context benchmarks |
| G/M4 | MTP, performance, DI | rollback-safe speculation, production kernels, complete cache transfer |

If Milestone C shows that BF16 full-model loading is unsafe, quantization moves
before D/M1 full-model validation. If the 512-dimensional MLA spike cannot
compile or has unusable performance, prioritize the NKI kernel before further
model assembly.

---

## Primary risks

1. **512-dimensional MLA kernel feasibility.** Mitigate with the pre-model
   compilation spike and early NKI prototype.
2. **Compressor-state lifecycle correctness.** Mitigate with engine-managed
   state and synthetic scheduler tests before model integration.
3. **BF16 peak memory.** Mitigate with streaming sharded loading and a measured
   gate; move quantization earlier if necessary.
4. **Private upstream coupling, especially NIXL.** Mitigate with guarded
   adapters, compatibility tests, and a pinned upstream revision.
5. **Config/checkpoint format drift.** Mitigate with normalization adapters,
   bounded dependency versions, and tests for both config forms.
6. **Static-graph shape explosion.** Mitigate by defining supported buckets and
   fixed-shape metadata early, then tracking compile count and graph memory.
7. **False correctness from short-context dense attention.** Mitigate with an
   exact eligibility bound, runtime rejection, and boundary tests.

---

## Completion criteria

DeepSeek-V4-Flash support is complete only when the documented configuration:

- loads without peak-memory failure on the target hardware;
- produces validated prefill and decode logits;
- preserves compressor and KV state across supported scheduler operations;
- uses native checkpoint quantization without load-time BF16 expansion;
- either implements CSA for the advertised context length or enforces a tested
  shorter limit;
- clearly rejects unsupported prefix caching, MTP, or DI combinations;
- includes reproducible dependency pins, run examples, tests, and benchmark
  results.
