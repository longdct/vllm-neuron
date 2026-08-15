# Track upstream vLLM and add DeepSeek-V4 to vllm-neuron

## Context

`vllm-neuron` is the AWS Trainium platform plugin for vLLM, pinned to `vllm==0.21.0`
(`requirements/core.txt:6`, package version `0.21.0.1.0.0`) while upstream is at 0.27.1. Two goals:
get back onto a current upstream, and serve DeepSeek-V4 — which the plugin cannot run at all today,
since nothing in the repo implements latent (MLA) attention, sparse attention, or block-scaled
FP8/FP4.

Target: **vLLM 0.26.0**, then **DeepSeek-V4-Flash** (284B total / 13B active), staged
correctness-first. Rename to `0.26.0.1.0.0`, branch `release-0.26.0.1.0.0`.

### Evidence behind the version choice

vLLM 0.21 *through 0.26* all pin `torch==2.11.0`; 0.27.0 jumps to `torch==2.13.0`. The Neuron pip
index tops out at `torch-xla 2.12` and `libtorch-neuronx-lite 2.12` — the packages this plugin
actually uses, via the redirector in `vllm_neuron/utils/import_redirector.py`, not the heavier
`torch-neuronx` (which is stuck at torch 2.9). So **0.21→0.26 needs no torch change, and 0.27 is
blocked on an AWS SDK release, not on this repo.** Confirm the torch-2.13 roadmap with AWS; it
alone sets the 0.27 date, and the evidence here is the package index, not an AWS statement.

vLLM ships roughly biweekly (0.21 → 0.27 was 2026-05-15 → 2026-08-10). Landing on 0.26 closes the
gap once; staying closed needs the guard rails in A1 plus a compat CI job.

---

## Architectural baseline

Pin the semantics of record before implementation and record them in the model README and
compatibility tests — **the pinned `vllm==0.26.0` tree, not upstream `main`**: the DeepSeek-V4
reference at `vllm/models/deepseek_v4/` (`attention.py`, `sparse_mla.py`, `compressor.py`,
`nvidia/model.py`), the KV-cache interfaces, the exact DeepSeek-V4-Flash checkpoint revision, a
bounded Transformers range, and the matching Neuron SDK / `torch-xla` /
`libtorch-neuronx-lite` / NIXL / compiler versions.

From `deepseek-ai/DeepSeek-V4-Flash/config.json` and that reference:

| | V4-Flash (target) | V4-Pro |
|---|---|---|
| hidden / layers / heads | 4096 / 43 / 64 | 7168 / 61 / 128 |
| `head_dim`, `num_key_value_heads` | 512, 1 | 512, 1 |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | 1024 / 1024 / 8 | 1536 / 1024 / 16 |
| experts: routed / top-k / shared / interm. | 256 / 6 / 1 / 2048 | 384 / 6 / 1 / 3072 |
| indexer: heads × dim, `index_topk` | 64 × 128, 512 | 64 × 128, 1024 |
| `compress_ratios` | `[0,0,4,128,4,128,…,0]` | `[128,4,128,…,0]` |

Shared: `scoring_func=sqrtsoftplus`, `topk_method=noaux_tc`, `norm_topk_prob`,
`swiglu_limit=10.0`, `num_hash_layers=3`, `sliding_window=128`, `hc_mult=4`,
`hc_sinkhorn_iters=20`, `num_nextn_predict_layers=1` (MTP), YaRN (factor 16 from 65536),
`vocab_size=129280`, FP8 `weight_block_size=[128,128]` with `ue8m0` scales and
`expert_dtype=fp4`.

**Two scope findings.** (1) Unlike V3, the V4 configs carry no `n_group`/`topk_group` — grouped /
node-limited top-k routing, which would have been the largest MoE gap, is **not needed**. (2) The raw
checkpoint contains `compress_ratios`, but the loaded Transformers config may expose either
`compress_ratios` or normalized `layer_types` + `compress_rates`, depending on the pinned
Transformers version. Normalize both forms into one internal per-layer representation, test their
equivalence, and fail loudly on an unrecognized form.

The five architectural pieces:

- **MLA.** `fused_wqa_wkv` → q-lora + a single 512-d latent KV per token; `wq_b` expands to
  64×512; MQA against the latent; low-rank grouped output projection (`wo_a`/`wo_b`, `o_groups`)
  with inverse RoPE; per-head attention sinks.
- **HCA.** Per-layer `compress_ratio` ∈ {4, 128}: `compressor.py:211 DeepseekCompressor` folds
  `compress_ratio` tokens into one cached latent using learned `ape` positional weights, carrying
  partial state across chunk boundaries in a `CompressorStateCache`. Ratio 0 → sliding-window (128).
- **CSA.** A 64-head/128-dim lightning indexer scores compressed entries; attention runs over the
  top `index_topk`. **The indexer only selects; it never weights** — so where the eligible
  compressed set is no larger than `index_topk`, CSA is exactly dense attention. This is the lever
  the staged bring-up rides on (see D1 — the bound must be *derived*, not assumed).
- **mHC.** `hc_mult=4` widens the residual stream 4×, with per-layer mixing matrices normalized by
  20 Sinkhorn iterations. Upstream fuses this into tilelang; a plain decomposition traces fine.
- **MoE.** 256 experts, top-6, one shared, `sqrtsoftplus` scoring, `noaux_tc` selection bias
  (`e_score_correction_bias` affects *selection only*), `routed_scaling_factor=1.5`. Layers 0–2 are
  **hash MoE**: `tid2eid[input_ids]` fixes which experts are selected, while the learned gate still
  produces the weights applied to those selected experts.

---

## Workstream A — upgrade vLLM 0.21.0 → 0.26.0

### A0. Capture the baseline before touching the pin

Save the fully resolved dependency set of the current working environment; capture baseline logits
for the supported GPT-OSS and Qwen3-VL recipes; capture a working 1P1D NIXL transaction if hardware
allows. The A3.3 drift risk is silent, so a pre-upgrade logit baseline is the only real check —
it must exist before the pin moves.

### A1. Patch registry with lifecycle phases and tripwires

`vllm_neuron/vllm/patches/__init__.py:15 apply_patches()` is empty and uncalled, while ~12 patch
sites are scattered across `platform.py`, `neuron_parallel_state.py`, `core/scheduler.py`,
`neuron_worker.py`, and `neuron_nixl_connector.py`. Make `patches/` the registry for definitions
and guards — but **do not collapse them into one hook**: they apply at genuinely different phases
and some must stay where they are. `apply_port_hold_patch()` is called at *import* time
specifically so it survives spawn-mode re-imports (`vllm_neuron/__init__.py:196-198` — the
EngineCore subprocess never calls `check_and_update_config`), whereas
`_register_neuron_all2all_backend` must run inside `check_and_update_config`. Preserve module/process
init, platform config, distributed init, model-runner init, and worker start/shutdown as distinct
phases.

Each patch: idempotent, guarded on symbol existence + owner identity + signature/schema shape,
failing loudly at startup on mismatch. Model the guards on the existing tripwire at
`vllm_neuron/vllm/core/scheduler.py:304-307`. This converts the silent-failure class — a renamed
upstream symbol that makes a patch a no-op — into loud errors, which is the difference between a
one-week upgrade and a month of chasing wrong behaviour. Priority tripwires:

- scheduler / async-scheduler class-path strings — `vllm/platform.py:376-379`; if these stop
  matching, the GPU scheduler silently runs on Neuron.
- the `ParallelConfig.__pydantic_core_schema__` 4-level walk — `vllm/platform.py:769-780`.
- `_patch_shutdown` / `_ensure_worker_termination` — `vllm/platform.py:863-902`.
- the model-registry log-message regex — `vllm/worker/neuron_worker.py:179`.

Also **deduplicate `in_the_same_node_as`**: `neuron_worker.py:486-490` and
`neuron_parallel_state.py:818-822` are byte-identical bodies. Keep one, guarded, at the
distributed-init phase.

### A2. Verified non-issues — confirm, don't rewrite

Diffed 0.21 vs 0.26 directly. No source change was identified for these by the static diff; retain
runtime coverage rather than re-deriving them unnecessarily:

- `Scheduler.__init__` — 8-arg signature **identical**; the mirror at `core/scheduler.py:89` stands.
- `Platform.get_attn_backend_cls(selected_backend, attn_selector_config, num_heads=None)` —
  **identical**.
- `KVCacheManager.allocate_slots` gained `reserved_blocks` and `has_scheduled_reqs`, both defaulted;
  the SWA-DI wrapper at `core/scheduler.py:321` takes `*args, **kwargs`, so they pass through and
  its tripwire still holds.
- `SchedulerOutput` is still a plain mutable `@dataclass` — the injected `_grammar_bitmask` /
  `num_scheduled_tokens_padded` fields still work.
- `ParallelConfig.all2all_backend` is still a Literal-typed field.
- All ~45 `vllm.*` modules the plugin imports resolve at 0.26; `Platform` changes are purely additive.

This is a *static* check only. It does not prove runtime compatibility — Transformers (0.24+ needs
`>=5.5.3`, repo has `>=5.5.1`), Pydantic, FastAPI, and NIXL all affect init and runtime behaviour.
Run import, construction, scheduling, and model smoke tests under the real Neuron dependency set.

### A3. The three real breaks

1. **NIXL connector — the big one.** Upstream split `NixlConnector` into `NixlBaseConnector` /
   `NixlPullConnector` / `NixlPushConnector`, moving worker and scheduler into `nixl/base_worker.py`
   (110 KB), `pull_worker.py`, `push_worker.py`, `pull_scheduler.py`, `push_scheduler.py`.
   `vllm_neuron/vllm/kv_connector/neuron_nixl_connector.py` subclasses `NixlConnectorWorker`,
   overrides five *private* methods (`_nixl_handshake`, `_read_blocks_for_req`,
   `_validate_remote_agent_handshake`, …) and reads ~15 private parent attributes — **none of which
   import at 0.26**. Re-derive against the pull/push split and document why Neuron's 1P1D/xPyD
   topology maps to the chosen direction. Reduce private-field reach with a narrow Neuron-owned
   adapter. The comment at `neuron_nixl_connector.py:96-99` shows this file already broke on the
   0.19→0.20 bump; budget accordingly. Test handshake/metadata validation, request registration and
   cleanup, block transfer and completion, disconnect cleanup, and one end-to-end 1P1D request.
2. **`InputBatch` construction** — `neuron_model_runner.py:7658`. Upstream dropped `pin_memory`
   (now a module-level `PIN_MEMORY`), made `max_num_blocks_per_req` required rather than optional,
   and added `slot_mapping_modes`. Add a construction test for one cache group and for a synthetic
   heterogeneous multi-group config.
3. **Re-sync the two upstream ports.** `GPUModelRunner` changed ~1900 of 7900 lines and `Scheduler`
   ~1050 of 2900 between the tags. Both of these are line-for-line ports that drift *silently*:
   - `NeuronModelRunner._update_states` (`neuron_model_runner.py:1815-2036`) vs upstream
     `GPUModelRunner._update_states` — re-diff field by field, especially
     `scheduler_output.scheduled_cached_reqs.*` and `CachedRequestState`.
   - `NeuronAsyncScheduler._update_after_schedule` (`core/scheduler.py:943-1023`) vs upstream
     `AsyncScheduler`.

   Test new/cached/finished requests, grammar state, async output, block-table changes, and batch
   compaction. Comment each port with the pinned upstream revision and the intentional Neuron deltas.

### A4. Exit gate

Dependency resolution and import smoke tests pass; patch tripwires prove every required patch is
active; scheduler and `InputBatch` tests pass; GPT-OSS and Qwen3-VL serve with acceptable logit
drift against the A0 baseline; NIXL 1P1D passes or is explicitly gated when hardware is
unavailable — and none of it requires DeepSeek code.

---

## Workstream B — cache infrastructure, before any model code

This is the hardest *systems* risk and it can be fully validated without DeepSeek weights. Do it first.

### B0. Cache types and layouts

0.26 already carries the engine-side machinery, added upstream precisely for this model:
`MLAAttentionSpec` gained `compress_ratio` / `alignment` / `model_version` / `kv_quant_mode`, plus
new `HiddenStateCacheSpec` and `RSWASpec` kinds, and `Platform` gained
`register_custom_kv_cache_specs()` and `_align_heterogeneous_kv_block_size()`. Wire these in
`vllm_neuron/vllm/platform.py` and register: the latent MLA cache in its real single-tensor layout,
the uncompressed SWA cache, the c4 and c128 compressed caches with their alignment, the compressor
carry-state cache, and (at F) the indexer cache.

Reproduce the pinned vLLM 0.26 cache-spec and logical-block semantics unless a Neuron constraint
requires a documented deviation. In particular, upstream models compressor residuals with
sliding-window cache semantics and uses one logical block size measured in native token positions;
c4 and c128 layers then store different numbers of compressed entries per logical block. Compressor
carry state must participate in the same block lifecycle as the corresponding compressed cache.

**Do not fake the latent cache as a dummy K/V pair.** The current allocator hands out
`(2, blocks, kv_heads, block_size, head_size)` at `neuron_model_runner.py:7731-7741`; a latent
stored in `k` with `v` unused wastes half the cache and desynchronizes byte accounting from
allocation shape, binding, and connector registration. `model/kv_cache.py` `LayerSpec` can already
express `num_kv_heads=1, head_size=512` — the spec is fine, the pair layout is not.

Teach `NeuronModelRunner.initialize_kv_cache` (`:7770-7773`, currently `NotImplementedError` for
anything but `FullAttentionSpec` / `SlidingWindowSpec`) to handle heterogeneous groups and per-layer
specs, and define explicitly how groups with differing page sizes, alignments, and layouts are
allocated.

### B1. Lifecycle semantics

For every cache and compressor-state type, define behaviour for: new-request allocation; chunked
prefill continuation; decode updates; batch reorder and compaction; completion and abort; block
remapping; prefix-cache hit and insertion; fork/copy if supported; disaggregated transfer; and
speculative decoding. The last two may be an explicit unsupported error — but it must be an error,
not a silent fallback.

Prefer the pinned upstream engine-managed cache representation over a runner-local dict keyed by
request id; do not preselect `HiddenStateCacheSpec` if vLLM 0.26 uses SWA-style semantics for that
state. If prefix caching cannot preserve compressor state initially, reject that configuration at
startup.

### B2. Synthetic heterogeneous-cache gate

Build a tiny synthetic model — extend `vllm_neuron/model/synthetic/` — exercising SWA, latent MLA,
and compressor-state caches with no DeepSeek weights. Test allocation, scheduling, chunk boundaries,
compaction, abort/cleanup, and repeated prefix reuse. **Workstream C does not start until this passes.**

---

## Workstream C — feasibility spikes

### C0. 512-dim MLA compilation spike

The smallest correct BF16 MLA op with `head_dim=512`, compiling and executing for both prefill and
decode shapes. `functional/attention/attention_cte.py:16` caps `MAX_HEAD_DIM=128` but falls back to
torch; `attention_segmented_cte.py:36` **raises** above 128, so the segmented path needs a
DeepSeek-specific reference path or an early NKI kernel. Cover latent MQA math, inverse RoPE,
sinks, SWA + compressed-history composition, paged decode reads, and fixed-shape metadata suitable
for Neuron compilation. Record compile time, graph size, numerical error, execution time. **Start
the NKI prototype here** against the NKI CPU simulator rather than deferring all 512-d kernel work
to F — this is the one item that can invalidate the schedule.

### C1. BF16 peak-memory spike

Trn2 is 96 GB/chip × 16 = 1,536 GB aggregate, and Flash's ~277B expert params are ~554 GB at BF16 —
but **aggregate HBM is not evidence of feasibility.** Build a per-rank model covering: all
parameters (not just experts); quantized source *and* BF16 destination tensors held simultaneously
during conversion; sharding and alignment imbalance; loader and page-cache duplication; compiler and
graph memory; activations and collective buffers; and minimum useful KV + compressor capacity.
Prototype streaming shard-by-shard dequantization directly into final sharded tensors
(precedent: `gpt_oss/weight_loaders_bf16.py:264 _dequantize_mxfp4_to_bf16`). Measure host and device
peak on the target instance.

**If BF16 does not fit with documented headroom, Workstream E moves ahead of D's full-model
milestone**, keeping BF16 for per-module validation only.

### C2. Exit gate

512-d MLA compiles and executes for representative prefill and decode buckets; the planned loading
path has measured headroom; the spike's cache layouts match B0's allocator.

---

## Workstream D — the model

`vllm_neuron/model/deepseek_v4/` with one implementation and **per-layer** component selection.
Copy llama3's `resolve_attention_mlp_classes` pattern (`model/llama3/quantization.py:236-309`),
which picks attention/MLP classes per layer — exactly what V4 needs, since SWA, c4, c128, hash-MoE,
and MTP layers differ structurally. Do **not** copy gpt_oss's whole-model fork per quantization
(`model_bf16.py` and `model_mxfp4.py` are near-duplicate 2200-line files).

Files: `config.py` (normalization + validation), `factory.py` (per-layer selection), `model.py`,
`attention.py`, `compressor.py`, `mhc.py`, `moe.py`, `weight_loaders.py`, `__init__.py`, plus
`examples/vllm_neuron/models/deepseek_v4/` and a model README
(template: `docs/model-dev/onboarding-models.md:675-712`). Register `DeepseekV4ForCausalLM` in
`model/registry.py:19-24` **only after** tiny-model construction and forward tests pass. Follow the
official 5-step process in `docs/model-dev/onboarding-models.md`.

### Reusable as-is

`NF.router` supports sigmoid scoring and the `PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER` order (i.e.
`norm_topk_prob`); `NF.moe_block_tkg` handles decode top-k renorm **and shared experts**;
`NF.build_blockwise_mapping` + `NF.moe_cte` cover prefill; expert parallelism with contiguous
placement (`functional/expert_parallel.py`); and notably `expert_parallel_interleaved_loader`
(`utils/weight_loader.py:676,727`) was written against DeepSeek checkpoint layouts (bf16 stride 2,
fp8 stride 4). Plus the TP/SP/DP/EP collectives, paged + FP8 KV cache, on-device sampling, and
`vllm_neuron/accuracy/`.

### Must be built

| Gap | Where it bites |
|---|---|
| MLA / latent attention | `attention_cte.py:16` (`MAX_HEAD_DIM=128`, falls back); `attention_segmented_cte.py:36` (**raises**) |
| Single-tensor latent KV cache | see B0 |
| HCA compressor + carry state | new; strided reduction with cross-chunk state |
| mHC (4× residual + Sinkhorn) | new; decomposed PyTorch first |
| CSA indexer + token-level top-k gather | only contiguous `bound_min/bound_max`, SWA, and block-granular `active_blocks_table` exist — `attention_cte.py:202`, `attention_decode.py:944` |
| `noaux_tc` bias, `routed_scaling_factor`, `sqrtsoftplus` | `router_bias` applies pre-activation so it also changes the gate value — wrong for `noaux_tc` (`functional/moe/router.py:75,568-599`) |
| Hash MoE (layers 0–2) | selection lookup is simple, but token-ID alignment and learned gate weighting require validation |
| Shared expert in **prefill** | only wired for decode; absent from the torch fallback, breaking CPU parity (`moe_block_tkg.py:361-363`) |
| FP8 128×128 block-scale + FP4 experts | `QuantScheme` has only `NONE` and `FP8_STATIC_PER_TENSOR` (`llama3/quantization.py:39-51`) |
| MTP | only EAGLE3 exists |

### D0. Component order

Config normalization and layer selection → checkpoint name/shape mapping → mHC/Sinkhorn →
compressor output and carry state across arbitrary chunk boundaries → MLA/SWA prefill → MLA/SWA
paged decode → routed MoE scoring/selection → shared expert (prefill, decode, CPU fallback) →
hash MoE with token ids preserved through padding and TP/SP/EP → full decoder layer → tiny
multi-layer model → real model loading.

**Compare discrete behaviour exactly** — layer types, selected and hash expert ids, block and slot
mappings, compressor-state positions, sparse indices. Reserve dtype-appropriate tolerances for
float tensors; a blanket `rtol=0.01` cannot validate routing or cache correctness.

### D1. Short-context dense-CSA mode — derive the bound

Omit the indexer only where dense attention over all eligible compressed entries is *mathematically
identical* to top-k selection. Derive the safe limit from the pinned compressor/indexer
implementation: compression kernel width and stride, initial offset and incomplete carry state,
sink or reserved entries, the local sliding-window candidate set, and **generated tokens as well as
prompt tokens** — a request can cross the bound mid-generation. Express it as a checked function of
sequence length, layer type, and compressor state; reject requests that could cross it. Test
immediately below, at, and above every boundary, repeated across prefill chunk sizes.

A naive reading suggests ~2048 tokens for Flash (`index_topk=512` × the c4 layers' ratio). **Do not
ship that number until the derivation and boundary tests prove it.**

### D2. M1 exit gate

Validate progressively — real checkpoint tensors for one component, then one real decoder layer,
then a random tiny model with multiple layer types, then selected real layers, then full
short-context prefill and decode. Use tensor capture/replacement
(`vllm_neuron/accuracy/`) to find the first diverging component; with mHC, HCA, and MLA all new at
once this is the only tractable debugging loop. Avoid requiring two full 284B BF16 copies per
iteration.

Passes when: multiple chunkings of one prompt produce matching logits; prefill-then-decode matches
unchunked reference; batch reorder/completion/abort leak no compressor state; the dense-CSA guard is
enforced; TP/SP/EP variants preserve routing and logits; measured memory stays within the C1 budget.

---

## Workstream E — native quantization

FP8 128×128 block-scaled dense weights, `ue8m0` scales, FP4 experts, and any distinct
indexer/cache quantization added later. **Do not assume the GPT-OSS MXFP4 dequantizer shares
storage or scale semantics** — MX is per-32-element microscaling, a different family with
MX-specific tiled scale shapes. Validate checkpoint metadata, packed layout, scale broadcasting,
sharding boundaries, and tail blocks independently. Compare each quantized component against its
dequantized BF16 reference before end-to-end validation. Re-run the D2 gate and establish the new
per-rank budget. The FP4 expert payload is approximately 138 GB; determine total resident and peak
memory through C1's measured accounting rather than treating that figure as the complete model.

---

## Workstream F — long context and production kernels

The CSA lightning indexer; token-level top-k compressed-KV selection; fixed-shape sparse metadata
and gathers for Neuron; optimized 512-d MLA prefill/decode kernels; quantized attention and indexer
cache paths. Validate indexer scores and selected indices **separately** from attention values.
Test tied/duplicate scores, partially filled caches, chunk boundaries, c4 vs c128 layers, and the
union with local SWA entries. Scale context gradually rather than jumping to 1M, tracking accuracy,
compile time, latency, memory, and cache capacity at each step.

---

## Workstream G — MTP, performance, DI

MTP/speculative decoding with cache fork/rollback semantics; kernel tuning for mHC, compressor, MoE,
attention; prefix caching once compressor-state semantics are proven; NIXL transfer extended to all
DeepSeek cache/state tensors; prefill/decode/throughput/long-context benchmarks. **V4-Pro lands
here** as a config, sharding, and capacity exercise once architectural parity is demonstrated.
Derive its resident and peak-memory requirements from checkpoint tensor accounting rather than the
provisional ~800 GB FP4 estimate. Unsupported combinations fail at configuration time, never as a
silent fallback.

---

## Test and CI

There is **no harness to plug into** — `test/` and `ci/` do not exist despite `pyproject.toml:46-50`
referencing them (this is a stripped public mirror), and `.github/workflows/` holds only
issue-labeling automation. Creating it is part of the work.

- **CPU/unit** — legacy/normalized config equivalence and validation; weight-name and shape mapping; mHC/Sinkhorn; compressor
  chunk-boundary and lifecycle; exact MoE routing; synthetic cache allocation and cleanup; the D1
  safe-limit derivation and boundaries; upstream API/signature tripwires. Use the doc's tiny-model
  pattern (`onboarding-models.md:864-915`): random 1-layer model, `enforce_eager=True`,
  `num_gpu_blocks_override=4`, `hidden_size % 256 == 0`.
- **Neuron compile / simulator** — tiny-model compilation; MLA prefill/decode buckets at
  `head_dim=512`; heterogeneous cache binding; TP/SP/EP routing and collective shapes; quantized
  component kernels.
- **Hardware integration** — supported-model regressions post-upgrade; DeepSeek tiny-model
  prefill/decode; full-model short-context correctness; NIXL 1P1D; long-context once CSA lands.

Component checks use `assert_close_three_way(target=neuron, expected=hf_fp32, baseline=hf_bf16)`
(`accuracy/testing.py:289`); end-to-end uses `multi_prompt_logit_validation`
(`accuracy/logit_validation.py`) across a TP/seqlen/batch grid. Store pinned baseline logits and
report drift — top-token agreement alone hides real regressions.

---

## Milestones and gates

| Milestone | Deliverable | Exit evidence |
|---|---|---|
| A | vLLM 0.26 compatibility | locked deps, patch/API tripwires, supported-model + NIXL regressions |
| B | heterogeneous cache infra | synthetic lifecycle tests across chunking, reorder, abort, prefix policy |
| C | feasibility | compiled 512-d MLA spike; measured per-rank memory headroom |
| D / M1 | short-context correctness | component → tiny-model → real-model prefill/decode, safe limit enforced |
| E / M2 | native FP8/FP4 | component accuracy, full regression, measured memory reduction |
| F / M3 | long context | correct CSA indices and attention; scaled context benchmarks |
| G / M4 | MTP, perf, DI, V4-Pro | rollback-safe speculation, production kernels, complete cache transfer |

Two re-orderings are pre-authorized: if C1 shows BF16 full-model loading is unsafe, E moves before
D's full-model validation; if the 512-d MLA spike cannot compile or is unusably slow, the NKI kernel
takes priority over further model assembly.

## Primary risks

1. **512-d MLA kernel feasibility** — mitigated by the C0 spike and an early NKI prototype.
2. **Compressor-state lifecycle correctness** — mitigated by engine-managed state and the B2 gate.
3. **BF16 peak memory** — mitigated by streaming sharded loading and the C1 measured gate.
4. **Private upstream coupling, especially NIXL** — mitigated by guarded adapters, compat tests, a
   pinned revision.
5. **Static-graph shape explosion** — define supported buckets and fixed-shape metadata early, then
   track compile count and graph memory.
6. **False correctness from short-context dense attention** — mitigated by the D1 derived bound,
   runtime rejection, and boundary tests.
7. **Neuron torch-2.13 availability** gates 0.27 entirely and is outside this repo's control.

## Completion criteria

DeepSeek-V4-Flash is done when the documented configuration loads without peak-memory failure on
target hardware; produces validated prefill and decode logits; preserves compressor and KV state
across all supported scheduler operations; uses native checkpoint quantization with no load-time
BF16 expansion; either implements CSA for the advertised context or enforces a tested shorter limit;
clearly rejects unsupported prefix-caching / MTP / DI combinations; and ships reproducible pins, run
examples, tests, and benchmarks.
