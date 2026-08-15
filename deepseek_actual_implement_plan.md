# DeepSeek-V4 on vllm-neuron — execution plan

Supersedes `deepseek_implement_plan_v3.md`. v3 is the **reference document**: it holds the
architectural baseline, the config tables, the per-subsystem gap analysis, and the risk register, and
this plan does not repeat them. What changes here is **sequencing**.

v3 was organized by workstream (A…G), which reads as "finish each subsystem, then move on". That
ordering defers the two items most likely to force architectural rework — the heterogeneous cache
lifecycle and 512-dimensional attention — behind a large volume of conventional work (MoE, quant,
loaders). This plan re-cuts the same content along the **critical path**, so that every phase either
retires a schedule-killing unknown or is the smallest thing that unblocks the next one.

```
  P0  vLLM 0.26 runtime (development gate; release gate is separate)
   └─ P1  heterogeneous cache lifecycle          ← systems risk
       └─ P2  512-d MLA spike, through Trn2      ← largest schedule risk
           └─ P3  tiny model: P3a vertical layer → P3b all layer variants
               └─ P4  component correctness vs a named oracle
                   └─ P5  dense-CSA safety bound (derived + enforced)
                       └─ P6  end-to-end chunked logit correctness
                           └─ P7  memory gate ─┬─ BF16 fits ──→ P8 → P9
                                               └─ doesn't ────→ P9 → P8
  P8  M1 — full Flash, BF16, short context   (bring-up, NOT completion)
  P9  M2 — native FP8/FP4                    (mandatory for "done")
```

Everything after P9 — CSA/long context, MTP, DI/NIXL restoration, perf, V4-Pro — stays as scoped in
v3 §F/§G.

**Guiding rule for P0–P3: build vertically, not horizontally.** The goal of the first four phases is
one token of output from structurally-faithful layers, not a complete subsystem anywhere.

---

## Implementation status

| Item | Phase | State |
|---|---|---|
| Version pins moved to vLLM 0.26 / `0.26.0.1.0.0` / `transformers>=5.5.3` | **P0.1** | **done** — pre-bump state preserved at commit `be0def6` |
| Patch registry, phases, guards, tripwires | P0.2 | **built**, validated against **0.21** — `VALIDATED_VLLM_VERSION` stays `0.21.0` until tripwires pass on a real 0.26 tree |
| Bare-interpreter test harness | — | **built** — `test/unit/conftest.py` loads vllm-free modules by path |
| **Config normalization + validation** (`model/deepseek_v4/config.py`) | **P3a.1** | **done** — 31 tests |
| **Dense-CSA bound + admission guard** (`model/deepseek_v4/dense_csa.py`) | **P5** | **arithmetic done** — 40 tests; compressor constants still required as inputs |
| **Per-rank memory accounting model** (`model/memory_budget.py`) | **P7a** | **done** — 32 tests; needs GPT-OSS calibration to pass its gate |
| Everything else | P0.2–P9 | environment-blocked (see below) |

Suite: **171 passed, 5 skipped**, no torch/vllm. New modules are mutation-checked — deliberately
breaking the bound formula, the admission rule, and the streaming-peak formula each produced failures.

**Why P5 and P7a landed before P0–P2.** P0.2, P1, P2 all need a Linux box with `vllm==0.26.0` or a
Trn2 instance, and neither exists on the development machine. The three items above are the critical
path's only environment-independent work: config normalization is data-shape logic, the CSA bound is
integer arithmetic, and weight accounting reads safetensors headers. All three were written
dependency-free and tested on a bare interpreter, which is the intended use of the local tier — not a
reordering. Their *dependent* work is untouched: P5's derivation still needs the pinned compressor
constants, P7a still needs calibration against a measured GPT-OSS peak, and P3a's remaining steps
(weight mapping, mHC, compressor, MLA, MoE, decode) all need torch.

**What the two new modules deliberately refuse to do.** `dense_csa.CompressorGeometry` has no default
kernel width, stride, or offset — a defaulted constant would yield a plausible bound that had never
been derived, which is precisely the ~2048-token trap the plan warns against. `memory_budget` marks
every line `EXACT` or `ESTIMATED`, and `MemoryBudget` exposes no scalar total, so a modelled range can
never be quoted as a measurement; `fits_in()` returns `None` when the range straddles capacity, which
is the signal to run P7b rather than to guess.

## Ground truth as of this plan

Verified against the working tree, not assumed:

| Fact | Evidence |
|---|---|
| Pin is `vllm==0.21.0`, package `0.21.0.1.0.0` | `requirements/core.txt:6`, `pyproject.toml:10` |
| Patch registry **already built** — phases, guards, tripwires | `vllm_neuron/vllm/patches/{registry,guards,tripwires,node_topology}.py`, uncommitted |
| Guards are validated against **0.21**, not 0.26 | `patches/guards.py:39` `VALIDATED_VLLM_VERSION = "0.21.0"` |
| Test harness **bootstrapped**, small | `test/unit/…`, `test/vllm_neuron/test_upstream_compat.py`, uncommitted |
| `InputBatch` constructed in one place | `neuron_model_runner.py:7658` |
| `initialize_kv_cache` rejects non-Full/SWA specs | `neuron_model_runner.py:7625`, raises at `:7767`, `:7778` |
| 512-d attention: CTE falls back to torch, segmented **raises** | `attention_cte.py:16,40`; `attention_segmented_cte.py:36,440,596` |
| Synthetic model exists and is env-gated | `model/synthetic/`, `model/registry.py:29-35` |
| Model registry has 4 entries, no DeepSeek | `model/registry.py:19-24` |
| Transformers pinned `>=5.5.1,<6.0.0` | `requirements/core.txt:12`, `requirements/test.txt:3` |

So v3's Workstream A1 is **largely delivered** and drops out of the critical path; what remains of it
is re-validation at 0.26. That is reflected in P0.

---

## Validation tiers — where a gate actually runs

Every exit gate below names the tier it must reach. "Compiles and executes" is not a claim until the
tier is stated, because these four things prove genuinely different properties. Vocabulary and flags
follow `docs/model-dev/cpu-development.md` and `docs/model-dev/nki_cpu_simulator.md`.

| Tier | How | Proves | Does **not** prove |
|---|---|---|---|
| **T0 — CPU mode, sim off** (laptop/macOS included) | default; NKI kernels take PyTorch fallbacks | math correctness on tiny tensors; config, routing, lifecycle logic | anything about kernels or Neuron |
| **T1 — CPU mode + NKI simulator** (Linux) | `VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1` (`NKI_PRECISE_FP=1` auto-set) | NKI kernel functional correctness | performance, graph capture, device behaviour |
| **T2 — CPU compilation** (Linux) | `VLLM_NEURON_CPU_COMPILE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2` | graph capture, NEFF builds, bucket/shape validity, compile time | **runtime behaviour — it never executes** |
| **T3 — on-device Trn2** | real hardware | NEFF load, execution, numerics vs reference, latency, peak memory | — |

Two constraints from the docs, both load-bearing:
- `VLLM_NEURON_CPU_COMPILE` and `VLLM_NEURON_CPU_MODE` **cannot be enabled together** — T1 and T2 are
  separate runs, never one invocation.
- The simulator is **slow** and suited to single functions/layers with tiny configs (<10M params);
  always set a timeout. It is not a path to whole-model validation.

**T2 is not a substitute for T3.** A graph that compiles can still produce wrong numbers, fail to
load, or run too slowly to matter.

### Where these tiers can actually run

The development machine this plan was written on is **macOS with no Neuron hardware, and currently no
`vllm`, `torch`, or `pytest` installed**. That is not a footnote — it decides which phases can be
worked locally at all:

| Tier | Available locally? | Needs |
|---|---|---|
| **T0-bare** — dependency-free Python (guards, registry, config normalization, bound arithmetic) | **yes** | pytest only |
| **T0-full** — CPU mode with torch + vllm | **no** | Linux + `vllm==0.26.0` + torch install |
| **T1** simulator | **no** | Linux |
| **T2** CPU compilation | **no** | Linux + Neuron compiler |
| **T3** on-device | **no** | Trn2 instance |

Consequence: **P0, P1, P2, P6, P7b, P8, and P9 cannot be completed on the development machine.** They
need a Linux CI/dev box for T0-full/T1/T2 and a Trn2 instance for T3. Work that *is* local is the
dependency-free tier — which the existing `test/unit/conftest.py` was already built to serve, loading
vllm-free modules straight from their file paths.

**Design implication, not just a logistics note:** components on the critical path should be written
dependency-free wherever the physics allows. Config normalization, layer selection, the P5 bound
arithmetic, and cache-index bookkeeping have no intrinsic need for torch, and keeping them importable
on a bare interpreter is what makes them testable before hardware is available. This is already the
established pattern in `patches/guards.py` and `scheduler_selection.py`; extend it rather than
inventing a second convention.

### Artifact requirements for every T2/T3 gate

A gate is not passed by a green terminal. Each T2/T3 run records, checked into the repo or a linked
run store: compiler version and Neuron SDK version; `NEURON_PLATFORM_TARGET_OVERRIDE` / instance
type; the **bucket configuration** exercised; NEFF size and count; compile wall time; full compile and
runtime logs; the numerical output plus its comparison against the named oracle; and host and device
**peak memory**. Remote Trainium time is scarce — a run nobody can reproduce or diff against is a
run that has to be repeated.

---

## P0 — vLLM 0.26 runtime foundation (development gate)

**Why first:** DeepSeek code must not be written against 0.21 interfaces. The 0.26 tree is also where
the engine-side machinery for this model already lives (`MLAAttentionSpec.compress_ratio`,
`HiddenStateCacheSpec`, `RSWASpec`, `Platform.register_custom_kv_cache_specs()`) — P1 is not
implementable on 0.21 at all.

Scope is deliberately narrow: **only what is required to construct and execute a model.** This is not
v3's Workstream A in full.

### P0.1 Dependency and version bump
- `requirements/core.txt:6` → `vllm==0.26.0`; `pyproject.toml:10` → `0.26.0.1.0.0`; branch
  `release-0.26.0.1.0.0`.
- Transformers: **keep the `<6.0.0` upper bound** and raise the floor to the 0.24+ requirement —
  `transformers>=5.5.3,<6.0.0` in both `requirements/core.txt:12` and `requirements/test.txt:3`.
  Then **narrow it to the range actually tested** (e.g. `>=5.5.3,<5.7`) once P0's smoke tests
  identify the versions exercised. An unbounded-within-major range is a silent-drift vector of
  exactly the kind the patch tripwires exist to close.
- Lock the fully resolved dependency set **before** the bump (v3 §A0) — the only way to attribute a
  later regression to the pin rather than to a transitive package.
- Capture **pre-upgrade baseline logits** for GPT-OSS and Qwen3-VL. A3.3-class drift is silent; this
  baseline must exist before the pin moves or it cannot be created afterwards.

### P0.2 Re-validate the existing patch registry at 0.26
The hard part is done. What is left:
- Bump `guards.py:39 VALIDATED_VLLM_VERSION` to `0.26.0` **only after** every tripwire passes against
  the real 0.26 tree — the constant is the claim, not a formality.
- Run `test/vllm_neuron/test_upstream_compat.py` against 0.26 and extend it where a startup guard can
  only prove a necessary condition (the file already documents which those are).
- Finish migrating any remaining scattered patch sites in `platform.py`, `neuron_parallel_state.py`,
  `neuron_worker.py` into a registered `Phase`. Keep the phase separation — `apply_port_hold_patch()`
  must stay at import time for spawn-mode survival, all2all registration must stay inside
  `check_and_update_config`.
- Confirm the `in_the_same_node_as` deduplication landed (`node_topology.py` exists; verify
  `neuron_worker.py:486-490` and `neuron_parallel_state.py:818-822` no longer carry duplicate bodies).

### P0.3 The two real breaks that block model execution
1. **`InputBatch` construction** (`neuron_model_runner.py:7658`). Upstream dropped `pin_memory` (now
   module-level `PIN_MEMORY`), made `max_num_blocks_per_req` required, added `slot_mapping_modes`.
   Add construction tests for one cache group **and for a synthetic heterogeneous multi-group
   config** — the latter is what P1 depends on.
2. **Re-sync the two line-for-line upstream ports.** `GPUModelRunner` moved ~1900/7900 lines and
   `Scheduler` ~1050/2900 between the tags; both drift silently.
   - `NeuronModelRunner._update_states` (`neuron_model_runner.py:1815-2036`) — re-diff field by
     field, especially `scheduler_output.scheduled_cached_reqs.*` and `CachedRequestState`.
   - `NeuronAsyncScheduler._update_after_schedule` (`core/scheduler.py:943-1023`).
   - Comment each port with the pinned upstream revision and the intentional Neuron deltas.

### P0.4 Deferred from the development gate — and what that costs
- **The NIXL connector rewrite** (v3 §A3.1 — the pull/push split, ~15 private parent attributes).
  Large, and DeepSeek bring-up does not need it. **Disable DI and fail loudly** at configuration time
  rather than silently degrading.
- v3 findings 4 and 6 (deployment-only).

### P0.5 Two distinct gates — do not conflate them

**Development gate (unblocks P1).** Deps resolve; imports smoke-test; every tripwire passes at 0.26
and `VALIDATED_VLLM_VERSION` is bumped; `InputBatch` and scheduler tests pass; GPT-OSS and Qwen3-VL
serve at **T3** with acceptable logit drift against the P0.1 baseline. NIXL deferred, DI rejected at
config time. No DeepSeek code exists yet.

**Release gate (publishing `0.26.0.1.0.0` as a normal replacement for 0.21).** Everything above,
**plus NIXL restored and 1P1D passing** — or the artifact is published as explicitly
**experimental/pre-release** with DI marked unsupported.

Shipping a normal-versioned 0.26.0.1.0.0 with known-broken supported-model DI, documented only in
release notes, is not acceptable: it reads as a drop-in successor to 0.21 and silently regresses
existing deployments. Deferring NIXL is a development decision; it must not become a release
decision by default. **The critical path below runs on the development gate only.**

---

## P1 — heterogeneous cache infrastructure

**Why here:** the first DeepSeek-specific priority and the hardest *systems* risk, and it is fully
validatable **without DeepSeek weights**. Getting it wrong late means re-plumbing the allocator, the
binding, the connector registration, and every lifecycle path at once.

### P1.1 Register the 0.26 cache specs
Wire `Platform.register_custom_kv_cache_specs()` and `_align_heterogeneous_kv_block_size()` in
`vllm_neuron/vllm/platform.py`, registering: the **latent MLA cache** in its real single-tensor
layout; the uncompressed SWA cache; the **c4 and c128 compressed caches** with their alignment; and
the compressor carry-state cache. (The indexer cache is F-phase; not now.)

### P1.2 Allocate a real single-tensor latent cache
**Do not fake the latent as a dummy K/V pair.** The allocator currently hands out
`(2, blocks, kv_heads, block_size, head_size)` at `neuron_model_runner.py:7731-7741`. Storing a
latent in `k` with `v` unused wastes half the cache *and* desynchronizes byte accounting from
allocation shape, binding, and connector registration — three places that then disagree under memory
pressure. `model/kv_cache.py` `LayerSpec` can already express `num_kv_heads=1, head_size=512`; the
spec is fine, the pair layout is not.

### P1.3 c4/c128 logical-block layouts
Reproduce the pinned 0.26 logical-block semantics unless a Neuron constraint forces a **documented**
deviation: one logical block size measured in *native token positions*, with c4 and c128 layers
storing different numbers of compressed entries per logical block.

### P1.4 Compressor carry state with upstream SWA-style lifecycle
Upstream models compressor residuals with **sliding-window cache semantics**. Follow that. Do **not**
preselect `HiddenStateCacheSpec` because the name fits, and do **not** invent a runner-local dict
keyed by request id — carry state must participate in the *same block lifecycle* as its corresponding
compressed cache, or it desynchronizes on exactly the paths (reorder, abort, preemption) that are
hardest to test.

### P1.5 Bind heterogeneous caches to layers
Teach `NeuronModelRunner.initialize_kv_cache` (`:7625`, currently raising at `:7767`/`:7778` for
anything but `FullAttentionSpec`/`SlidingWindowSpec`) to handle heterogeneous groups and per-layer
specs. Define explicitly how groups with differing page sizes, alignments, and layouts are allocated.

### P1.6 Lifecycle matrix, and the scope decision on prefix caching

**Decision: for the critical path, prefix caching is rejected, and P1 tests the rejection.**
Compressor carry state has no proven reuse semantics yet, and inventing them here would put
unvalidated state-sharing underneath every later correctness result. Successful prefix *reuse* moves
to the post-P9 backlog, where it lands only after carry-state semantics are proven.

Required lifecycle coverage — all at **T0**, on the synthetic model:

| Path | P1 requirement |
|---|---|
| New-request allocation | must work |
| Chunked-prefill continuation | must work |
| Decode update | must work |
| Batch reorder | must work |
| Batch compaction | must work |
| Completion | must work |
| Abort | must work |
| Block remapping | must work |
| Prefix caching | **rejected at startup, and that rejection is tested** |
| Disaggregated transfer | **rejected at config time** (P0.4) |
| Speculative decoding | **rejected at config time** |
| Fork/copy | rejected unless free |

Every rejection is an explicit error, never a silent fallback.

### P1.7 Synthetic gate
Extend `vllm_neuron/model/synthetic/` (already env-gated via `VLLM_NEURON_SYNTHETIC_MODEL`) to a
model exercising SWA + latent MLA + compressor-state caches with **no DeepSeek weights**.

**Exit gate — P2 does not start until this passes.** The full "must work" column above, at **T0**.
Discrete results (block ids, slot mappings, carry-state positions) compared **exactly** — no float
tolerances on cache bookkeeping.

---

## P2 — 512-d MLA feasibility spike

**Why before any model code:** the largest schedule risk in the project. If `head_dim=512` cannot
compile, or compiles unusably slowly, that must be discovered before mHC, MoE, and the loader are
written. `attention_cte.py:16` caps `MAX_HEAD_DIM=128` but *falls back to torch*;
`attention_segmented_cte.py:36` **raises** above 128. So this needs a DeepSeek-specific PyTorch
reference path, an early NKI kernel, or both.

Build the **smallest correct BF16 operation** covering all of: MLA projections and latent MQA against
the single 512-d latent; partial/inverse RoPE; per-head attention sinks; SWA composed with compressed
history; prefill **and** paged decode reads; and fixed-shape metadata across representative static
compilation buckets.

Cache layouts must be P1's, not a spike-local shortcut — otherwise the spike proves nothing about the
real path.

**Start the NKI prototype here**, at T1 against the simulator
(`docs/model-dev/nki_cpu_simulator.md`), rather than deferring all 512-d kernel work to the
long-context phase.

### P2 exit gate — staged by tier, and it ends on hardware

| Stage | Tier | Must show |
|---|---|---|
| P2.a | **T0** | MLA math correct on tiny tensors vs a float32 reference |
| P2.b | **T1** | the NKI kernel functionally correct under the simulator, tiny shapes, with timeout |
| P2.c | **T2** | graph captures and NEFFs build at `head_dim=512` for representative prefill **and** decode buckets; compile time and graph size recorded |
| P2.d | **T3** | **NEFF loads and executes on Trn2**; numerics compared against the P2.a reference; per-bucket latency and peak memory measured |

**P2 does not pass at T2.** CPU compilation cannot validate runtime behaviour, and "it compiled" is
precisely the false positive that would let the largest schedule risk survive undetected into P3.
Full artifact set (above) required for P2.c and P2.d.

**Pre-authorized re-ordering:** if 512-d cannot compile or is prohibitively slow, the NKI kernel takes
priority over all further model assembly, and P3 waits.

---

## P3 — tiny DeepSeek model: vertical first, then every layer variant

**Why this shape:** implementing every subsystem horizontally before anything runs makes the first
integration bug arrive with six new subsystems as suspects.

But one layer **cannot** exercise SWA, c4, and c128 attention plus hash and routed MoE
simultaneously — those are mutually exclusive per-layer choices. So P3 splits in two: a vertical
slice for fast integration, then a multi-layer model for structural coverage. Both use a tiny
synthetic configuration with the same structural features and dramatically fewer experts and
layers, following `docs/model-dev/onboarding-models.md:864-915` (random model, `enforce_eager=True`,
`num_gpu_blocks_override=4`, `hidden_size % 256 == 0`).

### P3a — one vertical layer, one token
**Exactly one c4 + routed-MoE layer**, end to end:
1. **Config normalization** — `compress_ratios` *or* normalized `layer_types` + `compress_rates`,
   depending on the pinned Transformers version. One internal per-layer representation; test the two
   forms are equivalent; **fail loudly on an unrecognized form**.
2. **Weight-name and shape mapping.**
3. **mHC** — 4× residual widening, 20 Sinkhorn iterations. Plain decomposed PyTorch; upstream's
   tilelang fusion is a performance concern, not a correctness one.
4. **Compressor** — strided reduction with learned `ape` weights and cross-chunk carry state, on
   P1's cache.
5. **MLA/SWA** — on P2's kernel or reference path.
6. **One shared expert and a few routed experts** — enough to exercise routing, not 256.
7. **Decoder layer.**
8. **LM head and a one-token decode.**

*Gate: one decoded token at **T0**.*

### P3b — tiny multi-layer model, all structural variants
Extend to a model containing at minimum: an **SWA + hash-MoE** layer (ratio 0, layers 0–2 semantics),
a **c4 + routed-MoE** layer, and a **c128 + routed-MoE** layer. This is what makes P1's heterogeneous
cache real — until three different cache layouts coexist in one model, the heterogeneity is
hypothetical.

*Gate: multi-layer forward and decode at **T0**; heterogeneous cache binding at **T2**. Every
structural variant is exercised before P4 begins.*

### Structure
Files land in `vllm_neuron/model/deepseek_v4/`: `config.py`, `factory.py`, `model.py`, `attention.py`,
`compressor.py`, `mhc.py`, `moe.py`, `weight_loaders.py`, `__init__.py`.

**One implementation with per-layer component selection.** Copy llama3's
`resolve_attention_mlp_classes` (`model/llama3/quantization.py:236-309`) — it picks attention/MLP
classes per layer, exactly what V4 needs. **Do not** copy gpt_oss's whole-model fork per quantization
(`model_bf16.py` / `model_mxfp4.py` are near-duplicate 2200-line files).

**Register `DeepseekV4ForCausalLM` in `model/registry.py:19-24` only after** P3b construction and
forward tests pass.

---

## P4 — component correctness against a named oracle

### The oracle problem — settle it before writing assertions
"Compare against the pinned vLLM reference" is not executable guidance: `vllm/models/deepseek_v4/`
contains CUDA and tilelang paths that will not run locally at all. **Each component names its oracle
explicitly**, and the choice is recorded in the test:

| Component | Oracle | Why |
|---|---|---|
| Config normalization | Transformers config loader | it is the thing being normalized |
| mHC / Sinkhorn | **extracted pure-PyTorch** logic from the reference | upstream fuses into tilelang; the math is small and portable |
| Compressor + carry state | **extracted pure-PyTorch** reference | portable; carry state is pure bookkeeping |
| MLA prefill / decode | pure-PyTorch reference at fp32, plus a **GPU reference capture** for the fused path | the fused kernel cannot be reproduced locally |
| MoE routing (ids, weights) | **extracted pure-PyTorch** — discrete, must match exactly | selection logic is portable and exactness matters most here |
| Hash MoE `tid2eid` | Transformers / checkpoint tables directly | it is a lookup, not a computation |
| Full-layer / logits | **GPU reference capture**, pinned and stored | no local path reproduces the whole layer |

Where the oracle is a **GPU reference capture**, it is generated once, stored with the checkpoint
revision and the vLLM revision that produced it, and treated as a fixture — not regenerated per run.

### Order
1. **mHC / Sinkhorn.**
2. **Compressor outputs and carry state across *different chunk boundaries*** — the highest-value
   test in this phase; chunk-boundary state is where the design either holds or doesn't.
3. **MLA prefill.**
4. **MLA decode against the paged cache.**
5. **Hash-MoE** (layers 0–2): selected expert ids from `tid2eid[input_ids]`, *and* the learned gate
   weights applied to them. Token-ID alignment must survive padding and TP/SP/EP.
6. **Routed MoE**: `sqrtsoftplus` scoring, `noaux_tc` with `e_score_correction_bias` affecting
   **selection only**, `routed_scaling_factor=1.5`. Note `functional/moe/router.py:75,568-599`
   applies `router_bias` pre-activation, so it also changes the gate value — wrong for `noaux_tc`.
7. **Shared expert in *both* prefill and decode** — currently wired for decode only, and absent from
   the torch fallback, which breaks CPU parity (`moe_block_tkg.py:361-363`).

### Comparison discipline
Discrete results — expert ids, cache positions, block mappings, layer types, sparse indices — match
**exactly**. Floating-point tolerances apply to float tensors only; a blanket `rtol=0.01` cannot
validate routing or cache correctness and will hide the bugs this phase exists to find. Use
`assert_close_three_way(target=neuron, expected=hf_fp32, baseline=hf_bf16)`
(`accuracy/testing.py:289`), and tensor capture/replacement (`accuracy/tensor_capture.py`,
`accuracy/tensor_replacement.py`) to isolate the first diverging component.

**Constraint inherited from P5 not yet existing:** every P4 test runs at a context length where the
eligible compressed-entry count is **asserted in the test** to be ≤ `index_topk`. Not assumed —
asserted. Until P5 derives the real bound, that assertion is what keeps a P4 mismatch from being
caused by wrongly skipping the indexer rather than by the component under test.

**Exit gate:** every component matches its named oracle, at **T0**, with the ≤ `index_topk` assertion
active in each test.

---

## P5 — dense-CSA safety bound (before any end-to-end logit claim)

**Why it precedes end-to-end comparison:** the indexer **only selects; it never weights**, so wherever
the eligible compressed set is no larger than `index_topk`, CSA is *exactly* dense attention and the
indexer can be skipped. Every logit result in this plan depends on that equivalence holding. If the
bound is unknown, an end-to-end mismatch is ambiguous between "the model is wrong" and "we skipped the
indexer outside its valid range" — so the derivation comes first. It needs only the pinned
implementation, not the full model, so there is no reason to defer it.

Derive the exact equivalence condition from the pinned compressor/indexer implementation, accounting
for: compression kernel width and stride; initial offset and **incomplete carry state**; sink or
reserved entries; the local SWA candidate set; and **generated tokens as well as prompt tokens**.

### The admission check must use maximum possible total length
Checking current sequence length at admission is insufficient — a request admitted inside the bound
can generate its way across it mid-decode, at which point the dense path silently becomes wrong.
Admit on:

```
prompt_tokens + max_requested_output_tokens  ≤  derived_bound(layer_type, compressor_state)
```

with the effective output budget taken from whichever of `max_tokens` / `max_model_len` / server
default actually caps the request. If the ceiling is unbounded, the request is rejected.

Express the bound as a checked function of sequence length, layer type, and compressor state. Test
immediately below, at, and above every boundary, repeated across prefill chunk sizes — and include a
test that a request admitted just inside the bound with a large output budget is **rejected**.

A naive reading suggests ~2048 tokens for Flash (`index_topk=512` × the c4 ratio). **Do not promise
that number** until the derivation and boundary tests prove it.

**Exit gate:** bound derived and documented; runtime guard enforces the max-total-length form;
boundary tests pass on both sides of every edge, at **T0**.

---

## P6 — end-to-end chunked logit correctness (tiny model)

Now that the valid dense-CSA range is known and enforced, end-to-end comparison is meaningful.

- Multiple chunkings of one prompt produce matching logits.
- Prefill-then-decode matches an unchunked reference.
- Batch reorder / completion / abort leak no compressor state.
- TP/SP/EP variants preserve routing **and** logits.
- All of it inside the P5 bound, with the guard active.

Oracle: the stored GPU reference capture from P4. End-to-end comparison uses
`multi_prompt_logit_validation` (`accuracy/logit_validation.py`) across a TP/seqlen/batch grid.

**Exit gate:** the above at **T0**, and the tiny multi-layer model executing at **T3** on Trn2 with
logits compared against the T0 result. Full artifact set for the T3 run.

---

## P7 — memory gate (before touching the 284B model)

Trn2 is 96 GB/chip × 16 = 1,536 GB aggregate. **Flash at BF16 is ~568 GB of weights in total** —
~284B parameters × 2 bytes — of which the ~277B expert parameters are ~554 GB. Quote the total, not
the expert payload; and note that **aggregate HBM is not evidence of feasibility** in either case.

This phase splits, because **no development machine can measure this**. A laptop cannot host
DeepSeek-V4 in any precision — Flash or Pro — so peak memory is *predicted* locally and *measured*
only on hardware. Do not let the prediction stand in for the measurement.

### P7a — analytical per-rank accounting model (local, T0-bare)
A dependency-free calculator, unit-testable without torch, that takes checkpoint tensor metadata plus
a parallelism configuration and returns a per-rank budget covering: all parameters (not just
experts); quantized source *and* BF16 destination tensors held simultaneously during conversion;
sharding and alignment imbalance; loader and page-cache duplication; compiler and graph memory;
activations and collective buffers; and minimum useful KV + compressor capacity.

Drive it from **real checkpoint tensor metadata** — safetensors headers give dtype and shape without
downloading or materializing weights, so the parameter accounting is exact even locally. Everything
else (compiler arena, activation peak, allocator fragmentation) is a **modelled estimate with a
stated error bar**, and must be labelled as such in the output.

Prototype **streaming shard-by-shard dequantization directly into final sharded tensors** — precedent
at `gpt_oss/weight_loaders_bf16.py:264 _dequantize_mxfp4_to_bf16` — validated at small scale on a
synthetic checkpoint. Note `expert_parallel_interleaved_loader` (`utils/weight_loader.py:676,727`)
was written against DeepSeek checkpoint layouts (bf16 stride 2, fp8 stride 4) and is directly
reusable.

*Gate: the model reproduces measured peaks for an already-supported model (GPT-OSS) within its stated
error bar. A calculator that has never been checked against a real measurement predicts nothing.*

### P7b — measured peak on Trn2 (T3, hardware required)
Host and device peak measured on the target instance, with the full artifact set, on the loading path
that will actually be used. Compare against P7a and **record the delta** — that delta is what
calibrates the model for V4-Pro later, where the same question recurs at larger scale.

### Decision point — requires P7b, not P7a
- **BF16 has documented headroom** (measured) → P8 (M1), then P9 (M2).
- **It does not** → **P9 runs first**, and P8's full-model validation happens directly on the
  quantized path. BF16 is retained for per-module validation only.
- **P7a alone predicts "no headroom"** → start P9 immediately rather than waiting for hardware; a
  predicted failure is sufficient to reorder work, even though it is not sufficient to declare
  success. The asymmetry is deliberate.

---

## P8 — M1: full DeepSeek-V4-Flash, BF16, short context

**This is bring-up, not completion.** It produces a working short-context model on a
non-final loading path.

Scale from the tiny model to the real checkpoint **progressively**, never in one jump: real
checkpoint tensors for one component → one real decoder layer → the P3b tiny multi-variant model →
selected real layers → full short-context prefill and decode. Avoid any loop requiring two full 284B
BF16 copies per iteration.

Ship alongside: `examples/vllm_neuron/models/deepseek_v4/` and a model README (template
`docs/model-dev/onboarding-models.md:675-712`), following the official 5-step onboarding process.

**Exit gate (M1), at T3 with full artifacts:** P6's correctness properties hold on the real
checkpoint; the P5 dense-CSA guard is enforced; TP/SP/EP variants preserve routing and logits;
measured memory stays within the P7 budget; unsupported combinations (prefix caching, MTP, DI) fail
at configuration time.

---

## P9 — M2: native FP8/FP4 quantization (mandatory)

**Not optional, and not "post-P8 backlog".** The completion criteria require native checkpoint
quantization with no load-time BF16 expansion, so M2 is on the path to done regardless of whether
BF16 happened to fit at P7. P7's decision only changes **when** P9 runs, never **whether**.

FP8 128×128 block-scaled dense weights with `ue8m0` scales; FP4 experts; any distinct indexer/cache
quantization added later. `QuantScheme` today has only `NONE` and `FP8_STATIC_PER_TENSOR`
(`llama3/quantization.py:39-51`).

**Do not assume the GPT-OSS MXFP4 dequantizer shares storage or scale semantics** — MX is
per-32-element microscaling, a different family with MX-specific tiled scale shapes. Validate
checkpoint metadata, packed layout, scale broadcasting, sharding boundaries, and tail blocks
independently. Compare each quantized component against its dequantized BF16 reference **before** any
end-to-end run.

**Exit gate (M2):** component-level accuracy vs BF16 reference; the full M1 gate re-run on the
quantized path at **T3**; a new measured per-rank memory budget showing the reduction.

---

## Backlog — after P9

Unchanged from v3; listed so nothing is lost:

| Item | v3 § | Note |
|---|---|---|
| CSA lightning indexer, token-level top-k, sparse metadata | F | unlocks context beyond P5's bound |
| Optimized 512-d MLA prefill/decode kernels | F | P2's prototype productionized |
| **NIXL connector rewrite** (pull/push split) | A3.1 | deferred from P0's *development* gate; **required by the release gate** (P0.5) |
| Prefix caching — successful reuse | G | moved here from P1; needs proven carry-state reuse semantics |
| MTP / speculative decoding with cache fork+rollback | G | |
| Perf tuning (mHC, compressor, MoE, attention) | G | |
| V4-Pro | G | config/sharding/capacity exercise once Flash parity holds |

---

## Test and CI

There is now a **partial** harness (`test/unit/`, `test/vllm_neuron/test_upstream_compat.py`,
uncommitted) where v3 recorded none. `ci/` still does not exist despite `pyproject.toml:46-50`
referencing it, and `.github/workflows/` holds only issue-labeling automation. Building it out
remains part of the work.

- **T0 (CPU/unit)** — config-form equivalence and validation; weight-name/shape mapping; mHC/Sinkhorn;
  compressor chunk boundaries and lifecycle; exact MoE routing; synthetic cache allocation and
  cleanup; prefix-caching **rejection**; the P5 bound derivation, its max-total-length admission form,
  and its boundaries; upstream API/signature tripwires.
- **T1 (simulator)** — NKI kernel correctness, single functions/layers, tiny configs, always with a
  timeout.
- **T2 (CPU compilation)** — tiny-model graph capture; MLA prefill/decode buckets at `head_dim=512`;
  heterogeneous cache binding; TP/SP/EP collective shapes.
- **T3 (Trn2)** — supported-model regressions post-upgrade; P2's MLA spike; tiny-model
  prefill/decode; full-model short-context correctness; memory measurement; long context once CSA
  lands.

**Store pinned baseline logits and report drift** — top-token agreement alone hides real regressions.
Add a **compat CI job** pinned to the next vLLM release so the 0.26 gap does not silently reopen;
vLLM ships roughly biweekly.

---

## Risks, ordered by when they bite

1. **512-d MLA kernel feasibility** — retired or exposed at **P2**, and only by the **T3** stage;
   a T2 pass is not a retirement. Pre-authorized: NKI takes priority over model assembly on failure.
2. **Compressor-state lifecycle correctness** — retired at **P1** via engine-managed state and the
   synthetic gate, before any weights exist.
3. **False correctness from short-context dense attention** — the specific danger of this staging.
   Retired at **P5**, which now precedes every end-to-end logit claim; P4 carries an explicit
   ≤ `index_topk` assertion in the interim. The admission guard must use prompt + max output tokens,
   or requests generate across the bound.
4. **BF16 peak memory** — gated at **P7**, which reorders P8/P9 rather than cancelling either.
5. **Static-graph shape explosion** — buckets and fixed-shape metadata defined at P2; track compile
   count and graph memory from there on, via the recorded artifacts.
6. **Private upstream coupling, especially NIXL** — deferred for development, **blocking for
   release**. Guarded adapters and a pinned revision when it returns.
7. **Neuron `torch-2.13` availability** gates vLLM 0.27 entirely and is outside this repo's control.
   0.21→0.26 needs no torch change (0.21 through 0.26 all pin `torch==2.11.0`; 0.27 jumps to 2.13,
   while the Neuron index tops out at `torch-xla 2.12` / `libtorch-neuronx-lite 2.12`). Confirm the
   torch-2.13 roadmap with AWS — the evidence here is the package index, not an AWS statement.

## Completion criteria

Unchanged from v3, and reachable only at **P9/M2**, not P8: DeepSeek-V4-Flash is done when the
documented configuration loads without peak-memory failure on target hardware; produces validated
prefill and decode logits; preserves compressor and KV state across all supported scheduler
operations; **uses native checkpoint quantization with no load-time BF16 expansion**; either
implements CSA for the advertised context or enforces a **tested** shorter limit; clearly rejects
unsupported prefix-caching / MTP / DI combinations; and ships reproducible pins, run examples, tests,
and benchmarks.
