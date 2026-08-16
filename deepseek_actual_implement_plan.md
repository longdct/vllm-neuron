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
| Version pins moved to vLLM 0.26 / `0.26.0.1.0.0` / `transformers>=5.5.3` | **P0.1** | **done** — pre-bump state preserved at commit `be0def6`; transformers ceiling since narrowed to `<5.16` (the version actually exercised) |
| Patch registry, phases, guards, tripwires | **P0.2** | **done** — all 4 tripwires execute clean against an installed **vLLM 0.26.0**; `VALIDATED_VLLM_VERSION` bumped to `0.26.0` and the gate is now automated |
| Bare-interpreter test harness | — | **built** — `test/unit/conftest.py` loads vllm-free modules by path |
| **`InputBatch` construction re-sync** (`vllm/worker/input_batch_params.py`) | **P0.3.1** | **done** — 10 tests incl. a real heterogeneous `InputBatch` (SWA + 512-d MLA + Full) |
| **Ported-surface guards** (`_update_states`, `_update_after_schedule`) | **P0.3.2** | **done** — both ports verified field-by-field against 0.26 and guarded by tests |
| **DI rejected loudly** (`vllm/di_support.py`) | **P0.4** | **done** — 9 tests; rejection fires on the import path *and* at config time |
| **Config normalization + validation** (`model/deepseek_v4/config.py`) | **P3a.1** | **done and oracle-verified** — 33 unit tests + 12 against the real `DeepseekV4Config` (Transformers 5.15.0). Three assumptions were wrong and are now corrected; see below |
| **Dense-CSA bound + admission guard** (`model/deepseek_v4/dense_csa.py`) | **P5** | **T0 done** — constants derived from Transformers 5.15 and max-total request admission wired |
| **Per-rank memory accounting model** (`model/memory_budget.py`) | **P7a** | **done** — 32 tests; needs GPT-OSS calibration to pass its gate |
| Encoder-decoder cache sizing risk (R1) | **P0.3** | **mitigated** — rejected before block-table construction until the runner grows `max_encoder_len` |
| Kernel/page block-size assumption (R2) | **P1 prerequisite** | **settled for the current backend** — QKV scatter and segmented gather address the page block directly; equality is guarded |
| Upstream heterogeneous lifecycle registration | **P1.1** | **done** — platform hook validates real MLA/c4/c128/SWA/hidden-state/R-SWA specs have upstream managers |
| Single-tensor latent MLA allocation | **P1.2** | **done in runner** — compressed physical shape uses `storage_block_size`; no dummy V tensor |
| Prefix/speculative feature guards | **P1.6** | **rejection done** — DeepSeek-V4 rejects both at configuration time; successful lifecycle semantics remain backlog |
| Real heterogeneous lifecycle matrix | **P1.6/P1.7** | **T0 done** — vLLM managers cover allocation, continuation, decode eviction, reorder/compaction identity, completion, abort, and remapping; synthetic model declares every layout |
| 512-d portable MLA reference | **P2.a** | **T0 done** — fp32 prefill/decode, partial/inverse RoPE, sinks, paged gathers, SWA/compressed composition, and representative buckets |
| 512-d NKI simulator prototype | **P2.b** | **T1 done** — causal prefill and decode execute through `nkilib`'s four-tile 512-d kernel and match the fp32 reference |
| 512-d NKI compilation | **P2.c** | **kernel sub-gate done; phase partial** — four prefill/decode buckets compile to NKI backend configs locally in 0.47–0.58 s; full graph capture/NEFF generation is blocked in this venv by missing `torch_xla`/`torch_neuronx` and a stale `neuronx-cc` launcher shebang |
| Checkpoint namespace and fused-shard contract | **P3a.2** | **done; loader integration remains** — official prefix/suffix/scale mappings, FP4-vs-FP8 expert scales, fused attention/compressor/MLP shard ids, and fail-fast shape validation are covered at T0 |
| Portable mHC/compressor/MoE primitives | **P3/P4** | **partial** — oracle-backed component math landed; decoder/model integration remains |
| Tiny structural CPU model | **P3a/P3b/P6-T0** | **T0 prototype done** — all independent attention/MLP variants, one-token decode, exact chunk invariance, and abort isolation; production loader/runner integration remains |
| Transformers component oracles | **P4** | **partial** — config, Sinkhorn, routed selection/weights, hash lookup, and complete c4/c128 compressors (carry, RMSNorm, and RoPE) match 5.15; fused MLA and full-layer fixtures remain |
| Streaming conversion into final shards | **P7a** | **prototype done** — one source tensor and converted tensor at a time with explicit temporary peak; native FP8/FP4 layouts remain P9 |
| Pinned compressor geometry and request admission | **P5** | **T0 done** — Transformers 5.15 complete-window emission proves c4's 2051-token bound; `NeuronPlatform.validate_request` enforces prompt plus `max_tokens` before scheduling |
| Everything else | P1–P9 | **not implemented**; T0–T2 work is locally actionable, while the named T3 gates remain hardware-blocked |

**P3a.1's three corrected assumptions.** The config normalizer had been written without a Transformers
5.x install to read; all three of its guesses were wrong, and all three passed its hand-built unit
tests. Recorded because it is the clearest available argument for oracle tests over fixture tests:

| Assumed | Actually |
|---|---|
| `layer_types` spelling `compressed_attention` | `compressed_sparse_attention` (c4) and `heavily_compressed_attention` (c128); `sliding_attention` was right |
| `compress_rates` is a per-layer list | a **dict keyed by layer type** — different length, different meaning. Read positionally it yields silently wrong ratios from layer 2 on |
| `num_hash_layers` selects hash-MoE layers | it is a **legacy kwarg upstream consumes** in `__post_init__`; `mlp_layer_types` is the live field. Same for `compress_ratios` |

The strict-raise design held: the vocabulary error would have failed loudly at startup rather than
mis-typing layers. The `compress_rates` shape error would **not** have — it was the dangerous one.

Also established, and worth carrying into P3b: **attention and MLP structure are independent lists.**
In the default V4 config, layers 0–2 are `hash_moe` *and* `heavily_compressed_attention` — so hash-MoE
layers are **not** the sliding-window layers, contrary to this plan's earlier "SWA + hash-MoE (ratio 0,
layers 0–2)" phrasing in P3b. A test pins the independence so it cannot be re-derived by inference.

Suite: **302 passed, 6 skipped**, plus **2 explicit T1 simulator tests** — the bare tier remains dependency-free; the component tier uses
torch, and the `test/vllm_neuron/` tier runs against a real vLLM 0.26.0 and a real
`DeepseekV4Config`. The `InputBatch` tests provide an explicit CPU `DeviceConfig`, so they no longer
depend on `VLLM_NEURON_CPU_MODE` merely to construct the fixture. New modules are mutation-checked —
deliberately breaking the bound formula, the admission rule, the streaming-peak formula, the
encoder-only skip, and the per-group block-count derivation each produced failures.

**P0's development gate is met except for its T3 clause.** Deps resolve, imports smoke-test, every
tripwire passes at 0.26, `InputBatch` and the scheduler ports are re-synced, and DI is rejected at
configuration time. What is *not* met, and cannot be here, is "GPT-OSS and Qwen3-VL serve at **T3**
with acceptable logit drift" — that needs Trn2. P1 is unblocked on everything except that
regression evidence.

**Why P5 and P7a landed before P1–P2.** They were implemented while the original development
environment lacked the dependencies required by P1/P2. That blocker has since been removed: the
current Linux/Python 3.12 environment supports T0–T2, so P1, P2.a–P2.c, and the local portions of
P3–P5 are now actionable in critical-path order. The pinned Transformers implementation has since
settled P5's emission constants, and runtime admission is wired through the platform request hook. P7a still needs calibration
against a measured GPT-OSS peak, and P3a still needs weight mapping, decoder/model integration, and
one-token execution despite its portable component primitives now existing.

**What the two new modules deliberately refuse to do.** `dense_csa.CompressorGeometry` has no default
kernel width, stride, or offset. `geometry_from_config` is the sole convenience constructor and reads
the pinned Transformers compression-rate mapping; the default c4 complete-window bound is 2051
tokens, not an assumed 2048. `memory_budget` marks
every line `EXACT` or `ESTIMATED`, and `MemoryBudget` exposes no scalar total, so a modelled range can
never be quoted as a measurement; `fits_in()` returns `None` when the range straddles capacity, which
is the signal to run P7b rather than to guess.

## Ground truth as of this plan

Verified against the working tree, not assumed:

| Fact | Evidence |
|---|---|
| ~~Pin is `vllm==0.21.0`~~ → now `vllm==0.26.0`, package `0.26.0.1.0.0` | `requirements/core.txt:6`, `pyproject.toml:10` |
| Patch registry **already built** — phases, guards, tripwires | `vllm_neuron/vllm/patches/{registry,guards,tripwires,node_topology}.py` |
| ~~Guards validated against 0.21~~ → **validated against 0.26.0**, tripwires executed | `patches/guards.py` `VALIDATED_VLLM_VERSION = "0.26.0"` |
| Test harness bootstrapped | `test/unit/…`, `test/vllm_neuron/…` |
| ~~`InputBatch` constructed in one place~~ → derivation extracted | `vllm/worker/input_batch_params.py` |
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

**This table was re-measured, and the earlier version of it was wrong.** The plan was originally
written against a macOS box with no `vllm`, `torch`, or `pytest`. The current development machine is
**Linux (Debian, Python 3.13) with `torch==2.11.0+cu130`, `vllm==0.26.0` and `transformers==5.15.0`
already installed in `.venv`**, no CUDA device and no Neuron hardware. That difference is what
unblocked P0.2–P0.4.

The environment has since been rebuilt on **Python 3.12.13** (`.venv`, with the previous 3.13 tree
preserved at `.venv313-backup`) specifically to clear the `neuronx-cc` ABI gap. Current state:
`torch 2.11.0`, `vllm 0.26.0`, `transformers 5.15.0`, `nki 0.5.0`, `neuronx-cc 2.26.6360.0`.

| Tier | Available locally? | Evidence / blocker |
|---|---|---|
| **T0-bare** — dependency-free Python | **yes** | pytest only |
| **T0-full** — CPU mode with torch + vllm | **yes** | `vllm_neuron.functional` and `NeuronModelRunner` both import (`NEURON_PLATFORM_TARGET_OVERRIDE=trn2` required) |
| **T1** simulator | **yes** | `nki.simulator.simulate_kernel` imports |
| **T2** CPU compilation | **partial and actionable** | the Python compiler package builds NKI backend configs, but `neuronx-cc` is not a shell executable on PATH; full graph/NEFF capture is still unproven |
| **T3** on-device | **no** | Trn2 instance — the only remaining hardware gate |

Four practical notes, each of which cost time to discover:

- **`nki` must come from the Neuron index.** PyPI's `nki` is a placeholder whose `__init__` raises
  `ImportError("WRONG PACKAGE...")`. Install with
  `--index-url https://pip.repos.neuron.amazonaws.com --index-strategy unsafe-best-match`. Passing
  `--extra-index-url` silently resolves the stub instead.
- **`nkilib` ships inside `neuronx-cc`** (884 entries in the wheel), not as its own package — and the
  published `vllm-neuron` declares neither `nki` nor `nkilib`, so this is invisible from the
  requirements files. `neuronx-cc` publishes `cp310`/`cp311`/`cp312` wheels only, which is the whole
  reason for the 3.12 pin.
- **`neuronx-cc` downgrades `networkx` to 2.8.8.** Noted in case a later dependency needs 3.x.
- **A CUDA box is a *third* environment.** P4's oracle table names "GPU reference capture" for the
  fused MLA path and full-layer logits. The dev machine has an RTX 3080 Ti (12 GB, sm_86) — ample for
  P4's tiny configs, but the installed cu130 torch cannot drive driver 550, and upstream's
  tilelang/fused paths may require Hopper. Pure-PyTorch extracted oracles run locally; the fused
  capture may not.

Consequence, revised again: **only T3 is remote.** P1, P2.a–P2.c, P3, P4 (bar the fused-path
capture), P5 and P6's T0 clause are all local work. Still needing Trn2: **P2.d, P6's T3 clause, P7b,
P8, P9**, plus the deferred shipped-model regression. Critically, **P2.c — whether `head_dim=512`
compiles at all, the project's largest schedule risk — no longer needs hardware.**

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

### P0.2 Re-validate the existing patch registry at 0.26 — **done**
- ✅ `VALIDATED_VLLM_VERSION` bumped to `0.26.0`, after all 4 tripwires executed clean against the
  installed 0.26 tree. The gate is now **automated**, not a manual ritual:
  `test_every_registered_tripwire_passes_against_installed_vllm` runs the real tripwire bodies, and
  `test_validated_version_matches_installed` fails the moment vLLM drifts past the validated version.
- ✅ `test/vllm_neuron/test_upstream_compat.py` passes against 0.26. One genuine break was found and
  fixed **in the test, not the plugin**: 0.26 made `SchedulerConfig` a pydantic model requiring
  `max_model_len` and `is_encoder_decoder`. The plugin's actual assumption — that upstream still
  resolves its default scheduler to the two paths in `UPSTREAM_DEFAULT_SCHEDULER_PATHS` — **holds**.
- Finish migrating any remaining scattered patch sites in `platform.py`, `neuron_parallel_state.py`,
  `neuron_worker.py` into a registered `Phase`. Keep the phase separation — `apply_port_hold_patch()`
  must stay at import time for spawn-mode survival, all2all registration must stay inside
  `check_and_update_config`.
- Confirm the `in_the_same_node_as` deduplication landed (`node_topology.py` exists; verify
  `neuron_worker.py:486-490` and `neuron_parallel_state.py:818-822` no longer carry duplicate bodies).

### P0.3 The two real breaks that block model execution — **done**
1. ✅ **`InputBatch` construction.** All three predicted breaks were real: `pin_memory` removed,
   `max_num_blocks_per_req` now required, `slot_mapping_modes` added. The per-group derivation now
   lives in **`vllm_neuron/vllm/worker/input_batch_params.py`**, deliberately split out of the runner:
   it needs only vLLM's cache interfaces, whereas importing the runner drags in
   `vllm_neuron.functional → nki → nkilib`, which is unavailable without the Neuron SDK. Splitting it
   is what makes the heterogeneous case testable *now* rather than on hardware.

   Tested for one cache group and for a heterogeneous multi-group config — and the tests construct a
   **real `InputBatch`** (SWA @16 + latent MLA @32 `head_size=512` + Full @64), because agreeing with
   our own helper proves nothing about whether upstream accepts the result. Also covers c4/c128
   `compress_ratio` groups coexisting, encoder-only groups being skipped without shifting later
   indices, and the positional-alignment invariant across all four lists.

2. ✅ **Both ports re-synced.** Verified field by field against 0.26: `CachedRequestData` still
   supplies `req_ids` / `resumed_req_ids` / `new_block_ids` / `num_computed_tokens` /
   `num_output_tokens`; `CachedRequestState` still accepts every keyword the runner passes, plus
   `prev_num_draft_len`; `Request` still carries `is_prefill_chunk`, `num_output_placeholders`,
   `spec_token_ids`, `num_computed_tokens`; `SchedulerOutput` still carries
   `pending_structured_output_tokens`. **Neither port needed a code change.**

   Rather than leave that as a one-off diff, `TestPortedUpstreamSurfaces` now asserts each surface,
   including that `AsyncScheduler` still overrides `_update_after_schedule` — if upstream drops that
   override, the Neuron port's deliberate bypass of it silently becomes meaningless.

### P0.4 Deferred from the development gate — and what that costs — **rejection done**
- **The NIXL connector rewrite** (v3 §A3.1) is deferred, and the §A3.1 diagnosis is **confirmed by
  execution**, not inference: 0.26 exports `NixlPullConnector` / `NixlPushConnector` with separate
  scheduler and worker classes, and `NixlConnectorWorker` — the class `NeuronNixlConnector`
  subclasses — is gone. The module raises `ImportError` on import at 0.26.
- ✅ **DI now fails loudly**, via `vllm_neuron/vllm/di_support.py`. Two entry points, because a
  config-time-only guard would have missed the realistic one:
  - **Import time.** With `kv_connector_module_path` set — the documented invocation — vLLM imports
    the connector during `VllmConfig` *construction*, before any platform hook runs. The connector
    now catches that `ImportError` and re-raises `UnsupportedDIConfigError` naming the missing symbol,
    the version that removed it, and how to run without it, with the original `ImportError` chained
    for whoever restores the port.
  - **Config time.** `check_and_update_config` rejects unsupported connectors before the early
    `model_config is None` return, so every configuration path is covered.
- **Scoped deliberately to `NeuronNixlConnector`.** The Neuron decode-bench connector subclasses
  upstream's `decode_bench_connector`, which 0.26 left intact, and still imports cleanly — blanket-
  disabling DI would have removed a working feature on the strength of an unrelated breakage. A test
  pins that scope.
- v3 findings 4 and 6 (deployment-only).

### P0.5 Two distinct gates — do not conflate them

**Development gate (unblocks P1).** Deps resolve ✅; imports smoke-test ✅; every tripwire passes at
0.26 and `VALIDATED_VLLM_VERSION` is bumped ✅; `InputBatch` and scheduler tests pass ✅; NIXL
deferred and DI rejected at config time ✅; **GPT-OSS and Qwen3-VL serve at T3 with acceptable logit
drift against the P0.1 baseline — outstanding, needs Trn2.**

That last clause is the whole of what remains, and it is not a formality: it is the only item that
would catch a 0.26 regression in a *supported* model, which is exactly the risk the pin move
introduces. P1 design work can proceed in parallel, but P0 is not closed until it runs.

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
Extend to a model containing at minimum: a **c128 + hash-MoE** layer (the default V4 layers 0–2
combination), an **SWA layer**, a **c4 + routed-MoE** layer, and a **c128 + routed-MoE** layer.
Attention and MLP kinds are independently selected; no implementation may infer one from the other.
This is what makes P1's heterogeneous
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
