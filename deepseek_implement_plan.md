# Track upstream vLLM and add DeepSeek-V4 to vllm-neuron

## Context

`vllm-neuron` is the AWS Trainium platform plugin for vLLM. It is pinned to `vllm==0.21.0`
(`requirements/core.txt:6`, package version `0.21.0.1.0.0`), while upstream vLLM is at 0.27.1.
Two goals: get back onto a current upstream, and serve DeepSeek-V4 — a model the plugin cannot
run at all today, since nothing in the repo implements latent (MLA) attention, sparse attention,
or block-scaled FP8/FP4.

Three findings from the research below shape the whole plan, and two of them change the obvious
approach:

1. **0.27 is externally blocked; 0.26 is free.** vLLM 0.21 *through 0.26* all pin `torch==2.11.0`.
   0.27.0 jumps to `torch==2.13.0`. The Neuron pip index tops out at `torch-xla 2.12` and
   `libtorch-neuronx-lite 2.12` (the plugin uses these, not the heavier `torch-neuronx`, via
   `vllm_neuron/utils/import_redirector.py`). So 0.21→0.26 is a **pure Python-API port with no
   torch change**, and 0.27 waits on an AWS SDK release. Per your call: land 0.26 now.

2. **The upgrade is much smaller than the coupling surface suggests.** All ~45 `vllm.*` modules the
   plugin imports still exist at 0.26, and the signatures most at risk are byte-identical
   (details in Workstream A). The real work is concentrated in three places, not spread across 110
   import sites.

3. **DeepSeek-V4 does not need grouped/node-limited top-k routing.** Unlike V3, the V4 configs carry
   no `n_group`/`topk_group`. That removes what would otherwise be the largest MoE-routing gap.

Target: **DeepSeek-V4-Flash first** (284B total / 13B active), staged correctness-first.

---

## What DeepSeek-V4 actually is

From `deepseek-ai/DeepSeek-V4-Flash/config.json` and the upstream reference implementation at
`vllm/models/deepseek_v4/` (present since vLLM 0.22; `nvidia/model.py` + `attention.py` +
`sparse_mla.py` + `compressor.py` are the semantics of record — download them, they are the spec):

| | V4-Flash (target) | V4-Pro |
|---|---|---|
| hidden / layers / heads | 4096 / 43 / 64 | 7168 / 61 / 128 |
| `head_dim`, `num_key_value_heads` | 512, 1 | 512, 1 |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | 1024 / 1024 / 8 | 1536 / 1024 / 16 |
| experts: routed / top-k / shared / interm. | 256 / 6 / 1 / 2048 | 384 / 6 / 1 / 3072 |
| indexer: heads × dim, `index_topk` | 64 × 128, 512 | 64 × 128, 1024 |
| `compress_ratios` | `[0,0,4,128,4,128,…,0]` | `[128,4,128,…,0]` |

Shared by both: `scoring_func=sqrtsoftplus`, `topk_method=noaux_tc`, `norm_topk_prob`,
`swiglu_limit=10.0`, `num_hash_layers=3`, `sliding_window=128`, `hc_mult=4`,
`hc_sinkhorn_iters=20`, `num_nextn_predict_layers=1` (MTP), YaRN (factor 16 from 65536),
`vocab_size=129280`, weights FP8 `weight_block_size=[128,128]` scale fmt `ue8m0` with
`expert_dtype=fp4`.

Five architectural pieces, and how each lands on this codebase:

- **MLA attention.** `fused_wqa_wkv` → q-lora + a single 512-d latent KV per token; `wq_b` expands
  to 64×512; MQA against the latent; low-rank output projection (`wo_a`/`wo_b`, grouped by
  `o_groups`) with inverse-RoPE; plus per-head attention sinks.
- **HCA (Heavily Compressed Attention).** Per-layer `compress_ratio` ∈ {4, 128}: the compressor
  (`compressor.py:211 DeepseekCompressor`) folds `compress_ratio` tokens into one cached latent
  using learned `ape` positional weights, carrying partial state across chunk boundaries in a
  `CompressorStateCache`. Layers with ratio 0 are sliding-window (128) instead.
- **CSA (Compressed Sparse Attention).** A 64-head/128-dim "lightning indexer" scores compressed KV
  entries; attention runs over the top `index_topk` only. **The indexer only affects selection, not
  weights** — so when the compressed KV length ≤ `index_topk`, CSA is exactly dense attention. For
  Flash that means prompts under ~2048 tokens make CSA a no-op on every layer. This is the lever the
  staged bring-up rides on.
- **mHC (Manifold-Constrained Hyper-Connections).** `hc_mult=4` widens the residual stream 4×, with
  per-layer mixing matrices normalized by 20 Sinkhorn iterations. Upstream fuses this into tilelang
  kernels; there is a plain decomposition (norm → Sinkhorn → mix) that traces fine.
- **MoE.** 256 experts, top-6, one shared expert, `sqrtsoftplus` scoring, `noaux_tc` selection bias
  (`e_score_correction_bias` added for *selection only*), `routed_scaling_factor=1.5`. Layers
  0–2 are **hash MoE**: routing is a `tid2eid[vocab, 6]` table lookup on the token id — no gate.

**Memory.** Flash's expert weights are ~277B params: ~138 GB at FP4, ~277 GB at FP8, ~554 GB at
BF16. A `trn2.48xlarge` is 16 devices, so even a BF16 dequantized-at-load bring-up fits on one
node — which is what makes the correctness-first milestone practical before any quantization work.

---

## Workstream A — upgrade vLLM 0.21.0 → 0.26.0

Rename to `0.26.0.1.0.0` (`pyproject.toml:10`, branch `release-0.26.0.1.0.0`), set
`vllm==0.26.0` and `transformers>=5.5.3` (0.24+ requires it) in `requirements/core.txt`.

### A0. Consolidate the monkeypatches first (do this before bumping the pin)

`vllm_neuron/vllm/patches/__init__.py:15 apply_patches()` is empty and nothing calls it, while
~12 patch sites are scattered across `platform.py`, `neuron_parallel_state.py`, `core/scheduler.py`,
`neuron_worker.py`, and `neuron_nixl_connector.py`. Move them behind that hook and give each one a
`hasattr`/`inspect.signature` guard modelled on the existing tripwire at
`vllm_neuron/vllm/core/scheduler.py:304-307`. This converts the silent-failure class (a renamed
upstream symbol that makes a patch a no-op) into loud startup errors — which is the difference
between a one-week upgrade and a month of chasing wrong behaviour. Highest-value guards:

- scheduler-class path strings `"vllm.v1.core.sched.scheduler.Scheduler"` /
  `AsyncScheduler` — `vllm/platform.py:376-379`; if these stop matching, the GPU scheduler silently
  runs on Neuron.
- `ParallelConfig.__pydantic_core_schema__` 4-level walk — `vllm/platform.py:769-780`.
- `_patch_shutdown` / `_ensure_worker_termination` — `vllm/platform.py:863-902`.
- the model-registry log-message regex — `vllm/worker/neuron_worker.py:179`.
- `in_the_same_node_as` (patched in two places) — `neuron_worker.py:490`,
  `parallel/neuron_parallel_state.py:822`.

### A1. Verified non-issues — confirm, don't rewrite

Diffed 0.21 vs 0.26 directly; these need no change:

- `Scheduler.__init__` — 8-arg signature **identical**, so the mirror at
  `vllm_neuron/vllm/core/scheduler.py:89` stands.
- `Platform.get_attn_backend_cls(selected_backend, attn_selector_config, num_heads=None)` —
  **identical**.
- `KVCacheManager.allocate_slots` gained `reserved_blocks` and `has_scheduled_reqs`, both defaulted;
  the SWA-DI wrapper at `core/scheduler.py:321` takes `*args, **kwargs`, so it passes them through
  and its tripwire still holds.
- `SchedulerOutput` is still a plain mutable `@dataclass`, so the injected
  `_grammar_bitmask` / `num_scheduled_tokens_padded` fields still work.
- `ParallelConfig.all2all_backend` is still a Literal-typed field.
- Every `vllm.*` module the plugin imports resolves at 0.26 (checked all ~45).
- `Platform` interface changes are purely additive.

### A2. The three real breaks

1. **NIXL connector — the big one.** Upstream split `NixlConnector` into
   `NixlBaseConnector` / `NixlPullConnector` / `NixlPushConnector`, and moved the worker/scheduler
   into `nixl/base_worker.py` (110 KB), `pull_worker.py`, `push_worker.py`,
   `pull_scheduler.py`, `push_scheduler.py`. `vllm_neuron/vllm/kv_connector/neuron_nixl_connector.py`
   subclasses `NixlConnectorWorker`, overrides five *private* methods
   (`_nixl_handshake`, `_read_blocks_for_req`, `_validate_remote_agent_handshake`, …) and reads ~15
   private parent attributes — **none of which import at 0.26**. Re-derive against the pull/push
   split; the in-file comment at `neuron_nixl_connector.py:96-99` shows this file already broke on
   the 0.19→0.20 bump, so budget accordingly. Decide early whether Neuron maps to the pull or push
   worker (1P1D/xPyD today is a pull).
2. **`InputBatch` construction** — `vllm_neuron/vllm/worker/neuron_model_runner.py:7658`. Upstream
   dropped the `pin_memory` parameter (now a module-level `PIN_MEMORY`), made
   `max_num_blocks_per_req` required rather than optional, and added `slot_mapping_modes`.
3. **Re-sync the two upstream ports.** `GPUModelRunner` changed ~1900 of 7900 lines and
   `Scheduler` ~1050 of 2900 between the tags. Two files carry line-for-line ports that will drift
   silently, not break loudly:
   - `NeuronModelRunner._update_states` (`neuron_model_runner.py:1815-2036`) vs upstream
     `GPUModelRunner._update_states` — re-diff field by field, especially
     `scheduler_output.scheduled_cached_reqs.*` and the `CachedRequestState` fields.
   - `NeuronAsyncScheduler._update_after_schedule` (`core/scheduler.py:943-1023`) vs upstream
     `AsyncScheduler`.

### A3. Adopt the new hooks (needed by Workstream B)

0.26's `Platform` adds `register_custom_kv_cache_specs(vllm_config)` and
`_align_heterogeneous_kv_block_size(...)`, and `MLAAttentionSpec` gained
`compress_ratio` / `alignment` / `model_version` / `kv_quant_mode`, plus new `HiddenStateCacheSpec`
and `RSWASpec` kinds. These exist precisely because of DeepSeek-V4's heterogeneous per-layer caches.
Wire them in `vllm_neuron/vllm/platform.py` and teach
`neuron_model_runner.initialize_kv_cache` (`:7770-7773`, currently `NotImplementedError` for
anything but `FullAttentionSpec`/`SlidingWindowSpec`) to handle MLA specs.

---

## Workstream B — DeepSeek-V4-Flash

Follow `docs/model-dev/onboarding-models.md` (the official 5-step process: implement → register →
compile/smoke → validate → benchmark). New package `vllm_neuron/model/deepseek_v4/` mirroring
`vllm_neuron/model/gpt_oss/` (the MoE template the doc points at), plus one line in
`vllm_neuron/model/registry.py:19-24` mapping `"DeepseekV4ForCausalLM"`.

Copy **llama3's** variant-selection pattern, not gpt_oss's: `resolve_attention_mlp_classes`
(`model/llama3/quantization.py:236-309`) picks attention/MLP classes *per layer*, which is exactly
what V4 needs — SWA layers vs cr=4 vs cr=128 layers vs hash-MoE layers vs MTP layer differ
structurally. gpt_oss instead forks the entire model file per quantization (`model_bf16.py` and
`model_mxfp4.py` are near-duplicate 2200-line files); do not repeat that.

Files: `config.py` (~150), `factory.py` (~80), `model.py` (~2000–2500), `attention.py`,
`compressor.py`, `moe.py`, `weight_loaders.py` (~600), `__init__.py`, plus
`examples/vllm_neuron/models/deepseek_v4/` run scripts and a model `README.md`
(template at `docs/model-dev/onboarding-models.md:675-712`).

### Reusable as-is

The MoE stack largely fits: `NF.router` supports sigmoid scoring and the
`PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER` order (i.e. `norm_topk_prob`);
`NF.moe_block_tkg` (decode) already handles top-k renorm **and shared experts**;
`NF.build_blockwise_mapping` + `NF.moe_cte` (prefill); expert parallelism with contiguous
placement (`functional/expert_parallel.py`) and — notably —
`expert_parallel_interleaved_loader` in `utils/weight_loader.py:676,727` was written against
DeepSeek checkpoint layouts (bf16 stride 2, fp8 stride 4). Also reusable: the whole
TP/SP/DP/EP collective plumbing, paged KV + FP8 KV cache, on-device sampling, and the
accuracy tooling in `vllm_neuron/accuracy/`.

### Must be built

| Gap | Why it's new | Where it bites |
|---|---|---|
| MLA / latent attention | Nothing in the repo does low-rank KV | `head_dim=512` vs `MAX_HEAD_DIM=128` at `functional/attention/attention_cte.py:16` (falls back to torch) and `_MAX_HEAD_DIM=128` at `attention_segmented_cte.py:36` (**raises**) |
| Single-tensor latent KV cache | Cache is a fixed K/V *pair* `(2, blocks, kv_heads, block_size, head_size)` | `neuron_model_runner.py:7731-7741`; `model/kv_cache.py` `LayerSpec` can express `num_kv_heads=1, head_size=512`, so the spec is fine — the pair layout is not |
| HCA compressor + carry state | Strided reduction with cross-chunk state | new; needs a KV-cache-like state store per request |
| mHC (4× residual + Sinkhorn) | No analogue | new; decomposed PyTorch first |
| CSA indexer + token-level top-k gather | Only contiguous `bound_min/bound_max`, SWA, and block-granular `active_blocks_table` exist | `functional/attention/attention_cte.py:202`, `attention_decode.py:944` |
| `noaux_tc` selection bias, `routed_scaling_factor`, `sqrtsoftplus` | `router_bias` is applied pre-activation so it changes the gate value too — semantically wrong for `noaux_tc` | `functional/moe/router.py:75,568-599` |
| Hash MoE (layers 0–2) | Table lookup routing | new, but trivial |
| Shared expert in **prefill** | Only wired in the decode kernel; also absent from the torch fallback, which breaks CPU parity | `functional/moe/moe_cte.py`, `moe_block_tkg.py:361-363` |
| FP8 128×128 block-scale + FP4 experts | `QuantScheme` has only `NONE` and `FP8_STATIC_PER_TENSOR`; MX is a different (per-32 microscaling) family | `model/llama3/quantization.py:39-51` |
| MTP | Only EAGLE3 exists (README lists MTP ❌) | `vllm/spec_decode/` |

### Milestones

**M1 — short-context BF16 correctness (the gate for everything else).**
Dequantize FP8/FP4 weights to BF16 at load (precedent: `gpt_oss/weight_loaders_bf16.py:264
_dequantize_mxfp4_to_bf16`); ~554 GB fits one trn2.48xlarge. Implement mHC, the HCA compressor, MLA,
SWA layers, and the full MoE (sqrtsoftplus + noaux_tc + hash layers + shared expert) in plain
traceable PyTorch, accepting the sub-128-head_dim kernel fallbacks. **Skip the indexer entirely** and
attend densely over compressed KV, restricting prompts to ≤2048 tokens where that is exactly
equivalent. Validate logits against HF on CPU (`VLLM_NEURON_CPU_MODE=1`).

**M2 — quantization.** New `QuantScheme` member + loaders for FP8 `weight_block_size=[128,128]`
with `ue8m0` scales, and FP4 experts. Drops the footprint to ~138 GB and makes the memory story real.

**M3 — long context.** The CSA indexer, a token-level top-k KV gather, and NKI kernels for
`head_dim=512` MLA. This is the dominant kernel risk in the whole plan; everything before it runs
on torch fallbacks.

**M4 — MTP, perf tuning, disaggregated inference.**

Defer V4-Pro until M2 lands: identical architecture, so it becomes a config + sharding exercise
(~800 GB at FP4).

---

## Verification

There is **no test harness to plug into** — `test/` and `ci/` do not exist in this repo despite
`pyproject.toml:46-50` referencing them (it is a stripped public mirror), and `.github/workflows/`
holds only issue-labeling automation. Part of this work is creating that scaffolding.

- **Workstream A gate:** `vllm serve openai/gpt-oss-20b --tensor-parallel-size 8` and the
  Qwen3-VL recipe both still serve, plus the 1P1D disaggregated example under
  `examples/vllm_neuron/vllm/disaggregated_inference/` (that one specifically exercises the
  rewritten NIXL path). Re-run the accuracy scripts in `examples/vllm_neuron/accuracy/` and compare
  against pre-upgrade logits — the drift risk in A2.3 is silent, so a logit diff is the only real
  check.
- **Workstream B gate:** the two levels the onboarding doc prescribes
  (`docs/model-dev/onboarding-models.md:828-915`) — per-module comparisons against HF on CPU using
  `assert_close_three_way(target=neuron, expected=hf_fp32, baseline=hf_bf16, rtol=0.01)`
  (`vllm_neuron/accuracy/testing.py:289`), then end-to-end
  `multi_prompt_logit_validation` (`accuracy/logit_validation.py`) across a TP/seqlen/batch grid.
  Use `tensor_capture` / `tensor_replacement` to bisect the first diverging module — with mHC,
  HCA, and MLA all new at once, this is the only tractable debugging loop.
- **Build `test/unit/` and `test/vllm_neuron/`** as the doc's tiny-model pattern describes
  (random 1-layer model, `enforce_eager=True`, `num_gpu_blocks_override=4`, `hidden_size % 256 == 0`),
  so the compat regressions from A2.3 get caught by CI rather than by a logit diff months later.

## Open items

- **Confirm the Neuron SDK roadmap for torch 2.13 with AWS.** It sets the date for 0.27, and
  nothing in this plan can move it. Evidence here is the pip index
  (`pip.repos.neuron.amazonaws.com`), not an AWS statement.
- vLLM ships roughly every two weeks (0.21 → 0.27 was 2026-05-15 → 2026-08-10). Landing on 0.26
  fixes the gap once; staying current needs the A0 guard rails plus a compat CI job, or this
  recurs.
- M3's NKI kernel work (512-dim MLA + token-level top-k gather) is the one item that could
  invalidate the schedule. Consider prototyping it against the NKI CPU simulator during M1 rather
  than waiting.
