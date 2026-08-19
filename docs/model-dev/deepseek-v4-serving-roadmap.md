# DeepSeek-V4: what remains before the model can be served

> **Status: Steps 1-3 done, validated against the tiny/synthetic config.**
> `model.py` now implements the real `attn_metadata`-driven batched forward
> contract, real `bind_kv_cache`, and real paged cache I/O for the SWA,
> compressed-MLA, and compressor-carry cache groups (design:
> [`deepseek-v4-carry-cache-design.md`](deepseek-v4-carry-cache-design.md)).
> TP scaffolding (column/row-parallel value/out projections) and EP
> scaffolding (dense-compute-plus-all-reduce, numerically correct at any
> `ep_degree`) are in place. `factory.py` matches the `ModelRegistry` pattern.
> Registration is opt-in behind `VLLM_NEURON_ENABLE_DEEPSEEK_V4=1`
> (`registry.py`) — **still not default**, because what follows is still
> true: real checkpoint loading/quantization (P9) and memory calibration
> (P7b/P8) are not built. (Attention was upgraded to the real multi-head
> q_lora/kv_lora/RoPE MLA architecture and the expert FFN to the real
> clamped-SwiGLU, `[out,in]`-layout `DeepseekV4Experts`/`DeepseekV4MLP` shape
> — see below; the output projection is the one attention piece still
> simplified, a plain dense `Linear` instead of the real grouped low-rank
> `o_a_proj`/`o_b_proj`.)
> An end-to-end `vllm.LLM()` run
> (`test/vllm_neuron/test_deepseek_v4_device_e2e.py`) now completes a real
> `generate()` call — registration, config resolution, weight loading,
> heterogeneous KV-cache initialization, forward, and CPU sampling all run
> through the actual engine. Getting there surfaced and fixed four
> previously-undiscovered bugs this effort is the first thing to exercise:
> a real one in `mla_cache_shape` (`input_batch_params.py`) that ignored
> `page_size_padded` when vLLM pads a merged heterogeneous cache group,
> sizing the physical tensor view wrong by exactly the padding factor; a
> missing `@torch.no_grad()` on the model's top-level forward (every other
> model in this plugin has one); and two config facts specific to this
> model's CPU-sampling (non-ODS) contract that needed making explicit
> rather than fixed — on-device sampling is this plugin's *default*
> (`async_scheduling` and `on_device_sampling_config` both assume it unless
> told otherwise), and this model doesn't implement it. All four are
> detailed in the test file's own comments.
>
> **Direct comparison against the real architecture**
> (`test/vllm_neuron/test_deepseek_v4_matches_real_architecture.py`): every
> wrapper module this pass added or touched — `DeepseekV4HyperConnection`,
> `DeepseekV4HyperHead`, the routed/hash MoE gates, and the HCA/CSA
> compressor's projection+reduction+norm — matches
> `transformers.models.deepseek_v4.modeling_deepseek_v4`'s real modules
> exactly (0.0 diff) once driven by the same weights. That comparison is
> also what found the one real bug this pass introduced and then fixed: an
> earlier version of `DeepseekV4DecoderLayer` fed the mHC-collapsed hidden
> state straight into attention/MoE, omitting the `input_layernorm`/
> `post_attention_layernorm` the real layer applies first. It also surfaced
> two harmless naming mismatches, now corrected to match both the real
> module and the existing `weight_loaders.py` checkpoint-prefix
> conventions: the final head is `hc_head`, not `hyper_head`, and mHC scale
> parameters are `hc_scale` everywhere (avoiding the separate collision
> with `weight_loaders.py`'s FP8-dequant `.scale` convention found earlier).
>
> **Attention was then rewritten to the real multi-head q_lora/kv_lora/
> partial-RoPE MLA design** (real `q_a_proj`/`q_a_norm`/`q_b_proj`,
> `kv_proj`/`kv_norm`, the real `DeepseekV4RotaryEmbedding`, attention sinks,
> K=V broadcast via an identity "up-projection" since this plugin's MLA
> cache stores one shared latent per token, and the real architecture's
> undo-RoPE-on-the-output step) and cross-validated against the real module
> both in isolation and through real paged cache I/O
> (`test_attention_matches_real_module_through_paged_cache_io`); see
> [`deepseek-v4-carry-cache-design.md`](deepseek-v4-carry-cache-design.md)
> for the full account.
>
> **The expert FFN (`DeepseekV4Expert`, both routed and shared) was then
> rewritten to match the real `DeepseekV4Experts`/`DeepseekV4MLP` exactly**:
> `[out, in]`-layout `gate_up_proj`/`down_proj` driven through `F.linear`
> (not the earlier `[in, out]`-layout plain `@`), and gate/up clamped to
> `swiglu_limit` before the SiLU*up product (not the earlier unclamped
> SwiGLU — a real numerical divergence for any input large enough to hit the
> clamp, not just a cosmetic layout difference). Cross-validated exactly
> (0.0 diff) against both the real routed-expert math and the real shared
> expert in `test_expert_wrapper_matches_real_module`.
>
> The output projection (grouped low-rank `o_a_proj`/`o_b_proj`) remains the
> one documented, deliberate simplification left — not compared, since it's
> not meant to match.
>
> The rest of this document is the original gap analysis and is otherwise
> unchanged below — read it as history for what was true before this pass,
> except where superseded above.

`DeepseekV4ForCausalLM` is not registered with vLLM, and adding it to
`vllm_neuron/model/registry.py` would not make the model servable. This document
records why, what carries over, and what has to be built.

The short version: the gap is **implementation, not test coverage**. What exists
today is a production-*shaped* CPU reference model that proves the math and the
cache contract. The scheduler-integrated inference path does not exist yet.

## Why registration alone does nothing

Registration itself is mechanical — `neuron_worker.py:381` loops over
`registry.get_models()` and calls `ModelRegistry.register_model(arch, cls)`.
Three things would break immediately afterward.

**The forward contract does not match.** A servable model on this plugin looks
like `llama3/model.py:1610`:

```python
forward(input_ids, positions, inputs_embeds, is_token_ids, attn_metadata,
        sampling_positions, sampling_params, spec_decode_metadata,
        logit_mask, rank, **kwargs) -> logits
```

`deepseek_v4/model.py:179` is:

```python
forward(input_ids, state: TinyModelState | None) -> (logits, TinyModelState)
```

No positions, no `attn_metadata`, no block tables, no sampling parameters. KV
state is threaded through a Python object the caller hands back in — the
scheduler has no way to supply that.

**`bind_kv_cache` validates but does not bind.** `deepseek_v4/model.py:159`
checks the cache key set and one-tensor-per-layer arity, then assigns
`self._kv_caches` — and nothing ever reads that attribute. Compare
`llama3/model.py:1797-1803`, which attaches the tensors onto each attention
module (`layer.self_attn.k_cache = kv_caches[layer_name][0]`) so prefill and
decode can scatter and gather through them. DeepSeek-V4 declares and allocates
its caches correctly; it never performs paged cache I/O.

**The forward pass is not a capturable graph.** `DeepseekV4Model.forward`
(`model.py:63`) raises unless `input_ids` is 1-D, then runs
`for token_id in input_ids:` — one token at a time, batch of one. There is also
no parallelism anywhere in the package: zero references to `world_size`,
`all_reduce`, `all_gather`, or any process group. `reference_config_from_hf`
builds, in its own words, "a small-expert reference geometry".

This is deliberate. `deepseek_actual_implement_plan.md:63` states it directly:
runner allocation and strict single-tensor binding are covered, "but
scheduler-metadata-driven cache I/O and graph capture remain, so registry
exposure is intentionally withheld." Registering the model would advertise
support that does not exist.

## Step 0 — confirm graph capture before writing model code

P2.c (FX→HLO→NEFF capture) was blocked solely on the Torch 2.9 / 2.11 mismatch.
The move to `release-0.24.0.1.1.0` is what plausibly unblocks it, and that is
**partially verified now**: the toolchain question itself is answered —
[`deepseek-v4-024-device-validation.md`](deepseek-v4-024-device-validation.md)'s
Step 5d ran real `torch.compile`/Dynamo tracing of the actual registered
model, through the real Neuron compile backend, on real Trn2 silicon, past
model construction/weight loading/device move and two real graph-break
fixes. Not yet a green compile — it stopped on a third, deeper Dynamo/
FakeTensor issue in the per-token attention loop's `position_ids`
construction. That one is now fixed in `model.py` and **confirmed on real
Trn2 silicon** (root cause: a per-token-loop-local Python int, symbolic
under Dynamo, was getting re-embedded in a fresh `new_tensor([[...]])`
Python-list construction that FakeTensorMode doesn't track —
`position_ids` is now built purely through tensor ops instead). Re-running
against the fix showed real forward progress — tracing got well past
`position_ids`/`rotary_emb` — then hit two more, different blockers in the
same class, one at a time, each confirmed by an actual device re-run: a
guard-clause branch in `attention.py::scatter_paged_latent` (fixed, same
pattern as the two graph-break fixes above — a throughput-only early
return, safe to make unconditional) and then a genuine, not-yet-fixed one
in `_swa_history`'s `if cached_seq_len == 0:` — this one is real
shape-determining logic (the SWA gather length), not a removable guard, so
it needs the redesign described below rather than a mechanical fix.
Designing that redesign surfaced a real, separate, pre-existing correctness
bug in the same code — confirmed by direct reproduction, not caught by any
existing test — that any redesign here needs to fix alongside the shape
problem, not paper over: see
[`deepseek-v4-swa-null-block-bug.md`](deepseek-v4-swa-null-block-bug.md).
Read Step 5d before spending more device time here: the remaining blocker is in
this plugin's model code (the per-token loop's cache-gather lengths, built
from a data-dependent `cached_seq_len` in ways Dynamo can't turn into a
static shape), not the toolchain. **This is not one of two paths to
real-hardware serving** — Step 5e confirms
`enforce_eager=True` is unconditionally rejected on real Neuron hardware
for every model on this plugin (`neuron_worker.py`'s
`assert not (eager_mode and not cpu_mode)`), so there is no eager
fallback to reach for while 5d's compile blocker stands.

## What carries over

The reference model is not throwaway — it becomes the **oracle** each device
step is validated against.

| Component | Status | Role going forward |
| --- | --- | --- |
| `config.py` | Verified against Transformers 5.15 | Ships as-is |
| `compressor.py`, `mhc.py`, `moe.py`, `attention.py` | Exact fp32 reference, oracle-verified | Correctness target for the device implementations |
| `dense_csa.py` | Enforced in `platform.validate_request` | Done |
| `weight_loaders.py` | Name-mapping contract correct | Must compose with TP/EP sharding |
| `get_kv_spec()`, `kv_spec_conversion.py` | Validated against vLLM's real `KVCacheManager` | Ships as-is |
| `nki_mla.py` | Matched fp32 oracle on Trn2 | Becomes the in-graph kernel |
| `memory_budget.py` | P7a analytical model | Calibrated by P7b |
| `streaming_loader.py` | Prototype | Must compose with sharding |
| `tiny_model.py` | T0 green | Oracle for steps 1–3 |

## What has to be built

### 1. A device-shaped model class

Rewrite `model.py` against the `llama3/model.py:1610` contract: batched inputs,
metadata-driven, no Python token loop, prefill/decode split on
`max_query_len > decode_token_threshold`. `bind_kv_cache` attaches tensors to the
attention modules rather than storing a dict.

Oracle: `tiny_model.py` — identical logits for the same inputs.

### 2. Paged MLA attention with real cache I/O

The hard part. Attention must scatter by `slot_mapping` and gather by
`block_table_tensor` from `attn_metadata`, as `llama3` does in `forward_prefill`
(`model.py:577-580`).

Two things make this harder than a routine port:

- The MLA cache is **one latent tensor**, not a K/V pair, so the existing
  read/write helpers do not apply unmodified.
- The `COMPRESSOR_STATE` carry caches have **no precedent in this plugin** — no
  existing model reads or writes a cross-chunk carry cache. Chunk invariance,
  which `compressor.py` currently guarantees in Python via explicit carry state,
  has to survive inside a compiled graph.

Oracles: `compressor.py`, `attention.py`, and the chunk-invariance properties
already covered at T0.

### 3. Parallelism

None exists today. Needed: TP/SP for attention, EP for the routed experts. The
onboarding guide's expert-sharding case (c),
`docs/model-dev/onboarding-models.md:575`, documents the interleaved DeepSeek
`gate_up` layout specifically.

Oracle: routing and logits must be invariant across TP/SP/EP configurations.

### 4. Real checkpoint loading, and the memory decision

Scale progressively — one real component, then one decoder layer, then selected
heterogeneous layers, then the full checkpoint. Per
`deepseek_actual_implement_plan.md:648`, avoid any loop requiring two full 284B
BF16 copies per iteration.

The sizing constraint (P7): ~284B parameters ≈ **568 GB at BF16**, against
96 GB × 16 = 1,536 GB aggregate on Trn2 — and aggregate HBM is not evidence of
feasibility once KV cache, activations, collectives, and compiler arena are
counted. Measured P7b peak decides whether BF16 short-context (P8) runs first or
whether native FP8/FP4 (P9) comes first. **P9 is mandatory either way**; P7 only
changes the ordering.

### 5. Registration and onboarding artifacts

The trivial part, done last:

- A real `factory.py`. The current one is a per-layer component selector
  (`ComponentRegistry`, `resolve_layer_components`), not the `nn.Module` +
  `from_configs` / `_select_implementation` / `_validate_config` pattern that
  `ModelRegistry` requires — compare `qwen3/factory.py`.
- The `registry.py` entry.
- `examples/vllm_neuron/models/deepseek_v4/` and a model README, template at
  `docs/model-dev/onboarding-models.md:688-712`.

## Ordering

```text
0. graph capture works?  ──no──▶  stop; this is an SDK problem
   │ yes
   ▼
1. device-shaped forward + real bind_kv_cache     oracle: tiny_model.py
   ▼
2. paged MLA + compressor carry-cache I/O         oracle: compressor.py, attention.py
   ▼
3. TP / SP / EP                                   oracle: routing invariance
   ▼
4. real checkpoint, progressively                 gate: P7b measured memory
   ▼
5. BF16 short context (P8)  or  native FP8/FP4 (P9)
   ▼
6. register + onboarding artifacts
```

Steps 1–3 are where the work is. Steps 4–5 are gated on hardware measurement,
not on more local development.

## What stays rejected regardless

Prefix caching and speculative decoding raise `NotImplementedError` for
`deepseek_v4` in `platform.py`, and the dense-CSA admission gate rejects
uncapped generations. These are correctness guards, not gaps — do not weaken
them to make a topology start. Prefix caching needs compressor carry-state reuse
semantics; speculative decoding needs heterogeneous cache fork and rollback.
Both are post-P9.
