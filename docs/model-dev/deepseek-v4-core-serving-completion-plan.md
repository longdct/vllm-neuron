# DeepSeek-v4 Core Serving Completion Plan

## Summary

Complete DeepSeek-v4 onboarding in priority order: eliminate the post-fix
segfault and all compile-unsafe graph behavior first, correct
checkpoint-derived architecture and sharding, implement the full runner
contract including batched decode and sampling, then validate on Trainium
before enabling registration by default. Prefix caching, speculative/MTP
decoding, and FP8/FP4 remain explicitly rejected.

## Implementation changes

### P0 — Localize and eliminate the segfault

- Guard tensor-value validation in `hash_experts` and `hash_topk` with
  `not torch.compiler.is_compiling()` so reductions and tensor-to-Python
  branching never enter the captured graph.
- Audit every DeepSeek-v4 forward-reachable operation for scalar extraction,
  boolean indexing, `nonzero`, data-dependent slicing, or tensor-driven Python
  control flow.
- Add full-graph capture tests for hash MoE, routed MoE, each
  attention/compressor variant, and complete tiny-model prefill/decode. Reject
  captured graphs containing `_local_scalar_dense`, `_assert_scalar`, unbacked
  `sym_size`, `nonzero`, or boolean-mask indexing.
- Add a serial capture diagnostic using
  `VLLM_NEURON_DISABLE_PARALLEL_TRACE=1` that records the active phase,
  component, layer, and bucket immediately before capture, HLO conversion,
  compilation, and load.
- Re-run one prefill and one decode bucket on Trainium after each component is
  restored. Treat the toolchain as responsible only if an individually
  captured static graph still crashes with all model-side dynamic behavior
  excluded.

### P1 — Correct model geometry and checkpoint loading

- Replace production use of `TinyDeepseekV4Config` with a dedicated config
  carrying every runtime-relevant HF field plus `NeuronConfig`; retain the tiny
  config only as an oracle fixture.
- Derive expert intermediate width, shared-expert count, routing scale,
  attention ranks/groups, dtype, embedding behavior, and layer variants from
  the HF checkpoint instead of synthetic defaults.
- Validate unsupported checkpoint variants in the factory before model
  allocation.
- Attach TP loaders to head-sharded attention/output parameters and EP loaders
  to contiguous expert ranges; keep replicated parameters identical across
  ranks.
- Make real-checkpoint loading strict: report missing, unexpected, duplicate,
  incorrectly shaped, or unsharded parameters. Preserve an explicit dummy
  checkpoint path only for tests.

### P2 — Complete the runner interface

- Use runner-provided `positions` directly for RoPE and cache addressing.
- Support multi-request decode by mapping every flattened token to its
  request's block-table row and cached length. Preserve supported
  single-request chunked prefill and reject unsupported mixed layouts.
- Merge `input_ids` and `inputs_embeds` according to `is_token_ids`, matching
  existing Llama behavior and padding semantics.
- Select LM-head rows with `sampling_positions`; consume `rank`,
  `sampling_params`, and `logit_mask` when their configured paths require them.
- Reject speculative metadata during platform validation.
- Preserve fixed-shape null-block cache writes and validate all heterogeneous
  SWA, compressed-MLA, and compressor-state cache layouts.

### P3 — Neuron configuration, parallelism, and sampling

- Retain `NeuronConfig` in the model config and honor TP, EP,
  embedding/MLP/attention/LM-head DP, dtype, buckets, and on-device sampling.
- Use a sharded `ColumnParallelLinear` LM head and the standard Neuron sampler.
- Support CPU sampling with full gathered logits and ODS with sampled token IDs
  plus gathered logits when requested.
- Validate unsupported quantization, DCP/SP combinations, prefix caching, and
  speculative/MTP decoding.
- Verify TP/EP output and routed-expert ownership against TP1/EP1 references.

### P4 — Replace prototype-scale graph paths

- Replace materialized identity-projection MLA and full-capacity reference
  operations with fixed-shape Neuron/NKI-backed prefill and decode paths while
  preserving compressor boundary semantics.
- Dispatch prefill and decode using `max_query_len` and
  `decode_token_threshold`, ensuring every warmup bucket is bounded.
- Adopt the repository's supported expert-parallel dispatch instead of dense
  execution of every expert; retain an oracle fallback only in tests.
- Benchmark graph size, compile time, host RSS, HBM, and runtime latency before
  attempting the full checkpoint.

### P5 — Validation and release

- Update device-validation documentation to distinguish the pre-`803dbcf`
  boolean-mask crash from any post-fix crash and attach evidence for each newly
  identified root cause.
- Keep `VLLM_NEURON_ENABLE_DEEPSEEK_V4=1` until every release gate passes.
- Enable default registration only after BF16 checkpoint loading, every
  required NEFF bucket, multi-request decode, CPU/ODS sampling, accuracy,
  memory, offline generation, and online serving pass on the target Trainium
  topology.

## Test plan

- Unit tests for compile-safe routing, exact HF-derived shapes, config
  rejection, TP/EP weight shards, strict checkpoint coverage, prompt
  embeddings, padding, and cache lifecycles.
- Full-graph CPU capture tests for every layer type and representative
  prefill/decode bucket, with forbidden-node inspection.
- Oracle comparisons against Transformers for attention, compressor, mHC,
  routing, experts, decoder layers, chunk invariance, and logits.
- Engine tests for prefill followed by decode, padded decode buckets, at least
  two concurrent requests, TP1/TP2, CPU sampling, and ODS.
- A Trainium ladder from isolated component capture through progressively
  scaled real checkpoints, retaining logs, HLO/NEFF inventories, versions,
  memory measurements, and numerical outputs.
- Regression tests confirming prefix caching, speculative/MTP decoding, and
  non-BF16 quantization fail clearly before compilation.

## Assumptions

- Core serving includes BF16, prompt token IDs/embeddings, single-request
  chunked prefill, multi-request decode, TP/EP, and CPU/ODS sampling.
- Prefix caching, speculative/MTP decoding, native FP8/FP4, and default public
  registration remain out of scope until separate cache-state and quantization
  work is validated.
- Existing user files and untracked patch bundles remain untouched.
- A post-fix native crash is not classified as a toolchain defect until a
  minimized static component graph reproduces independently of the full model.
