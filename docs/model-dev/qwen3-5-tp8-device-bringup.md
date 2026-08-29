# Qwen3.5 text decoder: first TP=8 device bring-up

Date: 2026-08-29. Host: trn2.48xlarge, `logical-neuroncore-config 2`, devices 8+9
(logical cores 32-39). Fixture: `tools/qwen3_5/build_tiny_checkpoint.py`, 8 layers
(6 Gated DeltaNet + 2 full attention), TP=8.

## Result

**TP=8 works on device.** Eight ranks, sharded GDN projections, sequence-parallel
collectives, the hybrid Mamba/attention state cache, and both the prefill and
decode compiles all complete, generating tokens end to end:

```
RESULT {"tensor_parallel_size": 8, "visible_cores": "32-39",
        "prompt_length": 32, "token_ids": [449, 731, 154, 145, 970, 891, 28, 120]}
```

That run has the **chunk-scan NKI kernel on its torch fallback**; the depthwise
conv kernel is active. See "Open blocker" below.

The fixture uses random weights, so this establishes *structure*, not accuracy --
the DeepSeek-V4 record is explicit that tiny-model logits cannot settle numerical
questions (`deepseek-v4-tiny-tp1-neuron-investigation.md:36-43`). Two passing
configurations produced different token sequences after the first token, which is
expected at this scale but means **the NKI conv's numerics have not been verified
against torch on device** -- only that both compile and run.

## Open blocker: the chunk-scan kernel fails neuronx-cc codegen

With the chunk-scan kernel enabled, the **decode** graph fails:

```
[INTERNAL_ERROR] [NCC_IXGM002] Expected function sg0000 in subgraph 0 to have
49 basic blocks, but on core 1 it has 1 basic blocks
```

Prefill compiles fine. Attribution is from a full 2x2, one device run per cell:

| conv kernel | scan kernel | outcome |
|---|---|---|
| off | off | pass |
| off | on  | fail (NCC_IXGM002, decode) |
| on  | on  | fail (NCC_IXGM002, decode) |
| on  | off | **pass** -- the best working configuration |

So the scan kernel is necessary and sufficient; the conv kernel is not implicated.

Two hypotheses were tested and **disproved**:

- *The LNC=2 core split.* Forcing `scan_lnc` to return 1 reproduced the error
  identically. The host is LNC=2 globally, so "core 1" is the second physical
  core of every logical core -- the message is about the whole subgraph, not the
  kernel's launch grid.
- *Decode does not use the scan.* Decode runs the recurrent path, yet enabling
  the scan is what breaks the decode compile. Whatever the mechanism, presence of
  the kernel is the trigger.

The message asks for a support ticket, so this may be a compiler issue rather
than a defect in the kernel source. Next step is `XLA_IR_DEBUG=1
XLA_HLO_DEBUG=1` and a per-core comparison of `sg0000`.

Reproduce: build the fixture, run `tools/qwen3_5/generate_tiny.py` at TP=8. To
get a working run meanwhile, make `can_use_chunk_scan_kernel` return False.

## Bugs found and fixed

All three were invisible to the 199-test suite because CPU torch silently
promotes dtypes; only the device rejects them. The first two predate the
tensor-parallel work.

1. **GDN projections built at fp32** (`gated_deltanet.py`). The layer stored
   `self.dtype = config.torch_dtype` and never passed it to its `nn.Linear` /
   `nn.Conv1d` constructors, so all six projections defaulted to fp32 -- 48 of
   the real model's 64 layers at double width. `Qwen3_5MLP` and
   `Qwen3_5Attention` both pass the dtype; the GDN was the outlier. Present since
   the layer was written.

2. **One fp32 mask applied to both paged states** (`gated_deltanet.py`,
   introduced in `877e7e9`). The validity mask was built at
   `recurrent_state.dtype` (fp32) and multiplied into both states, silently
   promoting the bf16 conv state. It then reached the depthwise conv through
   `torch.cat` and the kernel rejected the pair:
   `nc_matmul: if one input is tfloat32/float32, both must be`.

3. **Conv state dtype at the kernel boundary** (`nki_gdn.py`). The conv now casts
   its state to the activation dtype before concatenating, so the kernel call is
   dtype-consistent regardless of how the cache is typed. The function already
   treated `hidden_states.dtype` as authoritative on return.

## Design change, flagged for review

`get_kv_spec` now declares **both** GDN states fp32 rather than bf16 conv + fp32
recurrent. The runner carves both from one raw page
(`neuron_model_runner.py:8554`, mirroring vLLM's GPU carving) and the model
writes both back in place; two views over one storage with different dtypes,
each mutated, is rejected:

```
AssertionError: aot_autograd() does not yet handle input mutations on views
with different dtypes.
```

Matching the dtypes is the cheap way out. The conv window is ~2% of the page, so
the real 27B pays **+1.9%** state memory. The better fix is to give each state
its own allocation instead of sharing a page; that changes cache accounting, so
it was left as a deliberate choice.

## Blockers for serving a real checkpoint

Neither is caused by the TP work; both are hit before any sharding runs.

1. **Factory signature vs. the multimodal branch.** `platform.py:294`
   (`_resolve_vision_auto_config`) injects a `vision_neuron_config` for any
   config carrying `hf_config.vision_config`. That makes
   `neuron_model_runner.load_model:1267` take its multimodal branch and call
   `from_configs(hf_config=..., text_neuron_config=..., vision_neuron_config=...)`,
   but `Qwen3_5ForCausalLM.from_configs` accepts only `(hf_config, neuron_config)`.
   Every released Qwen3.5 checkpoint carries a vision tower, so this fires for all
   of them, at every TP degree including 1. Reproduce with
   `build_tiny_checkpoint.py --wrapper-config`.

2. **Hybrid page-size misalignment.** vLLM unifies two cache groups only when the
   Mamba page is an exact multiple of the attention page
   (`kv_cache_utils.py:1081`), or when the attention backend sets
   `indexes_kv_by_block_stride` so the page can be padded. The Neuron spec
   conversion sets neither. The real 27B at TP=8 is misaligned at **every** block
   size:

   ```
   block_size=8    att=8192    mamba=400896  remainder=7680
   block_size=16   att=16384   mamba=400896  remainder=7680
   block_size=32   att=32768   mamba=400896  remainder=7680
   block_size=64   att=65536   mamba=400896  remainder=7680
   block_size=128  att=131072  mamba=400896  remainder=7680
   ```

   This also constrains the fixture: page alignment and the kernel's
   `k_dim, v_dim in (16, 32, 64, 128)` assertion (`nki_gdn.py:234`) nearly
   exclude each other. `key_head_dim=64` / `value_head_dim=128` at `head_dim=32`,
   `--block-size 8` is one of the few pairs satisfying both.

3. **Parent-process architecture resolution.** vLLM's parent resolves the
   architecture against its own registry before any Neuron code runs, and
   `NeuronWorker` registers this repo's classes only inside the worker. vLLM ships
   no text-only Qwen3.5 entry -- only the multimodal
   `Qwen3_5ForConditionalGeneration` -- so the parent builds a Qwen3-VL
   preprocessing stack and demands an image processor. Every other Neuron model
   shadows an architecture vLLM already ships (e.g. `DeepseekV4ForCausalLM` at
   `registry.py:100`), which is why this had not come up. `generate_tiny.py`
   registers a text-only shim in the parent as a bring-up workaround; the durable
   fix is a plugin-level decision.

## Operational note

A failed compile can leave the EngineCore and its workers alive holding the
devices. The next launch then fails with `Failed to initialize Neuron Runtime:
status code 1`, which says nothing about contention and reads like a new bug.
Check `neuron-ls -a` for stray `VLLM::Worker_TP*` before each run.
