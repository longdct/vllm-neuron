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
conv kernel is active -- see "Root cause" below, which explains why and
what fixes it.

The same configuration runs with **both GDN states stored at bf16** instead of
fp32 (`mamba_state_dtype=bfloat16`), on the same devices and geometry:

```
RESULT {"tensor_parallel_size": 8, "visible_cores": "32-39",
        "prompt_length": 32, "token_ids": [449, 731, 154, 695, 337, 385, 413, 850]}
```

Three of eight tokens match the fp32-state run before the sequences diverge,
which is what a rounded state should do on random weights at this scale -- the
top1/top2 logit gap here is ~0.04, so nothing about token identity is evidence
either way (see the caveat above). The claim this run supports is narrow and
structural: **both storage dtypes compile and execute at TP=8**, with no
`aot_autograd` view-mutation error and no `nc_matmul` operand-dtype error.
Whether bf16 storage is numerically acceptable is a real-checkpoint question.

The fixture uses random weights, so this establishes *structure*, not accuracy --
the DeepSeek-V4 record is explicit that tiny-model logits cannot settle numerical
questions (`deepseek-v4-tiny-tp1-neuron-investigation.md:36-43`). Two passing
configurations produced different token sequences after the first token, which is
expected at this scale but means **the NKI conv's numerics have not been verified
against torch on device** -- only that both compile and run.

## Root cause: a grid=1 NKI launch on an LNC=2 host

**Resolved.** The chunk-scan kernel fails `neuronx-cc` codegen whenever it is
launched with **one** program on a host running `logical-neuroncore-config 2`:

```
[INTERNAL_ERROR] [NCC_IXGM002] Expected function sg0000 in subgraph 0 to have
49 basic blocks, but on core 1 it has 1 basic blocks
```

Under LNC=2 a logical NeuronCore is **two physical NeuronCore-v3 sharing one
address space** (Trn2: 8 physical -> 4 logical). Codegen is emitted and checked
per physical core. A kernel launched at grid 1 puts its body on core 0 and
leaves core 1 with a stub -- which is exactly 49 basic blocks against 1.

The launch grid is data-dependent, and that is what made this look like a
property of the model rather than of the kernel:

```python
def scan_lnc(rows: int) -> int:      # rows = batch x heads
    return 2 if rows % 2 == 0 else 1
```

With `num_seqs_buckets: [1]` and 24 value heads on the fixture, `rows` is just
the per-rank value-head count, so the grid is decided by the TP degree:

| TP | v heads/rank | rows | grid | outcome |
|---|---|---|---|---|
| 8 | 3 | 3 | **1** | fail (`NCC_IXGM002`) |
| 4 | 6 | 6 | 2 | **pass** |
| 2 | 12 | 12 | 2 | (not run) |

Isolated with a control that varies **only** the grid, at fixed TP=4, fixed
fixture, scan kernel enabled in both:

| TP | grid | source | outcome |
|---|---|---|---|
| 4 | 2 | natural (`rows=6`) | pass -- tokens generated |
| 4 | 1 | `scan_lnc` forced to 1 | fail, error identical to TP=8 |

So TP is not the variable; the grid is. The depthwise conv kernel is never
implicated because it hardcodes `_wrapped_depthwise_conv1d[_LNC]` with
`_LNC = 2` -- it cannot launch at grid 1.

### Two earlier conclusions in this document were wrong

Both are corrected above, and both are worth recording because the reasoning
failed in an instructive way.

1. *"The LNC=2 core split was tested and disproved."* The test forced
   `scan_lnc` to return **1** and observed the error reproduce. But grid 1 is
   the *failing* value, so that experiment could only ever reproduce. The
   informative direction was forcing it to **2**, which requires an even `rows`
   and is why TP=4 was the run that settled it. A hypothesis test that cannot
   fail is not a test.

2. *"The decode graph is what fails."* The failure is reported at the decode
   extraction, but the log says plainly: *"Neuron errors are reported
   asynchronously. The Python traceback shows where the error was detected
   (synchronization point), not necessarily where the failing operation was
   dispatched."* Prefill logs `Successfully extracted` -- extracted, not
   compiled -- ~30s before the decode step reports the failure. Prefill is
   where the grid=1 scan kernel actually lives; decode takes the recurrent
   path and never calls the scan. The "decode fails even though decode does
   not use the scan" paradox was an artifact of async reporting.

### Fix

Launch the scan at a fixed grid of 2, as the conv kernel already does, and pad
`rows` to even so the grid is never data-dependent. That removes the failure
mode by construction rather than avoiding it, and makes the kernel usable at
TP=8 instead of permanently on its torch fallback. Not yet implemented -- it
changes kernel input shapes and wants review.

Until then, `can_use_chunk_scan_kernel` returning False keeps the scan on its
torch fallback; that is the conv-on/scan-off configuration both TP=8 device
runs used.

Reproduce either side: `tools/qwen3_5/generate_tiny.py` at TP=8 (fails) or
TP=4 (passes), same fixture, scan kernel enabled.

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

## Both GDN states share one storage dtype

`get_kv_spec` declares the conv window and the recurrent state at **the same**
dtype, chosen by `Qwen3_5TextConfig.mamba_state_dtype` (`float32` by default,
`bfloat16` supported). It is one knob and not two on purpose.

**Why they cannot differ.** The runner carves both states out of a single raw
page as two strided views over one storage (`neuron_model_runner.py:8546`), and
the model mutates both in place -- that in-place write *is* how the state
survives to the next step. Tracing rejects the pair:

```
AssertionError: aot_autograd() does not yet handle input mutations on views
with different dtypes.
```

**Casting does not dodge it.** This is worth stating because it is the obvious
first move and it does not work. The rejected condition is *two differing-dtype
views of one storage, both mutated* -- a property of the storage, not of any
value. Whatever you cast, the write must still land back in the view, so the
condition is unchanged. Only two things remove it: make the dtypes equal, or
stop sharing the storage.

Note this is a **tracer** limitation, not a layout one. vLLM's own `MambaSpec`
takes a per-state `dtypes` tuple (`kv_cache_interface.py:629-646`) and
`kda_state_dtype` ships a genuinely mixed `(model_dtype, float32)` pair, on the
identical carving (`gpu_model_runner.py:7160-7181`). It works there because the
state writes happen inside custom kernels the tracer never enters.

**Which dtype.** fp32 keeps the recurrent accumulator exact and is the
conservative default here. bf16 is what **vLLM itself defaults to** for this
architecture: `gated_delta_net_state_dtype` -> `_mamba_state_dtype` with
`mamba_ssm_cache_dtype="auto"` makes the recurrent state follow the model dtype,
and fp32 is opt-in via `--mamba-ssm-cache-dtype float32`. So an earlier note in
this document -- that the recurrent state "has to be fp32" -- was stricter than
upstream and is corrected here.

Internal arithmetic is fp32 either way: `chunk_gated_delta_rule`,
`recurrent_gated_delta_rule` and the NKI path all upcast the incoming state on
entry. The storage dtype therefore rounds the state once per step; it does not
degrade the arithmetic.

The trade is a straight halving of the state page. Against the real 27B that is
the difference between paying +1.9% (matching upward to fp32) and saving ~49%
(matching downward to bf16) -- so bf16 is worth measuring on a real checkpoint
before fp32 is left as the default. Giving each state its own allocation would
lift the restriction entirely and allow a true mixed pair, but that changes
cache accounting and is left open.

Both settings are covered by tests -- the declaration invariant and the page
halving in `test_qwen3_5_state_cache.py`, a prefill->decode round trip through
the real write-back seam in `test_qwen3_5_paged_state.py` -- and both are
verified on device at TP=8 (see Result above).

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
