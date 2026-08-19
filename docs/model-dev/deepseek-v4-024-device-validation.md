# DeepSeek-V4 on vLLM 0.24: device validation

How to validate the `deepseek-V4-on-0.24` branch on a Trainium instance, in
dependency order, and what each step does and does not prove.

This is the entry point for the 0.24 port. Once the ladder below is green,
[`deepseek-v4-trainium-completion.md`](deepseek-v4-trainium-completion.md)
covers the longer-horizon completion campaigns (shipped-model regression, memory
calibration, full-checkpoint BF16/FP8). Where the two disagree on vLLM versions
or the NIXL connector, this document is current and that one is not.

## Read this before booking instance time

**DeepSeek-V4 cannot be served, and this is not a testing gap.** The
scheduler-integrated inference path does not exist yet: the model's `forward`
takes no `attn_metadata`, `bind_kv_cache` validates without binding, the forward
pass is a Python loop over single tokens, and there is no parallelism anywhere in
the package. Registering it in `vllm_neuron/model/registry.py` would advertise
support that isn't there. See
[`deepseek-v4-serving-roadmap.md`](deepseek-v4-serving-roadmap.md) for the full
gap analysis and what has to be built.

Everything below validates components, cache declarations, and the kernel —
which is exactly what the current implementation is *for*. Read the status table
as "how far the reference implementation is verified", not "how close serving
is".

**What is actually verified today:**

| Tier | Scope | Status |
| --- | --- | --- |
| T0 | Bare interpreter, no torch/vLLM/Neuron | Green — 226 passed, 4 skipped (Linux/Trn2 host; `test_factory.py` included) |
| T1 | vLLM 0.24 installed, still CPU | Green — 88 passed, 2 skipped, after 4 test-vs-0.24-API fixes (see below) |
| T1-sim | NKI CPU simulator | Green — 2 passed |
| T3 | Trn2 device | 5a/5b green on real Trn2 silicon; 5c blocked on an SDK gap (see Step 5c) |
| T3 (full model) | Real `vllm.LLM()` on Trn2, compiled (`enforce_eager=False`) | Every data-dependent-shape blocker in this plugin's model code is now fixed and device-confirmed (`position_ids`, `scatter_paged_latent`, `_swa_history`, `_carry_rows`, `_compressed_history`) -- full-model FX graph capture now succeeds on the real compile backend for the first time. Blocked past that on a new, different-in-kind issue: a deterministic segfault inside torch_xla's PjRt execution runtime during actual on-device graph execution, not a Dynamo/tracing/model-code problem (see Step 5d) |
| T3 (full model, eager) | Real `vllm.LLM()` on Trn2, `enforce_eager=True` | Not possible on this plugin at all, any model (see Step 5e) -- so 5d's blocker is the only path to real-hardware inference |

The port was developed on macOS, where vLLM cannot be installed. Every claim
about vLLM 0.24's API surface was established by reading the 0.24 and 0.26
wheels, not by importing them. T1 is the first step that tests that reading.

**T1 fixes applied when this ladder was first run on Trn2** (all porting
mistakes in the tests themselves, per the "fix the assertion, don't delete the
test" rule below): `KVCacheManager.__init__` takes `max_num_batched_tokens`,
not `max_in_flight_tokens`; `KVCacheManager.remove_skipped_blocks` takes two
args, not three; the communicator class is `NeuronDeviceCommunicator`, not
`NeuronCommunicator`; and `tripwire:parallel_config_all2all_literal` is
correctly *absent* from `applied_patches()` on 0.24 (the docstring on
`TestAll2AllBackendSelection` already said this — the tripwire assertion just
hadn't been flipped to match).

## Step 1 — environment gate

Do this before anything else. It is the first honest signal on the torch
question that killed the 0.26 attempt.

```bash
git clone -b deepseek-V4-on-0.24 <your-fork> vllm-neuron
cd vllm-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
pip install -r requirements/test.txt
```

`requirements/core.txt` declares `libtorch-neuronx-lite` **unpinned**, with the
comment "Left unpinned to resolve based on vLLM's torch requirement". That is
AWS's fix for the 0.26 blocker — on 0.21 the compile stack was vendored in-tree
and dragged in a `torch_neuronx` pinned to Torch 2.9 while vLLM wanted 2.11.
Whether it genuinely resolves is a question only the resolver can answer:

```bash
python - <<'PY'
import torch, vllm, libtorch_neuronx_lite
print("torch  ", torch.__version__)
print("vllm   ", vllm.__version__)
print("libtnl ", libtorch_neuronx_lite.__file__)

from libtorch_neuronx_lite.compile.platform import get_platform_target
print("target ", get_platform_target())

# These two imports were re-pointed during the port and have never executed.
from libtorch_neuronx_lite.nki.nki_compile import compile_nki
from libtorch_neuronx_lite.nki.nki_cpu_sim import simulate_nki_kernel
print("re-pointed NKI imports OK")
PY
pip show torch libtorch-neuronx-lite | grep -E '^(Name|Version)'
```

**If the two NKI imports fail**, the symbols moved somewhere else inside
`libtorch_neuronx_lite` and two files need their import lines corrected:
`vllm_neuron/model/deepseek_v4/nki_mla.py` and
`tools/deepseek_v4/compile_p2_nki.py`. That is a rename, not a design problem.

**If `torch` resolves to a version the Neuron runtime rejects**, stop. Do not
pin torch by hand to force it — that reintroduces exactly the mismatch this port
exists to escape, and invalidates every gate below.

## Step 2 — T0, on the device host

Confirms the checkout and interpreter are sane before any dependency is trusted.
Costs seconds.

```bash
pytest test/unit -q --ignore=test/unit/model/deepseek_v4/test_factory.py
```

Expect **173 passed, 12 skipped**. The 12 skips are `importorskip("torch")` and
`importorskip("vllm")` guards, which on a Trainium host will now *not* skip —
so expect more passes and fewer skips here than on a laptop. `test_factory.py`
is excluded only because it imports through the `vllm_neuron` package `__init__`,
which needs vLLM; on this host you can drop the `--ignore` and it should pass.

A failure here is a porting mistake, not a version difference. The modules this
tier covers — `config.py`, `dense_csa.py`, `factory.py`, `memory_budget.py`,
`registry.py`, `guards.py`, `scheduler_selection.py` — have zero vLLM coupling.

## Step 3 — T1, the real port gate

```bash
pytest test/vllm_neuron -q
```

This is where the port is actually tested, and where most remaining work will
surface. Three areas are load-bearing:

**`test_upstream_compat.py`** asserts the shape of vLLM internals the plugin
patches. Its field sets were pinned against 0.26 and re-derived by reading, not
running — expect failures in `TestPortedUpstreamSurfaces` around
`CachedRequestData`, `SchedulerOutput`, `CachedRequestState`, `Request`, and
`Scheduler._update_after_schedule`. Each failure names the symbol and what
changed; fix the assertion to match 0.24, do not delete the test.
`test_validated_version_matches_installed` should pass —
`VALIDATED_VLLM_VERSION` was set to `0.24.0`.

**`TestTripwiresRunClean`** executes every registered tripwire against the
installed vLLM. This is the real verdict on whether the patch registry's
assumptions hold at 0.24. Three tripwires remain (scheduler default detection,
termination timeout targets, `in_the_same_node_as`); the all2all pydantic
tripwire was removed because 0.24 designed that patch away.

**`test_input_batch_params.py` and `test_kv_spec_conversion.py`** cover the
three API deltas the port had to bridge:

- `slot_mapping_modes` — dropped; 0.24's `InputBatch` does not accept it.
- `max_num_blocks_per_req` — not a `KVCacheSpec` method until 0.26, so it is
  now derived with the same formula `MultiGroupBlockTable` uses for its own
  default. **If one number is wrong here, block tables are silently mis-sized
  and every downstream cache result is meaningless.** Check the heterogeneous
  multi-group assertions specifically.
- `RSWASpec` — absent from 0.24, so `CacheKind.RSWA` now raises
  `NotImplementedError`. Only the synthetic model declares R-SWA.

`test_deepseek_cache_lifecycle.py` exercises the P1 matrix against vLLM's real
`KVCacheManager`. It is the strongest evidence available short of device time
that the heterogeneous cache declarations are coherent.

## Step 4 — T1-sim, the NKI simulator

```bash
NKI_SIMULATOR=1 pytest test/vllm_neuron/test_deepseek_v4_nki_simulator.py -q
```

Opt-in by env var; skips silently otherwise. Runs the 512-wide MLA kernel
through NKI's CPU simulator against the fp32 oracle. Closes the numerical
question without a device, so run it before booking one.

## Step 5 — T3, device diagnostics

All three tools are standalone: no vLLM import, no checkpoint, no scheduler.
That is what makes them runnable while the model stays unregistered.

Set up an artifact directory first — a green terminal with no artifacts does not
close a gate:

```bash
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
artifact_dir="artifacts/deepseek-v4/${run_id}"
mkdir -p "${artifact_dir}"
git rev-parse HEAD          > "${artifact_dir}/git-revision.txt"
python -m pip freeze        > "${artifact_dir}/pip-freeze.txt"
neuronx-cc --version        > "${artifact_dir}/neuronx-cc-version.txt" 2>&1
neuron-ls                   > "${artifact_dir}/neuron-ls.txt" 2>&1
```

### 5a. Compile the P2 buckets

```bash
python tools/deepseek_v4/compile_p2_nki.py \
  --output "${artifact_dir}/p2-compile" \
  --max-context 512
```

Emits reproducible JSON for the four `P2_REPRESENTATIVE_BUCKETS`. This is the
first thing to exercise the re-pointed `compile_nki` import under real load.

### 5b. Execute the 512-d MLA kernel

```bash
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2   # or gen3; the script requires one
export NEURON_LOGICAL_NC_CONFIG=2
python tools/deepseek_v4/run_p2_nki_device.py \
  --output "${artifact_dir}/p2-device" \
  --lnc 2 --warmup 1 --iterations 3
```

Synthetic BF16 inputs, decode and causal-prefill at `head_dim=512`, compared
against an fp32 NumPy oracle, with every input and output saved to `tensors.npz`
for off-device re-checking. Warm-up is recorded separately from steady state.

**Proves:** the kernel runs on real silicon and matches the oracle.
**Does not prove:** FX→HLO/NEFF graph capture, scheduler integration, or
anything about the full model.

### 5c. Tiny structural model

```bash
python tools/deepseek_v4/run_p6_tiny_device.py \
  --output "${artifact_dir}/p6-tiny-device" \
  --tokens 1 4 16
```

Runs the portable mHC, attention, compressor and MoE primitives eagerly through
Torch-XLA. Deliberately not a benchmark — Python-side expert dispatch fragments
it into many small XLA graphs. Its job is to expose unsupported component
operations before a captured graph exists.

**Currently blocked on this SDK release, and it is an SDK gap, not a script
bug.** `torch_xla/__init__.py` in this build hardcodes
`_found_libneuronxla = False` ("Neuron library initialization is handled by
neuronx-cc package directly"), so `torch_xla.device()` silently resolves to the
CPU PJRT pseudo-device instead of `NEURON` — installing `libneuronxla` does not
change this, since the auto-detection path itself is compiled out, not merely
unmet. The script now checks `torch_xla.runtime.device_type() == "NEURON"`
before running and raises `SystemExit` if it isn't, rather than silently
mislabeling a CPU run as a device gate. The real Neuron dispatch path on this
release goes through `libtorch_neuronx_lite`'s `torch.compile` backends
(`neuron_libtorch`, `neuron_libtorch_graph_capture`) and `nkilib`'s own kernel
entry points — both exercised successfully by 5a/5b above. Eager per-op
dispatch via `torch_xla.device()` has no working path to hardware here;
redesigning P6 onto the compile-backend path is new integration work, not a
retest, and belongs with Step 1 of
[`deepseek-v4-serving-roadmap.md`](deepseek-v4-serving-roadmap.md).

### 5d. Full model graph capture (registry-gated, `vllm.LLM()`)

The compile-backend path 5c names as the real Neuron dispatch route
(`neuron_libtorch`/`neuron_libtorch_graph_capture`) is exactly what
`neuron_model_runner.py` uses for a real (non-`enforce_eager`) run. This step
attempts it for the first time against the **full registered model** — not
the isolated 512-d kernel (5a/5b) or a hand-driven eager loop (5c blocked) —
using the same tiny synthetic config as
`test/vllm_neuron/test_deepseek_v4_device_e2e.py`, but with
`VLLM_NEURON_CPU_MODE` unset and `enforce_eager=False`, on real Trn2 silicon:

```bash
export VLLM_NEURON_ENABLE_DEEPSEEK_V4=1
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_SKIP_EFA_AFFINITY=1   # this instance has no EFA device
python - <<'PY'
from vllm import LLM, SamplingParams
# ... build LLM(model=<tiny checkpoint dir>, load_format="dummy",
#     enforce_eager=False, ...) and call llm.generate(...)
PY
```

**Result: real, incremental progress, not yet a green compile.** Three
attempts, each one exercising real `torch.compile`/Dynamo tracing through
`libtorch_neuronx_lite`'s parallel-trace fork on the actual device:

1. **`RuntimeError: No EFA device found`** — an environment fact, not a
   model bug: this `trn2.3xlarge` has no EFA interface, and
   `neuron_worker.py::_set_efa_affinity` doesn't skip its lookup without
   `NEURON_SKIP_EFA_AFFINITY=1`. The error message names its own fix.
2. **`torch._dynamo.exc.Unsupported: Data-dependent branching`** in
   `mhc.py::sinkhorn_positive`'s `if (x < 0).any(): raise ValueError(...)`.
   Dynamo cannot trace a Python `if` on a tensor value ("this graph break is
   fundamental... use `torch.cond`"). This is a pure input-validation guard
   with no effect on the numerical result along the non-raising path, so it
   was fixed by skipping it under `torch.compiler.is_compiling()` rather
   than removing the check for eager callers. The same pattern was fixed at
   `attention.py::gather_paged_latent`'s block-table bounds check (same
   `.any()`-on-`if` shape, not yet reached by tracing but certain to break
   the same way), and `model.py::DeepseekV4MoE.forward`'s
   `if not bool(mask.any()): continue` per-expert dead-computation skip was
   made unconditional (`bool(tensor)` is a device→host sync and an
   unconditional graph break; the skip was already a throughput
   optimization only — masked contributions are exactly zero either way, so
   always computing is numerically identical, just denser). All three fixes
   are in `mhc.py`, `attention.py`, and `model.py`; CPU-mode
   (`test/unit test/vllm_neuron`) stays green after them.
3. **`torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake
   tensor call`** in `DeepseekV4Attention._forward_one_token`
   (`model.py:485-486` at the time of this attempt; fixed below, so the
   line numbers have since moved):
   `position_ids = hidden.new_tensor([[cached_seq_len]], dtype=torch.long)`
   followed by `self.rotary_emb(hidden, position_ids=position_ids, ...)`.
   `cached_seq_len` is a per-token-loop-local Python `int` (derived once per
   chunk from `attn_metadata` via `int(...)`, then advanced by plain integer
   addition per loop iteration — see `_forward_one_token`'s caller). Under
   Dynamo's FakeTensorMode, constructing a fresh tensor from a nested Python
   list around a traced scalar (`[[cached_seq_len]]`) and then indexing/
   unsqueezing it inside `rotary_emb` trips
   `AssertionError("Please convert all Tensors to FakeTensors first...")`
   on `aten.unsqueeze.default`. Unlike the two guard-clause breaks above,
   this is not a removable validation check — it is how every per-token
   iteration computes its own absolute RoPE position, load-bearing for
   correctness.

Artifacts: `artifacts/deepseek-v4/<run-id>/p6-full-model-compile/` holds the
three full logs (`attempt1-efa-failure.txt`,
`attempt2-sinkhorn-graph-break.txt`,
`attempt3-position-ids-faketensor-error.txt`).

**Root cause, fix, and confirmation on real Trn2 silicon (attempts 4-5).**
`cached_seq_len` reaches `_forward_one_token` as a Python `int`, but under
Dynamo it is not a plain int — `int(fake_tensor)` on the `attn_metadata`
scalar produces a symbolic value, and embedding that inside a fresh
`[[cached_seq_len]]` Python list handed to `Tensor.new_tensor(...)` goes
through the same eager list-construction path `torch.tensor(...)` uses,
which FakeTensorMode does not track as a proper proxied tensor — so the
very next real op on it (`rotary_emb`'s internal `unsqueeze`) sees a tensor
FakeTensorMode never faked, tripping the "Please convert all Tensors to
FakeTensors first" assertion. `_forward_one_token` and its caller now build
`position_ids` without that round trip: the caller slices `attn_metadata`'s
`cached_seq_len` tensor directly (never calling `int()` on that copy) and
threads it into the loop as a real tensor, advanced by tensor-scalar
addition (`base_position_id + local_index`) and reshaped with `.view(1,
1)` — slice, add, view are all real proxied ops Dynamo traces natively, so
the symbolic value never gets re-embedded in a Python list.

Re-running attempt 3 against this fix, on real Trn2 silicon
(`trn2.3xlarge`, same tiny-config `vllm.LLM()`/`enforce_eager=False` setup),
**confirms the fix**: tracing gets cleanly past `position_ids`/`rotary_emb`
and well beyond, into the decoder-layer/attention/cache-write call chain —
real, forward progress, not a different flavor of the same error.

4. It landed on a new error one call deeper:
   `torch._dynamo.exc.UserError: Could not guard on data-dependent
   expression Eq(u0, 0)`, caused by `scatter_paged_latent`'s
   `if slot_mapping.numel() == 0: return` (`attention.py`). `slot_mapping`
   at that point has already been filtered by a boolean mask
   (`slot_mapping[valid]`) a few lines up — a data-dependent (unbacked
   SymInt) size by construction, and the file's own docstring already
   called this out: "costs a data-dependent shape ... fine here, this pass
   is eager/CPU-mode." Branching on it is exactly the "data-dependent
   branching" pattern the two guard-clause fixes above (`mhc.py`,
   `attention.py::gather_paged_latent`) already fixed elsewhere in this
   file. Unlike those two, this branch isn't a validation check, it's a
   throughput short-circuit — `index_put_` with an empty (or
   data-dependent-sized) index is already a well-defined no-op — so it was
   fixed the same way as `model.py`'s `if not bool(mask.any()): continue`
   fix from attempt 2: removed, always executing the (possibly empty)
   write. CPU-mode suites stay green (346 passed, 7 skipped) after this
   change.
5. Re-running again with that second fix reached a **third** blocker, this
   time a real, not-yet-fixed one:
   `Could not guard on data-dependent expression Eq(u0, 0)` again, now from
   `if cached_seq_len == 0:` in `_swa_history` (`model.py`). Unlike attempt
   4's, this one is not a removable/unconditional-izable guard clause —
   `cached_seq_len` genuinely varies per request per step (it is the live
   KV-cache length), Dynamo does not treat it as a compile-time constant,
   and this function's job is to gather a **runtime-length-dependent
   slice** of the SWA cache (`gather_len = min(cached_seq_len,
   self.sliding_window)`, then `gather_paged_latent(..., gather_len)`) —
   genuinely shape-determining, not a validation nicety. `_compressed_history`
   and the compressor's carry-window arithmetic
   (`carry_gather_length`/`carry_replay_already_emitted` in
   `compressor.py`, and `DeepseekV4Compressor._carry_rows`) share the same
   shape problem: a Python-int `cached_seq_len` driving a slice length via
   `min`/`//`/`%`. This is the real remaining redesign, not a guard clause:
   gather a fixed, compile-time-static length (`self.sliding_window`, the
   compressor's max carry width) unconditionally on every call, and use
   `mla_attention_reference`'s existing position-based masking (`kpos`/
   `qpos`, already there for causality and the sliding-window cap) —
   extended to also mask off rows beyond the real `cached_seq_len`, since
   today it assumes the supplied tensor's length *is* the true history
   length — rather than varying the gathered tensor's actual size.

   **This is where a real, separate, pre-existing correctness bug was found
   while designing that redesign — see
   [`deepseek-v4-swa-null-block-bug.md`](deepseek-v4-swa-null-block-bug.md)
   for the full account.** `_swa_history`'s `block_table[:required]` read
   (via `gather_paged_latent`) silently returned null-block content instead
   of real history once `cached_seq_len` exceeded one `sliding_window` —
   confirmed by direct reproduction, not caught by any existing test.
   `_carry_rows` (the compressor's `state_cache`) had the identical bug;
   `_compressed_history` (the non-evicting `mla_cache`) did not.

   **That correctness bug is now fixed** (`gather_paged_latent` takes a
   `start_token` offset; see the bug doc's status block) — but only the
   correctness half. It was fixed with a variable-length gather driven by a
   Python-int `cached_seq_len`, the same dynamic-shape behavior as before,
   so it does **not** close this attempt-5 Dynamo blocker: `_swa_history`'s
   `if cached_seq_len == 0:` and its runtime-length-dependent
   `gather_paged_latent` call are unchanged in shape, and the fixed-size
   gather + extended `mla_attention_reference` masking redesign described
   above is still real, not-yet-attempted work.

6. **Confirmed directly on real Trn2 silicon, not just predicted -- then
   fixed, advancing to a new (7th) blocker.** First a CPU-only proxy
   (`torch.compile(fullgraph=True, dynamic=True)` on the default `eager`
   backend, no hardware) reproduced the identical `Could not guard on
   data-dependent expression Eq(u0, 0)` error at the identical
   `_swa_history` line. Then this exact attempt was re-run for real --
   `VLLM_NEURON_ENABLE_DEEPSEEK_V4=1`, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`,
   `NEURON_LOGICAL_NC_CONFIG=2`, `NEURON_SKIP_EFA_AFFINITY=1`, same
   tiny-config `vllm.LLM()` / `enforce_eager=False` setup as attempts 1-5
   above, on an actual `trn2.3xlarge` -- after landing the null-block
   correctness fix (see
   [`deepseek-v4-swa-null-block-bug.md`](deepseek-v4-swa-null-block-bug.md)).
   Same result: `RuntimeError: Parallel trace fork failed (rank=0): ...
   status=ERROR`, with the identical `Eq(u0, 0)` guard failure at
   `_swa_history`'s `if cached_seq_len == 0:` (`model.py:498`) as the
   underlying cause -- the real compile backend's own error surface this
   time, not just Dynamo's. This confirmed the correctness fix alone left
   this blocker completely untouched, as expected.
7. `_swa_history` was then redesigned to be Dynamo-shape-static (see the
   bug doc's "Suggested fix direction" item 2: `gather_recent_window`, a
   fixed-size gather at a tensor-derived offset, plus
   `mla_attention_reference`'s new `key_valid` masking). Re-running the
   same real-hardware attempt against that redesign **advances cleanly
   past the `_swa_history` guard failure that blocked attempts 5-6** --
   real, confirmed forward progress on the actual compile backend, not a
   proxy. It lands on a new blocker one step later: first a pure guard
   clause (`carry_gather_length`'s `if cached_seq_len < 0:`,
   `compressor.py` -- fixed the same mechanical way as the other guard
   clauses in this section, skipped under `torch.compiler.is_compiling()`),
   then a genuine one: `Could not guard on data-dependent expression
   Eq(PythonMod(u0, 128), 0)`, from `DeepseekV4Compressor._carry_rows`'s
   `if gather_n == 0:` (`model.py:215`) -- the compressor carry-state
   equivalent of `_swa_history`'s old problem, but **not** self-contained
   the same way: fixing it properly cascades into
   `compress_hca_chunk`/`compress_csa_chunk`'s internal windowing, the
   write-side slot-filtering, and the compressed-entry RoPE position math
   all needing to become consistent with a fixed-candidate-count,
   filter-on-write design -- real, substantial, separate work with
   correctness risk if rushed. See the bug doc's "Suggested fix direction"
   item 3 for the full account. Not attempted (at the time of this attempt).
8. **`_carry_rows` made Dynamo-shape-static (closes item 3 above) --
   confirmed on real Trn2 silicon, advancing to a new (8th, separate)
   blocker.** Investigation found the cascade item 3 worried about is
   smaller than feared: `DeepseekV4Compressor.forward` has exactly one call
   site (`_forward_one_token`'s per-token loop), so it always compresses
   exactly one new raw token -- never a multi-token chunk. That collapses
   the write-side risk: with a fixed `coff*ratio`-row carry window (reusing
   `gather_recent_window`, the same helper item 2 introduced, plus a new
   `carry_gather_length_tensor` tensor-valued "unconsumed" formula and a
   `carry_valid` gate-softmax mask threaded through
   `compress_hca_chunk`/`compress_csa_chunk`), `compressed` always has
   exactly `coff` static rows and the currently-completing window (if any)
   is unconditionally the *last* one -- no data-dependent count, no slicing,
   no separate slot-matching machinery. Full design, staged CPU-eager
   oracle tests (including a new eviction-past-carry-window regression test
   mirroring `_swa_history`'s, and a full real-`transformers`-module
   comparison through real paged cache I/O past eviction), and this
   real-hardware confirmation are in
   `docs/model-dev/deepseek-v4-swa-null-block-bug.md`'s updated status
   block and "Suggested fix direction" item 3.

   First a CPU-only proxy (`torch.compile(fullgraph=True, dynamic=True)`,
   same technique as item 6) confirmed `_carry_rows`'s line no longer
   appears in the trace at all -- tracing now fails one call later, at
   `_compressed_history`'s own (separate, simpler) `if num_entries == 0:`
   (`model.py:552`, `cached_seq_len // self.ratio`) instead. Then the exact
   same real-hardware `vllm.LLM()`/`enforce_eager=False` recipe as attempts
   1-7, re-run against this fix on the same `trn2.3xlarge`, reproduces the
   identical result on the real compile backend: `RuntimeError: Parallel
   trace fork failed`, with `Could not guard on data-dependent expression
   Eq((u0//128), 0)` at `_compressed_history:552` as the underlying cause --
   no mention of `_carry_rows`/`gather_n`/`compress_hca_chunk`/
   `compress_csa_chunk` anywhere in the trace, confirming the fix closes
   item 3 on real hardware, not just in the CPU proxy.

   `_compressed_history` is a genuinely separate, simpler item, not part of
   this fix: it reads the non-evicting `mla_cache` (no null-block
   correctness bug, per the bug doc's "Affected call sites" section) and
   its only Dynamo problem is the same "Python-int `cached_seq_len` driving
   a slice length" pattern `_swa_history`/`_carry_rows` already had --
   likely a small, `_swa_history`-item-2-shaped fix (fixed-length gather +
   mask), not entangled with any windowing/write-side complexity. Not
   attempted here; the next open item in this family.
9. **`_compressed_history` made Dynamo-shape-static -- closes every
   data-dependent-shape blocker in this plugin's model code. Full-model FX
   graph capture now succeeds on the real compile backend for the first
   time. Blocked one stage later by a new, different-in-kind issue: a
   deterministic segfault during actual on-device graph *execution*, inside
   torch_xla's PjRt runtime -- not a Dynamo/tracing/model-code problem.**
   Unlike `_swa_history`/`_carry_rows`, this group never evicts -- entries
   are addressed from 0 and simply accumulate, so "the first `num_entries`
   columns" is a fixed, growing *prefix*, not a sliding window. That makes
   the fix simpler than either: no tensor-derived offset needed. It always
   gathers the *entire* block-table-addressable capacity (`max_entries` --
   `block_table_row.shape[0] * self.mla_cache.shape[2]`, a plain Python int
   from real tensor shapes, no config threading needed) and masks off
   entries beyond the real current count, rather than branching on
   `cached_seq_len`'s value at all. `gather_paged_latent` itself needed no
   changes -- called with a fixed, shape-derived length, its own internal
   `if sequence_length == 0:` guard becomes an ordinary compile-time-constant
   branch, not a data-dependent one. `_forward_one_token`'s Python-int
   `cached_seq_len` parameter (and `forward()`'s `int(cached_seq_len_row[0])`
   device-to-host read that built it) are now gone entirely -- every
   consumer uses the tensor `position_ids` instead.

   Verified the same way as item 8: new CPU-eager unit tests (a direct
   gather/mask-boundary check against a hand-built paged `mla_cache` with
   known entries, for both HCA and CSA) plus the existing chunk-invariance
   and real-engine `generate()` tests, all green; then the CPU
   `torch.compile(fullgraph=True, dynamic=True)` proxy -- which now traces
   **25 steps with zero graph breaks at all**, not just past the one line
   that used to fail; then the real hardware `vllm.LLM()`/
   `enforce_eager=False` recipe. That real-hardware run is where the
   picture changes: tracing succeeds completely (FX graphs written for
   every one of the 3 parallel-trace lanes, `capture_backend.py`'s "FX Pass
   metadata" logged for each), but graph *execution* then crashes:
   ```
   !!!!!!! Segfault encountered !!!!!!!
     File "<unknown>", line 0, in torch_xla::runtime::PjRtComputationClient::ExecuteComputation(...)
     File "<unknown>", line 0, in torch_xla::XLAGraphExecutor::ScheduleSyncTensorsGraph(...)
     ...
   RuntimeError: Parallel trace fork failed (rank=0): lane=1 pid=... exit_code=-1 status=ERROR err=no status file written
   ```
   Reproduced twice, including once against a fully cleared
   `~/.cache/neuron_libtorch` compile cache (ruling out stale-cache
   corruption) -- deterministic, not flaky. HBM is not the cause (23.99 GiB
   free logged just before the crash). This is a crash inside
   `libtorch_neuronx_lite`/torch_xla's native execution path, not raised
   Python-side and not traceable to a specific line in this plugin's code --
   a toolchain/SDK-level issue, the same category as the EFA-device and
   eager-mode gaps in Steps 5d(attempt 1)/5e, not a DeepSeek-V4 model bug.
   Not investigated further here: root-causing a native segfault in the
   compile/execution runtime is a different kind of work (SDK bug report,
   toolchain version bisection, ...) than the Dynamo-shape fixes this
   document's whole thread has been about, and is out of scope for this
   pass.

Artifacts: `artifacts/deepseek-v4/<run-id>/p6-full-model-compile-retry/`
holds the two new full logs (`attempt-position-ids-fix-retry.log`,
confirming the position_ids fix and showing the new `scatter_paged_latent`
error; `attempt-scatter-guard-fix-retry.log`, confirming that fix and
showing the new `_swa_history` error). Attempt 6's real-hardware re-run
after the null-block fix is
`artifacts/deepseek-v4/20260819T022008Z-8df62fb/p6-null-block-fix-step5d-retry/attempt-post-null-block-fix.log`.
Attempt 7's two real-hardware re-runs (post-`_swa_history`-fix landing on
`_carry_rows`; post-`carry_gather_length`-guard-fix landing on the same
`_carry_rows` line) are in
`artifacts/deepseek-v4/<run-id>/p6-dynamo-shape-static-swa/`. Attempt 8's
CPU proxy and real-hardware confirmation (both landing cleanly on
`_compressed_history`, past `_carry_rows`) are in
`artifacts/deepseek-v4/20260819T035235Z-2f88686-wip/p6-carry-rows-dynamo-static-fix/`.
Attempt 9's CPU proxy (full 25-step clean trace) and both real-hardware
segfault reproductions (fresh cache and cleared cache) are in
`artifacts/deepseek-v4/20260819T040508Z-055e15f-wip/p6-compressed-history-dynamo-static-fix/`
(directory names note these ran against an uncommitted working tree at
capture time -- see each directory's `git-revision.txt`).

**What this means for Step 0's open question.** The toolchain-level question
that blocked the 0.26 attempt (Torch 2.9 vs. 2.11) is now answered as far as
Dynamo tracing goes: this is real `torch.compile`/Dynamo tracing of the
*actual, complete* registered model through the real Neuron compile backend
on real silicon, past model construction, weight loading, the device move,
and now full FX graph capture for every layer -- substantially past 5a/5b's
isolated-kernel-only evidence, and past every model-code blocker this
document's whole thread was chasing. The per-token attention loop
(`docs/model-dev/deepseek-v4-carry-cache-design.md`'s "Mid-chunk compression
boundaries force a per-token attention loop") built several tensors per
iteration from Python-level scalars in ways Dynamo's fake-tensor tracing did
not accept cleanly; every instance of that pattern (`position_ids`,
`_swa_history`, `_carry_rows`/compressor-carry-window, `_compressed_history`,
plus the two guard-clause instances in `scatter_paged_latent`/`mhc.py`) is
now fixed and device-confirmed. What appeared to replace it as the open
question was a genuinely different one: a deterministic segfault inside
torch_xla's PjRt execution runtime once graph *execution* actually starts
(item 9 above), read at the time as a toolchain/SDK-level question rather
than a "which line needs a shape fix" question.

**That reading was wrong — see item 12 below.** The segfault was a 57th
instance of exactly this same shape-fix series, hiding in
`scatter_paged_latent`'s padding filter, and it is fixed in this repo. The
lesson worth carrying: a graph that captures with *zero Dynamo graph breaks*
can still be shape-dynamic, because `fullgraph=True` converts what would
have been a break into a silently-captured unbacked SymInt. "No graph break"
was the wrong success signal to read the earlier attempts against.

#### Item 12: the segfault was model code after all — a 57th data-dependent shape

**The "toolchain/SDK-level, not fixable in this repo" conclusion above is
wrong, and this item supersedes it.** The `ExecuteComputation` segfault has a
single root cause in this repo's own model code, it is the *same* family as
every other Step 5d fix, and it is fixed in `scatter_paged_latent`
(`vllm_neuron/model/deepseek_v4/attention.py`).

**Mechanism, confirmed end to end.** `xla_builder.create_placeholder_tensor`
— which the compile backend's FX→HLO stage
(`libtorch_neuronx_lite/compile/hlo.py::convert_fx_to_hlo`) builds its replay
inputs from — returns XLA tensors with *no backing device buffer*. They are
valid only for lowering. Reduced to first principles, hardware-free:

| What is done to a placeholder tensor | Result |
| --- | --- |
| `LoweringContext().build(...)` / `.hlo()` (capture) | fine — HLO comes out |
| `torch_xla.sync(wait=True)` (execute) | **SIGSEGV** in `PjRtComputationClient::ExecuteComputation` |
| `.item()` (materialize) | **SIGSEGV**, same frame |

`convert_fx_to_hlo` does not merely lower the FX graph — it *replays* it
(`gm(*xla_placeholders)`). Replay is safe only while every op in the graph
has a data-independent output shape. Probing the op families the DeepSeek-V4
graph actually contains against placeholder inputs, exactly two segfault:
`torch.nonzero` and **boolean-mask indexing**. `index_put_`, `setitem`,
`gather`, `scatter_`, `index_select`, `embedding`, `einsum`, `where`,
`clamp`, `masked_fill`, `softmax`, `rms_norm` and the rest all replay
cleanly.

**The offending line.** `scatter_paged_latent` filtered padding rows out with
`slot_mapping = slot_mapping[valid]` / `values = values[valid]`. Its own
comment recorded the assumption that made this look safe — *"Filtering to the
valid rows costs a data-dependent shape (fine here — this pass is
eager/CPU-mode)"*. That assumption expired the moment the model reached the
device compile path. Under the `fullgraph=True` tracing
`neuron_model_runner.py` uses, Dynamo cannot break on the mask, so it
captures it as an unbacked SymInt instead — which is why the capture looked
*clean*: **no graph break is not the same as static shapes.**

The crashed run's own captured FX graph proves this is the whole story: it
holds 112 `aten._assert_scalar` guard nodes, and every one of them traces
back through 56 `aten.sym_size.int` calls to a `slot_mapping` produced by
those two mask filters. There is no other unbacked-SymInt source anywhere in
the full-model graph, and no other `nonzero`/`masked_select`/`.item()`/
boolean-mask site remains in `vllm_neuron/model/deepseek_v4/`.

**The fix.** Redirect padding rows to the reserved null block instead of
filtering them out — `slot_mapping.clamp(min=0)` sends `-1` to flat slot 0,
i.e. block 0 offset 0, which is vLLM's `NULL_BLOCK_ID`: never handed to a
real request, masked out on every read path here, and already this plugin's
write-side padding convention everywhere else (see `llama3/model.py`'s
`_write_kv_cache`, and the runner's own `slot_mapping[pad:] = NULL_BLOCK_ID`).
The write stays a single unconditional fixed-shape `index_put_`. This is the
same clamp-and-mask idiom `gather_recent_window` already uses on the read
side.

The collision hazard the old comment worried about — a padding row
redirected onto slot 0 racing a real row that also targets slot 0, with
`index_put_` leaving the winner unspecified — cannot occur, for two
independent reasons: the null block is reserved so no real row ever targets
it, and both production call sites pass **exactly one row** (`model.py`'s
per-token `[local_index : local_index + 1]` slice, and
`compressed_entry_slot_mapping`'s `[1]`-shaped result), so a padding row can
never share a call with a real row at all.

**Evidence.** A ~35-line reproduction that needs neither vLLM nor Neuron
hardware — the capture backend runs entirely on the CPU PjRt client, which is
what `PJRT_DEVICE=CPU` in `vllm_neuron/__init__.py` selects — drives the real
`scatter_paged_latent` through `torch.compile(backend=
"neuron_libtorch_graph_capture", fullgraph=True)` on meta inputs. Before the
fix it segfaults immediately after the `FX Pass metadata` log line with no
HLO written, matching the reported crash exactly; after the fix it reaches
`CaptureComplete` and the captured graph contains zero `_assert_scalar`
nodes. Regression coverage:
`test_scatter_paged_latent_is_dynamo_shape_static` (asserts the traced graph
stays data-independent) plus the two padding-contract tests in
`test/unit/model/deepseek_v4/test_paged_cache_helpers.py`; all three fail on
the pre-fix code. Full DeepSeek-V4 unit and integration suites are green
(153 + 24 passed).

**Still owed: the end-to-end hardware re-run.** The `vllm.LLM()` recipe has
not been re-run on real silicon since the fix — the instance's Neuron device
was occupied for the whole of this session by an unrelated root-owned
`vllm serve openai/gpt-oss-20b` (TP=2, both logical cores). Re-run
`artifacts/deepseek-v4/20260819T150816Z-0d84188-wip/p6-disable-parallel-trace-workaround/step5d_run_script_disable_parallel_trace.py`
once the device is free to confirm capture now completes for all buckets.

**What this leaves for the SDK.** A much narrower report than the original
draft: the capture backend should *reject* a graph whose replay needs a
data-dependent shape, rather than segfaulting in a worker thread with no
Python frame. See
`artifacts/deepseek-v4/<run-id>/p6-capture-segfault-root-cause/sdk-bug-report.md`.

### 5e. `enforce_eager=True` is not an escape hatch on real hardware

A natural question after 5d: if compiled (`enforce_eager=False`) graph
capture doesn't complete, does eager mode work on real silicon instead,
independent of Dynamo? Tried directly (same tiny config, real Trn2, no
`VLLM_NEURON_CPU_MODE`, `enforce_eager=True`) and the answer is no --
categorically, not a DeepSeek-V4-specific gap:

```
AssertionError: Eager mode on Neuron is not yet supported.
```

Raised unconditionally by `neuron_worker.py`
(`assert not (eager_mode and not cpu_mode)`,
`neuron_model_runner.py:1250`) for *every* model on this plugin, real
hardware only -- `VLLM_NEURON_CPU_MODE=1` is the only way `enforce_eager`
is accepted, and that's not real hardware at all. There is no eager
fallback to fall back to here: closing 5d's Dynamo/FakeTensor blocker is
not one of two ways to reach real-hardware inference, it is the only one.
Artifact: `artifacts/deepseek-v4/<run-id>/p7-eager-real-device/attempt1.txt`.

## What this port changed that needs device eyes

These are behavioral changes introduced by the 0.24 migration itself. They pass
CPU tests but have never touched hardware.

**MLA cache allocation switched from allocate to view.** The 0.26 branch forked
on `is_native_backend()`, allocating a fresh zeroed tensor on the native backend
and viewing the raw buffer otherwise. Upstream 0.24 removed that fork in favor of
a memoizing `_shared_dtype_view`, so the MLA branch now always views the raw
tensor. On the native backend this is a real semantic change: the latent cache is
no longer zero-initialized and now aliases the shared raw allocation. Watch for
garbage in the first decode step, or aliasing between layers pooled onto one
buffer.

**Disaggregated inference is enabled again.** The 0.26 branch rejected
`NeuronNixlConnector` loudly because 0.26 split the connector and dropped
`NixlConnectorWorker`. On 0.24 that symbol still exists, just in
`…v1.nixl.worker`, so the rejection is gone. Nothing here has exercised a 1P1D
topology — treat DI as untested rather than working.

**Block-table row lengths are now computed in-plugin.** See Step 3. This is the
single highest-consequence numeric change in the port.

## Triage

| Symptom | Likely cause |
| --- | --- |
| `ImportError` on `libtorch_neuronx_lite.nki.*` | Symbols live at a different path; fix the two re-pointed imports (Step 1) |
| `TypeError: InputBatch.__init__() got an unexpected keyword` | A kwarg delta beyond the three handled; compare against the installed `gpu_input_batch.py` |
| Tripwire `PatchError` at startup | Upstream drift at 0.24. The message names the symbol — this is the registry working as designed |
| `NotImplementedError` naming `RSWASpec` | Expected. Only the synthetic model reaches this path |
| Block tables sized wrong / cache corruption | The `max_num_blocks_per_req` derivation. Compare against `MultiGroupBlockTable`'s own default |
| First decode step returns garbage | The allocate→view change above |
| `PackageNotFoundError: torch-neuronx` from the T3 tool scripts | Expected on 0.24 — the compile stack lives in `libtorch-neuronx-lite` instead. Fixed by trying both names |
| `torch_xla.runtime.device_type()` reports `CPU` instead of `NEURON` | This SDK build compiles out `torch_xla`'s classic PJRT auto-detection. Not fixable by installing `libneuronxla`; see Step 5c |
| `RuntimeError: No EFA device found` from `neuron_worker.py::_set_efa_affinity` | Expected on an EFA-less instance (e.g. `trn2.3xlarge`). Set `NEURON_SKIP_EFA_AFFINITY=1`; see Step 5d |
| `torch._dynamo.exc.Unsupported: Data-dependent branching` compiling the full model | A Python `if` on a tensor value inside a traced module. If it is a pure validation guard, skip it under `torch.compiler.is_compiling()`; see Step 5d's fixes in `mhc.py`/`attention.py`/`model.py` |
| `AssertionError: Eager mode on Neuron is not yet supported.` | Expected on real hardware with `enforce_eager=True` -- this plugin only accepts eager mode under `VLLM_NEURON_CPU_MODE=1`, for every model, not a DeepSeek-V4 gap; see Step 5e |

## What is still out of scope

Registry exposure, end-to-end serving, FX→HLO/NEFF capture for the full model,
full-checkpoint BF16/FP8/FP4, prefix caching, MTP/speculative decoding, and
1P1D. The platform deliberately raises `NotImplementedError` for prefix caching
and speculative decoding on `deepseek_v4`, and the dense-CSA admission gate
rejects uncapped generations. Those rejections are features; do not weaken them
to make a topology start.

The work required to close the serving gap is laid out in
[`deepseek-v4-serving-roadmap.md`](deepseek-v4-serving-roadmap.md). Its Step 0 —
confirm graph capture actually works on this base — is answered by the ladder
above, and should be answered before any device model code is written.

Once this ladder is green, pick up
[`deepseek-v4-trainium-completion.md`](deepseek-v4-trainium-completion.md) at
Campaign 2. Campaign 1 in that document (shipped-model regression) needs its
baseline re-derived: it was written to bracket a 0.21→0.26 upgrade, and on this
branch upstream's own `release-0.24.0.1.1.0` is the baseline instead.
