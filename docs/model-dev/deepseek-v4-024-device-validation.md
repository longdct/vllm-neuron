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
| T3 (full model) | Real `vllm.LLM()` on Trn2, compiled (`enforce_eager=False`) | `position_ids` and one `scatter_paged_latent` blocker fixed and device-confirmed; now blocked on `_swa_history`'s data-dependent `cached_seq_len` shape (see Step 5d) |
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
   (via `gather_paged_latent`) silently returns null-block content instead
   of real history once `cached_seq_len` exceeds one `sliding_window` —
   confirmed by direct reproduction, not caught by any existing test.
   `_carry_rows` (the compressor's `state_cache`) has the identical bug;
   `_compressed_history` (the non-evicting `mla_cache`) does not. Fixing the
   Dynamo shape problem and this correctness bug are naturally one piece of
   work, since both live in the same column-selection logic — read that doc
   before starting either. Not attempted in this pass: it is real,
   non-trivial numerical-correctness surface (attention masking, cache-write
   filtering), not a mechanical trace-compatibility fix, and deserves its
   own pass with dedicated oracle comparisons rather than being rushed
   alongside two guard-clause fixes.

Artifacts: `artifacts/deepseek-v4/<run-id>/p6-full-model-compile-retry/`
holds the two new full logs (`attempt-position-ids-fix-retry.log`,
confirming the position_ids fix and showing the new `scatter_paged_latent`
error; `attempt-scatter-guard-fix-retry.log`, confirming that fix and
showing the new `_swa_history` error).

**What this means for Step 0's open question.** The toolchain-level question
that blocked the 0.26 attempt (Torch 2.9 vs. 2.11) is answered further than
any previous gate reached it: this is real `torch.compile`/Dynamo tracing of
the actual registered model through the real Neuron compile backend on real
silicon, past model construction, weight loading, and the device move —
substantially past 5a/5b's isolated-kernel-only evidence. The remaining
blockers are in this plugin's own model code, not the toolchain: the
per-token attention loop (`docs/model-dev/deepseek-v4-carry-cache-design.md`'s
"Mid-chunk compression boundaries force a per-token attention loop") builds
several tensors per iteration from Python-level scalars in ways Dynamo's
fake-tensor tracing does not yet accept cleanly. The `position_ids` instance
of that pattern is fixed and device-confirmed above, as is one guard-clause
instance in `scatter_paged_latent`; attempt 5 confirms the real remaining
instance is `_swa_history`/`_compressed_history`/the compressor's
carry-window math, which slice cache tensors to a Python-int length
computed from `cached_seq_len` (`min(cached_seq_len, self.sliding_window)`,
`cached_seq_len // self.ratio`) — genuinely shape-determining, not a
removable guard. The fix is the shape one flagged in the roadmap: gather a
fixed, compile-time-static length (the window/entry cap) unconditionally
and mask instead of slicing to a runtime-computed length, rather than
threading a symbolic int into a slice bound. It belongs with
`deepseek-v4-serving-roadmap.md`'s Step 0.

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
