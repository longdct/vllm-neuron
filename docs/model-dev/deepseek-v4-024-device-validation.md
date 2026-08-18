# DeepSeek-V4 on vLLM 0.24: device validation

How to validate the `deepseek-V4-on-0.24` branch on a Trainium instance, in
dependency order, and what each step does and does not prove.

This is the entry point for the 0.24 port. Once the ladder below is green,
[`deepseek-v4-trainium-completion.md`](deepseek-v4-trainium-completion.md)
covers the longer-horizon completion campaigns (shipped-model regression, memory
calibration, full-checkpoint BF16/FP8). Where the two disagree on vLLM versions
or the NIXL connector, this document is current and that one is not.

## Read this before booking instance time

**DeepSeek-V4 is not registered.** `vllm_neuron/model/registry.py` is unchanged
from upstream and never registers `DeepseekV4ForCausalLM`. You cannot
`vllm serve` a DeepSeek-V4 checkpoint on this branch, and no step below does.
Registry exposure is deliberately withheld until scheduler-metadata-driven cache
I/O and graph capture exist. Everything here validates components, cache
plumbing, and the kernel — not end-to-end serving.

**What is actually verified today:**

| Tier | Scope | Status |
| --- | --- | --- |
| T0 | Bare interpreter, no torch/vLLM/Neuron | Green — 173 passed, 12 skipped |
| T1 | vLLM 0.24 installed, still CPU | **Never run.** Needs Linux |
| T1-sim | NKI CPU simulator | **Never run** on this branch |
| T3 | Trn2 device | **Never run** on this branch |

The port was developed on macOS, where vLLM cannot be installed. Every claim
about vLLM 0.24's API surface was established by reading the 0.24 and 0.26
wheels, not by importing them. T1 is the first step that tests that reading.

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

## What is still out of scope

Registry exposure, end-to-end serving, FX→HLO/NEFF capture for the full model,
full-checkpoint BF16/FP8/FP4, prefix caching, MTP/speculative decoding, and
1P1D. The platform deliberately raises `NotImplementedError` for prefix caching
and speculative decoding on `deepseek_v4`, and the dense-CSA admission gate
rejects uncapped generations. Those rejections are features; do not weaken them
to make a topology start.

Once this ladder is green, pick up
[`deepseek-v4-trainium-completion.md`](deepseek-v4-trainium-completion.md) at
Campaign 2. Campaign 1 in that document (shipped-model regression) needs its
baseline re-derived: it was written to bracket a 0.21→0.26 upgrade, and on this
branch upstream's own `release-0.24.0.1.1.0` is the baseline instead.
