# DeepSeek-V4 Trainium completion runbook

This runbook covers the gates that cannot be closed in the repository's current
local environment. T0, the NKI simulator, backend-config lowering, exact cache
declaration/binding, and the local regression suite are already green. Full
FX-to-HLO/NEFF capture is also included here because the available local
`torch_neuronx` package pins Torch 2.9 while vLLM 0.26 pins Torch 2.11.

## Required hardware and revisions

- A Trn2 instance for execution and device-memory measurements.
- Two isolated environments for the shipped-model regression:
  - commit `be0def6`, vLLM 0.21;
  - the current `release-0.26.0.1.0.0` revision, vLLM 0.26.
- The exact model revisions and prompts must be identical on both sides.
- Store model weights in a shared immutable location; never allow a floating
  model revision during a comparison.

Create one artifact directory per run:

```bash
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
artifact_dir="artifacts/deepseek-v4/${run_id}"
mkdir -p "${artifact_dir}"
git rev-parse HEAD > "${artifact_dir}/git-revision.txt"
python --version > "${artifact_dir}/python-version.txt"
python -m pip freeze > "${artifact_dir}/pip-freeze.txt"
neuronx-cc --version > "${artifact_dir}/neuronx-cc-version.txt" 2>&1
neuron-ls > "${artifact_dir}/neuron-ls.txt" 2>&1
```

Every gate must retain the command, environment, complete log, bucket
configuration, NEFF inventory and sizes, numerical outputs, comparison report,
wall time, host peak RSS, and Neuron device peak memory. A green terminal with
none of these artifacts does not close a gate.

Before spending device time, verify the SDK environment imports the complete
compile stack without changing the vLLM Torch pin:

```bash
python - <<'PY'
import torch
import torch_xla
import torch_neuronx
assert torch.__version__.startswith("2.11."), torch.__version__
print(torch.__version__, torch_xla.__file__, torch_neuronx.__file__)
PY
```

If that assertion or either import fails, stop and select a compatible Neuron
DLAMI/DLC. Do not downgrade Torch in the vLLM 0.26 environment to make the
compiler import: that would invalidate the compatibility and regression gates.

## Campaign 1: shipped-model regression

The pre-upgrade baseline was not captured before the dependency bump. Recover
it first; a 0.26-only run is a smoke test, not a regression test.

### Baseline side

1. Check out `be0def6` in a separate worktree.
2. Install its pinned dependencies and Neuron SDK.
3. Serve GPT-OSS and Qwen3-VL independently using the production topology,
   excluding DI unless the baseline being measured is explicitly a DI baseline.
4. For each pinned prompt, save full token-by-token logits (or the agreed
   deterministic top-N logit tensor), generated IDs, seeds, sampling parameters,
   bucket selection, and peak memory.

### Candidate side

Repeat the identical matrix on vLLM 0.26. Compare:

- tensor shape and finiteness;
- maximum and percentile absolute/relative logit error;
- top-token and top-N agreement;
- first divergent token and its complete logits;
- latency and peak-memory regressions.

The acceptable tolerance must be chosen and recorded before inspecting the
candidate results. Top-token agreement alone is insufficient. If only the
candidate side can be run, label the artifact `smoke-only` and leave P0.5 open.

## Campaign 2: 512-d MLA and tiny-model execution

First run the recorded T2 graph-capture matrix and retain its HLO, compiler log,
NEFF, compile duration, and graph size. This closes P2.c only if every required
prefill/decode bucket produces a NEFF. Then, for every bucket:

1. Load the NEFF on Trn2.
2. Run the stored P2 T0 input tensors.
3. Compare output against the fp32 reference and BF16 baseline.
4. Record warm-up separately from steady-state latency.
5. Record device peak memory and any graph/runtime warnings.

Then run the P6 tiny multi-layer model matrix:

- each structural layer combination;
- multiple prefill chunkings of the same prompt;
- prefill followed by decode;
- reorder, completion, and abort cases;
- supported TP/SP/EP layouts.

This closes P2.d and the T3 portion of P6 only when numerical comparisons and
the full artifacts exist. Successful NEFF loading by itself is not sufficient.

## Campaign 3: memory calibration and full model

### P7b calibration

Run the already-supported GPT-OSS loading path used to calibrate P7a. Capture
host RSS and device memory over time, including source tensors, conversion,
final shards, graph loading, activations, collectives, and minimum useful cache
allocation. Record the delta from P7a and update its estimated ranges rather
than replacing estimates with one measured scalar.

### P8/P9 ordering

- If measured BF16 headroom is adequate, run P8 before P9.
- If it is not, run P9 first and retain BF16 only for component references.

For both paths scale progressively: one component, one decoder layer, selected
heterogeneous layers, then the full checkpoint. Abort before OOM when measured
headroom crosses the documented safety margin.

The full-model gate requires prefill and decode logits, chunk invariance,
TP/SP/EP routing, enforced dense-CSA admission, measured memory inside budget,
and loud rejection of prefix caching, MTP, and DI. P9 additionally requires
native checkpoint FP8/FP4 storage with no full BF16 expansion and a measured
memory reduction.

## NIXL, DCP, and the release gate

The current 0.26 build intentionally rejects `NeuronNixlConnector`. Restore the
0.26 pull/push connector split before testing 1P1D or DCP-dependent layouts.
After the port is locally unit-tested, validate on separate producer/consumer
workers:

- transfer and resume for Full and SWA caches;
- heterogeneous group registration and byte accounting;
- abort and retry behavior;
- DCP block-table sizing;
- byte-exact transferred state and subsequent logits.

Until 1P1D passes, publish only as experimental/pre-release with DI explicitly
unsupported. Do not weaken the current rejection to make a topology start.

## Final sign-off checklist

- [ ] 0.21 baselines and 0.26 comparisons for GPT-OSS and Qwen3-VL
- [ ] P2 prefill/decode NEFF execution and numerical comparison
- [ ] P6 tiny-model T3 matrix
- [ ] GPT-OSS P7 calibration delta
- [ ] BF16/quantized ordering decision based on measured memory
- [ ] Full P8 or P9 correctness and memory gate
- [ ] Native FP8/FP4 P9 gate
- [ ] NIXL 1P1D/DCP validation or experimental-release designation
- [ ] Every run includes the required reproducibility artifacts
