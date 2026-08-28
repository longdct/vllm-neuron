# DeepSeek-V4 development and test environment handoff

Status: environment snapshot from 2026-08-26. This document records the host,
software stacks, paths, cache discipline, and development workflow used for the
DeepSeek-V4 compilation investigation. It deliberately does not describe the
model architecture.

Read this together with
[DeepSeek-V4 Q512 MLA compiler graph explosion](deepseek-v4-q512-mla-compile-explosion.md).

## Non-negotiable context

- The measured DeepSeek and Qwen comparisons used the TorchNeuron Native
  development virtual environment at
  `/home/ssm-user/.venv-torch-neuronx-dev`.
- The machine also has a packaged vLLM/XLA environment under `/opt`. It is a
  different stack. Do not combine Python, libraries, or command wrappers from
  the two environments in one process.
- Neuron graph compilation is host CPU/RAM work. NeuronCores execute compiled
  graphs and warmups; assigning more NeuronCores does not make one
  `neuronx-cc` process consume all host CPUs or compile proportionally faster.
- The host has no swap. Two large TP-rank compiler processes can exhaust all
  124 GiB and be killed even if either process might survive alone.
- Use a fresh local cache for every cold comparison and the exact same cache
  for a warm-relaunch check. Cache state is part of every benchmark result.

## Host snapshot

| Item | Value |
| --- | --- |
| EC2 instance | `trn2.3xlarge` |
| Instance ID at snapshot | `i-07ffee41c3174485d` |
| OS | Ubuntu 24.04.4 LTS (Noble) |
| Kernel | `6.17.0-1019-aws` |
| Architecture | x86-64 |
| CPU | Intel Xeon Platinum 8488C |
| Host CPUs | 12 threads, 6 cores, one socket/NUMA node |
| Host RAM | 124 GiB |
| Swap | none |
| Root/local storage | 968 GiB ext4 on `/dev/nvme0n1p1` |
| Free storage at snapshot | about 819 GiB |
| Repository | `/home/ssm-user/vllm-neuron` |
| Time zone used in logs | UTC |

`/home/ssm-user` and `/tmp` are currently on the same local NVMe-backed root
filesystem. Recheck with `findmnt` and `df` after moving to another host.

## Neuron hardware and system packages

`neuron-ls` reported:

```text
instance-type: trn2.3xlarge
logical-neuroncore-config: 2
NeuronDevice 0: 4 NeuronCores (core IDs 0-3), 96 GB device memory
PCI BDF: 0000:33:00.0
CPU affinity: 0-11
NUMA node: 0
```

TorchNeuron Native reported four Neuron devices through
`torch.neuron.device_count()`. The TP2/LNC2 runs used
`NEURON_VISIBLE_DEVICES=0,1`; with the host's logical-NeuronCore configuration,
the two ranks use the four physical NeuronCores as two LNC2 execution groups.
Do not infer this mapping on a different instance. Record both `neuron-ls` and
`torch.neuron.device_count()` before running.

Installed host packages at the snapshot:

| Package | Version |
| --- | --- |
| `aws-neuronx-dkms` | `2.30.2.0` |
| `aws-neuronx-runtime-lib` | `2.34.10.0-ac18d186d` |
| `aws-neuronx-tools` | `2.32.28.0-526c2b7f6` |
| `aws-neuronx-collectives` | `2.34.10.0-74eaafac6` |
| `aws-neuronx-oci-hook` | `2.18.25.0` |
| EFA | `3.1.0-1.amzn1` |

There is one device node, `/dev/neuron0`, representing the physical
NeuronDevice. The runtime, tools, compiler, and Python packages do not all have
the same release number; record the complete set rather than reporting only
"Neuron 2.x".

## Active development stack: use this for this investigation

Virtual environment:

```text
/home/ssm-user/.venv-torch-neuronx-dev
```

It uses `/usr/bin/python3.12` through the venv and activates the TorchNeuron
Native/PrivateUse1 backend. It intentionally does not contain `torch_xla` or
`libtorch_neuronx_lite`.

| Component | Version/path |
| --- | --- |
| Python | `3.12.3` |
| PyTorch | `2.12.1+cpu` |
| `torch-neuronx` | `2.12.3.0.0+aa8779f4.dev` |
| `neuronx-cc` | `2.27.5334.0+f702b353` |
| HWM reported by compiler | `2.27.5334.0+f702b353` |
| NKI | `0.6.0+31049202112.g85070674` |
| NumPy | `2.5.2` |
| vLLM | `0.24.0` |
| vLLM Neuron package metadata | `0.24.0.1.1.0` |
| Transformers | `5.15.0` |
| Safetensors | `0.8.0` |
| pytest | `9.1.1` |
| psutil | `7.2.2` |
| Repository import | `/home/ssm-user/vllm-neuron/vllm_neuron` |

The compiler entry point used by the benchmarks is:

```text
/home/ssm-user/.venv-torch-neuronx-dev/bin/neuronx-cc
```

It is a Python console-script wrapper importing
`neuronxcc.driver.CommandDriver`. There is no compiler executable at
`/opt/aws/neuron/bin/neuronx-cc` on this host, even though that directory may
appear in the default `PATH`.

Importing this development stack currently emits this warning:

```text
Failed to import
torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.wrappers:
No module named
'torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.neg_kernel'
```

This warning was present during the recorded benchmarks. Do not silently
"repair" it by pulling packages from the `/opt` stack; either reproduce the
current environment or upgrade the development stack as a coherent unit and
record that as a new baseline.

## Packaged `/opt` stack: reference only

The AMI also provides:

```text
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
```

Its relevant versions are:

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | `2.11.0` |
| Torch XLA | `2.11.0` |
| `libtorch-neuronx-lite` | `2.11.0.1.0.1284+f49d8626` |
| `neuronx-cc` | `2.27.5334.0+f702b353` |
| NKI | `0.6.0+31049202112.g85070674` |
| NumPy | `2.3.5` |
| vLLM | `0.24.0` |
| vLLM Neuron | `0.24.0.1.1.0` |
| Transformers | `5.15.0` |

Its compiler wrapper is:

```text
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/neuronx-cc
```

The compiler version matches the development venv, but the Python execution
stack does not. A matching compiler version does not make the two environments
interchangeable.

## Selecting the development environment

Prefer explicit absolute paths in benchmark scripts. A reliable shell setup is:

```bash
export DSV4_REPO=/home/ssm-user/vllm-neuron
export DSV4_VENV=/home/ssm-user/.venv-torch-neuronx-dev
export PATH="$DSV4_VENV/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/bin:/bin"
export PYTHONPATH="$DSV4_REPO"

cd "$DSV4_REPO"
command -v python
command -v neuronx-cc
python --version
neuronx-cc --version
```

Expected paths begin with `$DSV4_VENV/bin`. Avoid relying on `source activate`
inside service, vLLM worker, timeout, or GNU-time wrappers; child processes must
inherit the correct `PATH` too.

Check backend discovery before a long compile:

```bash
python - <<'PY'
import torch
import vllm_neuron

print("privateuse1 backend:", torch._C._get_privateuse1_backend_name())
print("Neuron devices:", torch.neuron.device_count())
PY
```

The backend should be `neuron`, and repository logs should say
`Using TorchNeuron Native backend`.

## Repository and worktree state

At the snapshot:

```text
branch: deepseek-v4-tp-ep
HEAD:   b2763b2bcc7ebb267709a8d8094adc6623887810
remote: https://github.com/longdct/vllm-neuron.git
```

Relevant branch/worktree locations:

| Branch | Worktree |
| --- | --- |
| `deepseek-v4-tp-ep` | `/home/ssm-user/vllm-neuron` |
| `deepseek-v4-runtime-csa-indexer` | `/home/ssm-user/vllm-neuron-runtime-csa-indexer` |
| `worktree-ds-compile-fast-loop` | `/home/ssm-user/vllm-neuron/.claude/worktrees/ds-compile-fast-loop` |
| `deepseek-v4-lightning-indexer` | `/home/ssm-user/vllm-neuron/.claude/worktrees/ds-v4-indexer` |

The primary worktree is dirty with unrelated TorchNeuron Native backend work.
Before editing or committing:

```bash
git status --short --branch
git diff --name-only
git diff --cached --name-only
git worktree list
```

Do not reset, checkout, clean, reformat, or include unrelated modified/untracked
files. Stage only explicit paths belonging to the DeepSeek task. The two
environment/defect handoff documents were initially uncommitted at this
snapshot.

## Checkpoints and retained benchmark data

| Purpose | Path | Approximate size |
| --- | --- | ---: |
| Official-weight DeepSeek-V4 tiny slice | `/home/ssm-user/ds-v4-tiny-real` | 7.3 GiB |
| Qwen comparison checkpoint | `/home/ssm-user/Qwen3-8B` | 16 GiB |

Do not rebuild or redownload these for routine testing. Verify their
`config.json` and model files before assuming a path copied to another host is
equivalent.

Important retained results:

```text
/tmp/dsv4-full-q512-tp2-lnc2-merged-20260826/
/tmp/qwen3-8b-full-q512-tp2-lnc2-20260826/
```

DeepSeek files:

```text
cold.log
cold.time
cache/
```

Qwen files:

```text
cold.log
cold.time
cold-result.json
warm.log
warm.time
warm-result.json
cache/
```

`/tmp` is not durable across instance replacement or reboot policies. Copy
anything required for a long-lived bug report before changing hosts.

## Environment variables used by the full benchmark

The known TP2/EP2 invocation used:

```bash
export VLLM_NEURON_ENABLE_DEEPSEEK_V4=1
export VLLM_NEURON_VALIDATE_CACHE_METADATA=1
export NEURON_SKIP_EFA_AFFINITY=1
export NEURON_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/ssm-user/vllm-neuron
export VLLM_CACHE_ROOT=/tmp/<one-isolated-run>/cache
```

Notes:

- `VLLM_NEURON_ENABLE_DEEPSEEK_V4=1` enables model registration guarded as
  experimental.
- `VLLM_NEURON_VALIDATE_CACHE_METADATA=1` keeps cache metadata checks enabled.
- `NEURON_SKIP_EFA_AFFINITY=1` skips optional EFA CPU-affinity handling. It
  does not disable Neuron compilation or collectives.
- `NEURON_VISIBLE_DEVICES=0,1` is the tested TP2/LNC2 setting on this host.
- `VLLM_CACHE_ROOT` controls the vLLM/Neuron model cache used by these runs.
- `NKI_ENABLE_TRACE_CACHE=1` enables persistent NKI trace caching and can hide
  NKI cold-compilation work. Use it only when that is the intended measurement.
  Disable it, or use an isolated NKI cache as supported by the installed SDK,
  for a strict cold NKI measurement.
- Do not set `VLLM_NEURON_CPU_MODE` or `VLLM_NEURON_CPU_COMPILE` for device
  compilation.
- Keep compiler optimization at the repository's Neuron default `-O1` for
  comparable results.

Avoid copying the interactive shell's entire environment into a benchmark.
Record only deliberate variables; stale cache/compiler/debug variables are a
common source of irreproducible results.

## Cache discipline

### Cold run

Use a new directory instead of deleting or reusing a possibly incomplete one:

```bash
export DSV4_RUN_ROOT="$(mktemp -d /tmp/dsv4-cold-XXXXXXXX)"
export VLLM_CACHE_ROOT="$DSV4_RUN_ROOT/cache"
mkdir -p "$VLLM_CACHE_ROOT"
```

Record the run directory in the log and result JSON. The local NVMe filesystem
is important: do not place cold compiler caches on network storage.

### Warm run

Relaunch the identical command with the identical `VLLM_CACHE_ROOT`. Do not
change the checkpoint path, source graph, TP/EP layout, bucket list, maximum
length, cache layout, compiler version, or flags. Any such change may produce a
legitimate new cache key.

Monitor for accidental compilation:

```bash
pgrep -af 'neuronx-cc|neuron_parallel_compile'
```

A valid warm-cache result launches no `neuronx-cc` process and submits no model
or NKI specialization for compilation.

### Incomplete runs

Keep failed cache directories until their logs, cache keys, and partial
artifacts have been recorded. Do not manually mark incomplete entries as
complete or copy a NEFF under a different key. Use a new isolated directory for
the next cold experiment.

## Safe compile monitoring

Before a long run:

```bash
neuron-ls
free -h
df -h /tmp /home/ssm-user
pgrep -af 'python.*generate_tiny|neuronx-cc|neuron_parallel_compile' || true
```

During compilation, monitor host memory and compiler processes separately from
device memory:

```bash
watch -n 2 'free -h; ps -eo pid,ppid,rss,%cpu,etime,cmd --sort=-rss | head -n 20'
```

Use `/usr/bin/time -v` around the outer process to capture total wall time and
peak RSS. Log stdout and stderr together so asynchronous Neuron errors and
compiler cache keys are retained.

The known broken full graph ran two TP-rank compilers concurrently and reached
119.85 GiB RSS. Do not repeat it unchanged on this host. Begin with standalone
components or a one-rank diagnostic identity bisection.

The large `sg06`/`sg04` dependency explosion was subsequently traced to
`DeepseekV4Compressor.forward_packed` gathering a raw compression window for
every query, not to the HCA MLA compressed-history span. The HCA capacity and
uniform-span changes remain separate secondary optimizations. A boundary-only
NKI compressor removes the old fan-out, but its 2026-08-28 TP1 diagnostic still
exceeded a 20-minute bound in the remaining outer graph. Do not advance to
TP2/EP2 until the one-rank compile satisfies the acceptance ladder.

## Development test ladder

Run the cheapest relevant layer first. A successful CPU test does not prove NKI
compilation; a successful standalone NKI compile does not prove that the full
outer graph is small.

### 1. Static and CPU tests

```bash
python -m pytest -q \
  test/unit/model/deepseek_v4 \
  test/unit/vllm/test_deepseek_v4_admission.py \
  test/unit/vllm/test_deepseek_v4_feature_guards.py \
  test/unit/vllm/test_deepseek_v4_registry_gate.py
```

For focused changes, prefer the smallest applicable files, especially:

```text
test/unit/model/deepseek_v4/test_indexer.py
test/unit/model/deepseek_v4/test_components.py
test/unit/model/deepseek_v4/test_parallelism.py
test/unit/model/deepseek_v4/test_tiny_model.py
test/vllm_neuron/test_deepseek_v4_nki_simulator.py
```

### 2. Portable correctness oracles

Keep the CPU portable implementations working. They are the reference for
streaming top-k, shared-latent attention, compression boundaries, MoE duplicate
route aggregation, clamps, and shared-expert addition.

Useful tools include:

```text
tools/deepseek_v4/compare_against_reference.py
tools/deepseek_v4/compare_tiny_logits.py
tools/deepseek_v4/check_indexer_device.py
tools/deepseek_v4/check_mhc_portable.py
tools/deepseek_v4/verify_tiny_from_official.py
```

Read each tool's `--help` and source before using it against the real checkpoint;
some defaults target the synthetic tiny checkpoint.

### 3. NKI simulator

Run the simulator tests after modifying NKI code:

```bash
python -m pytest -q test/vllm_neuron/test_deepseek_v4_nki_simulator.py
```

Cover Q1 and fixed prefill query tiles, sink-aware softmax, padding, null
blocks, tied scores, and partially visible final pages.

### 4. Standalone device compile

Use an isolated cache and compile one component at a time:

```bash
python tools/deepseek_v4/compile_runtime_mla.py \
  --query 512 --compressed 0 --output "$DSV4_RUN_ROOT/mla-sliding.json"

python tools/deepseek_v4/compile_runtime_mla.py \
  --query 512 --compressed 512 --output "$DSV4_RUN_ROOT/mla-csa.json"

python tools/deepseek_v4/compile_runtime_mla.py \
  --query 512 --compressed 1024 --output "$DSV4_RUN_ROOT/mla-hca.json"

python tools/deepseek_v4/compile_runtime_indexer.py \
  --query 512 --block-columns 1024 --logical-slots 128 \
  --visible 131072 --output "$DSV4_RUN_ROOT/indexer.json"

python tools/deepseek_v4/compile_runtime_moe.py \
  --query 512 --output "$DSV4_RUN_ROOT/moe.json"

python tools/deepseek_v4/compile_runtime_compressor.py \
  --ratio 128 --head-dim 512 --query 512 \
  --output "$DSV4_RUN_ROOT/compressor-hca.json"

python tools/deepseek_v4/compile_runtime_compressor.py \
  --ratio 4 --head-dim 512 --query 512 \
  --output "$DSV4_RUN_ROOT/compressor-csa.json"

python tools/deepseek_v4/compile_runtime_compressor.py \
  --ratio 4 --head-dim 128 --query 512 \
  --output "$DSV4_RUN_ROOT/compressor-indexer.json"
```

These scripts report their Python process RSS. Also wrap them in
`/usr/bin/time -v` when comparing compiler peak memory across revisions.

### 5. Full-graph bisection

The current `tools/deepseek_v4/benchmark_prefill_components.py` sets
`VLLM_NEURON_DEEPSEEK_V4_DIAGNOSTIC_IDENTITY`, but the model does not read that
variable. Its advertised identity replacement is therefore not functional yet.
Implement and test graph-stable identity boundaries before trusting a
full-model component-bisection result.

Start full-graph diagnosis with one rank to limit simultaneous compiler memory.
TP1 is a diagnostic only; rerun TP2/EP2 before accepting the production target.

### 6. Full acceptance run

Use the exact full command recorded in the Q512 MLA defect handoff. Capture:

- graph extraction time;
- compiler wall time and peak RSS;
- NEFF count and total/largest sizes;
- initialization time;
- time to first generated token;
- warm relaunch with zero compiler processes;
- official token/logit equivalence.

## Reproducibility snapshot commands

Run these before and after changing the SDK or moving hosts:

```bash
python tools/deepseek_v4/write_device_preflight.py \
  /tmp/deepseek-v4-preflight.json \
  --cache-root "$VLLM_CACHE_ROOT"

neuron-ls > /tmp/neuron-ls.txt
uname -a > /tmp/uname.txt
dpkg-query -W > /tmp/dpkg-packages.txt
python -m pip freeze > /tmp/dev-venv-pip-freeze.txt
neuronx-cc --version > /tmp/neuronx-cc-version.txt 2>&1
git status --short --branch > /tmp/git-status.txt
git rev-parse HEAD > /tmp/git-head.txt
```

Store these beside benchmark logs. The Git SHA alone is insufficient because
the worktree, Python stack, system runtime, compiler, and cache can all change a
result.

## Common failure interpretations

- A traceback at `.to(device)` may be an asynchronous error from an earlier
  dispatched graph. Use timestamps, cache keys, and emitted NKI artifacts to
  identify the real target.
- `No available shared memory broadcast block` once per minute generally means
  workers are busy compiling; it is not proof of a deadlock.
- `neuronx-cc` exit 70 / `F137` means the compiler was forcibly killed, usually
  for host-memory exhaustion. Check `/usr/bin/time -v`, `free`, and kernel logs.
- A standalone kernel compiling successfully proves only that specialization.
  It does not prove the full model graph compiles.
- A warm run is not valid if any graph key, compiler flag, source path affecting
  the graph, or cache directory changed.
- Weight size is not a proxy for compiler graph size. The Qwen3-8B comparison
  succeeded while the three-layer DeepSeek tiny graph exhausted host RAM.

## Before handing off a change

Record all of the following:

1. Git branch, SHA, and exact changed paths.
2. Development or packaged stack, including the Python executable path.
3. Compiler, NKI, torch, torch-neuronx, vLLM, and transformers versions.
4. `neuron-ls`, visible devices, TP/EP/LNC layout, and host free memory.
5. Checkpoint path and whether weights are real, dummy, or quantized.
6. Prefill and decode buckets, maximum model length, cache block override, BF16
   and `-O1` settings.
7. Cold cache path and whether NKI trace caching was enabled.
8. Wall time, peak RSS, NEFF sizes, runtime latency, and correctness result.
9. Warm-cache relaunch evidence showing no compilation.
10. Any asynchronous warning or error, including the earlier operation that may
    actually have caused it.
