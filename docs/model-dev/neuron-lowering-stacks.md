# Which Neuron stack are you talking about?

> Migration note (2026-08-24): this plugin now runs stack 3, TorchNeuron
> Native, using the external `torch-neuronx` package and `backend="neuron"`.
> Statements below that stack 3 cannot run the plugin describe the historical
> lite-based release and are retained as provenance, not current behavior. See
> [TorchNeuron Native backend](../design/compilation/torch-neuronx-native.md).

Two different things get called "torch 2.12", and they give **opposite answers**
about the `Tensor.split` defect. Claims about it are irreproducible unless they
name the stack, so this is the procedure for reproducing any of them, and the
evidence to demand before believing a result — including one of mine.

The short version: **"torch 2.12" is not the discriminator. Whether the backend
decomposes `split` before lowering it is.**

## The stacks on this machine

| # | stack | torch | lowering path | `split([4,4,16], dim=-1)` |
|---|---|---|---|---|
| 1 | `~/.venv-vllm-neuron`, `vllm-neuron/.venv-neuron`, `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0` | 2.11.0 | libtorch-neuronx-lite 2.11 + torch-xla 2.11 | **wrong** |
| 2 | `~/.venv-vllm-neuron-212` | 2.12.1 | libtorch-neuronx-lite 2.12 + torch-xla 2.12 | **wrong** |
| 3 | `~/.venv-torch-neuronx-dev` | 2.12.1 | from-source torch-neuronx 2.12.3, torch-mlir/StableHLO | correct |

Stacks 1 and 2 are what this plugin ships on; vLLM only ever runs there. Stack 3
cannot run this plugin at all (see [Why stack 3 cannot run
vLLM](#why-stack-3-cannot-run-vllm)).

### Provenance: lite is not a third-party stack

Worth knowing before treating "get off lite" as an option. vLLM does not require
it — zero files in the `vllm` package reference it — and it is not an artifact
of this project's work either. Upstream's `release-0.24.0.1.1.0` (`ed3580d`)
**ships with it**: that commit is a *root* commit, a vendor drop with no parent
(this repo has two unrelated roots, `ae6c10e` for 0.21 and `ed3580d` for 0.24),
so there is no transition to inspect as a diff — 0.24 simply arrives this way.
In that pristine tree:

* `requirements/core.txt` carries `libtorch-neuronx-lite` under upstream's own
  comment, "Left unpinned to resolve based on vLLM's torch requirement";
* `vllm_neuron/compile/` **does not exist** — upstream removed the in-repo
  compile stack for 0.24;
* `get_compile_backend_name()` returns `neuron_libtorch`, with the docstring
  reserving the name `"neuron"` for torch_neuronx, verbatim.

This branch's only diffs against that tree in the relevant files are a
`transformers` ceiling, two DeepSeek-V4 validation env vars, and a patch-registry
refactor. Nothing here touches the backend choice or the dependency.

More to the point, **it is the same code the plugin used to carry itself**. On
`release-0.21.0.1.0.0` the compile stack lived in-repo at `vllm_neuron/compile/`
— `backend.py`, `hlo.py`, `capture_backend.py`, `cache.py`, `parallel_compile.py`,
`parallel_trace.py`, `platform.py`, `schema.py`, `artifacts.py` — which is the
file list now inside `libtorch_neuronx_lite/compile/`. Diffing 0.21's `hlo.py`
against lite's gives **23 lines, every one an import rename**:

```
< from torch_neuronx.pyhlo import hlo_pb2, xla_data_pb2
> from libtorch_neuronx_lite.pyhlo import hlo_pb2, xla_data_pb2
```

Note what 0.21 imported it *from*: `torch_neuronx.pyhlo` / `torch_neuronx.xla_impl`,
because the torch-neuronx of that era **was** the XLA-based one. So "torch-xla vs
torch-neuronx" was never the real axis — old torch_neuronx and lite are one
lineage, and only the *from-source* torch-neuronx (torch-mlir/StableHLO) is
architecturally different. That is why stack 3 alone behaves differently.

The consequence: there is no pre-lite lowering path to retreat to. The same
FX-to-HLO conversion has been in place since 0.21, under the plugin's own
`vllm_neuron` backend name; the `split` defect predates the extraction.

### Two true statements that sound contradictory

So both of these are true, and they are not in conflict:

* "torch 2.12 on Neuron lowers `split` correctly" — **stack 3**.
* "upgrading vllm-neuron to torch 2.12 does not fix `split`" — **stack 2**.

If someone reports one without naming the stack, that is the first thing to
establish. It is the whole disagreement.

## The mechanism, and why the torch version is a red herring

Dump what actually reaches the backend. On stack 2, `split` survives Dynamo as a
single call, and the backend mis-lowers it below FX (`fxgraph.txt`, written into
the compile cache next to every NEFF):

```
%split : [num_users=3] = call_method[target=split](args = (%l_x_, [8, 8, 8]), kwargs = {dim: -1})
%getitem   : call_function[target=operator.getitem](args = (%split, 0))
%getitem_1 : call_function[target=operator.getitem](args = (%split, 1))
%getitem_2 : call_function[target=operator.getitem](args = (%split, 2))
```

On stack 3 the same source never produces a `split` node at all — it is
decomposed to `aten.slice.Tensor`, which is the *same explicit-slice form* used
as the manual workaround in `mhc.py`:

```
slice_1: "f32[2, 8]neuron:0" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 8)
slice_2: "f32[2, 8]neuron:0" = torch.ops.aten.slice.Tensor(arg0_1, 1, 8, 16)
slice_3: "f32[2, 8]neuron:0" = torch.ops.aten.slice.Tensor(arg0_1, 1, 16, 24)
```

Stack 3 is therefore not evidence that the bug was found and fixed — it never
executes the buggy path. `neuronx-cc` is the *same version* (2.27.5334.0) in all
three stacks; only the IR reaching it differs. That is why bumping torch under
lite changes nothing: the decomposition table belongs to the backend, not to
torch.

## Reproducing

### The split defect (~15 s per stack)

`tools/repro_neuron_split_lowering.py` is self-contained and exits non-zero
while the defect is present. On stack 1 or 2:

```bash
VENV=~/.venv-vllm-neuron-212          # or ~/.venv-vllm-neuron for stack 1
PATH="$VENV/bin:$PATH" \
PYTHONPATH=$PWD \
NEURON_VISIBLE_DEVICES=0 NEURON_RT_VISIBLE_CORES=0 \
VLLM_CACHE_ROOT=$(mktemp -d) \
  $VENV/bin/python tools/repro_neuron_split_lowering.py
```

On stack 3, which has no vLLM and no torch-xla, use the port at
`~/repro_split_torch_neuronx_dev.py`:

```bash
PATH=~/.venv-torch-neuronx-dev/bin:$PATH \
NEURON_VISIBLE_DEVICES=0 NEURON_RT_VISIBLE_CORES=0 \
  ~/.venv-torch-neuronx-dev/bin/python ~/repro_split_torch_neuronx_dev.py
```

To see the decomposition itself, set `TORCH_LOGS=aot_graphs` and grep for
`aten.slice` / `aten.split`.

### The indexer's ops (~1 min per stack)

```bash
tools/deepseek_v4/check_indexer_device.py --stack xla       # stacks 1, 2
tools/deepseek_v4/check_indexer_device.py --stack neuronx   # stack 3
```

Point `--indexer-path` at a copy of `indexer.py` with the `.long()` cast removed
to test the *backend* rather than our fix — with the cast in place every stack
passes and tells you nothing.

### The model, on real weights (~15 min, mostly compile)

```bash
VLLM_NEURON_ENABLE_DEEPSEEK_V4=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_NEURON_TINY_VALIDATION_DIR=<capture> \
NEURON_VISIBLE_DEVICES=0 NEURON_SKIP_EFA_AFFINITY=1 \
VLLM_CACHE_ROOT=<private-cache> PATH="$VENV/bin:$PATH" \
  $VENV/bin/python tools/deepseek_v4/generate_tiny_tp1.py ~/ds-v4-tiny-real \
  --output <json> --load-format auto \
  --prompt 671,6102,294,8760,344,1024,2048,4096 --max-model-len 16
```

Drop `--enforce-eager` for device, add `VLLM_NEURON_CPU_MODE=1` and
`--enforce-eager` for the CPU oracle.

## Evidence to demand (of anyone, including me)

Four ways one of these runs lies to you. All four bit me here.

**A shared compile cache.** Every stack defaults to
`~/.cache/neuron_libtorch/neuron/compile_cache`, so a "2.12 run" can replay a
NEFF built by 2.11. Always set a private empty `VLLM_CACHE_ROOT`, and check for
`Local cache miss for key:` followed by `Compiling...` — a run that only prints
`Local cache hit` measured nothing. (The key does include a version string, so
the two do not normally collide; do not rely on that.)

**A silent CPU fallback.** A stack that quietly ran on CPU *passes*, which looks
like a fix. Confirm the tensors are on `neuron:0` and that NEFFs were compiled.
`repro_split_torch_neuronx_dev.py` carries an `x+1000` negative control for
exactly this.

**Testing the fix instead of the backend.** `indexer.py` casts top-k indices to
int64, so it passes everywhere. Any claim about a *backend* needs the pre-fix
form.

**One regime only.** The uint32 top-k defect appears only in rows that are
entirely `-inf`. A probe using representative inputs reports MATCH while the
model faults. See
[deepseek-v4-lightning-indexer.md](deepseek-v4-lightning-indexer.md).

## Why stack 3 cannot run vLLM

Not a packaging problem:

* `vllm_neuron.envs.get_compile_backend_name()` returns only `neuron_libtorch`
  or libtorch-neuronx-lite's native backend, and its docstring reserves the name
  `"neuron"` for torch_neuronx as a *separate* install.
* 49 files under `vllm_neuron/` import lite APIs.

vLLM itself is not the obstacle: `vllm==0.24.0` pins `torch==2.11.0`, but that
is a metadata pin, not an ABI wall — its extensions are `abi3`, and vLLM plus
`vllm_neuron` import and generate correctly on torch 2.12.1. Stack 2 was built
by cloning stack 1 rather than upgrading in place:

```bash
cp -a ~/.venv-vllm-neuron ~/.venv-vllm-neuron-212
~/.venv-vllm-neuron-212/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 torchvision
~/.venv-vllm-neuron-212/bin/python -m pip install --no-deps \
  --extra-index-url https://pip.repos.neuron.amazonaws.com \
  torch-xla==2.12.0 libtorch-neuronx-lite==2.12.0.1.0.1284+f49d8626
~/.venv-vllm-neuron-212/bin/python -m pip uninstall -y torchaudio
```

`torchaudio` has no torch-2.12 build and must be removed, not upgraded;
transformers guards that import with `is_torchaudio_available()`, a presence
check.

Note the lite version suffix tracks the torch it was built against, not the
source: 2.11.0.1.0.1284 and 2.12.0.1.0.1284 share the build hash `f49d8626` and
ship byte-identical Python. The compiled `libtorchneuron.so` does differ, so the
2.12 wheel is a real rebuild — it simply does not change this behaviour.

## What stack 2 measured

Real 3-layer slice, 8-token prompt, `--max-model-len 16`, private caches on both
sides:

| comparison | worst `max\|diff\|` | argmax |
|---|---|---|
| device stack 1 vs device stack 2 | **0** — bit-identical | identical |
| device vs CPU, stack 1 | 0.25 | identical |
| device vs CPU, stack 2 | 0.25 | identical |
| CPU stack 1 vs CPU stack 2 | 0.125 — one bf16 ULP here | identical |

Tokens `85, 85276, 50955, 125488` in all four runs. The FX graphs handed to the
backend are **byte-identical** between stacks 1 and 2 (`diff` of the cached
`fxgraph.txt` is empty), which is why the device outputs match bit for bit —
and is the cleanest single statement of why a torch bump changes nothing here.
