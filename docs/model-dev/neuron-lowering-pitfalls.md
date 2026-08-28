# Neuron lowering pitfalls

Torch operations that are correct on CPU and **wrong, or unbuildable, on
Neuron**. Each entry here cost real debugging time; none of them is visible from
reading model code, and several fail silently.

Read this before writing a new model, and before spending a day on a device-only
numerical difference.

## Silently wrong results

### `Tensor.split` with a *list* of sizes, on any dim but 0

**The worst one on this list, because nothing fails.** It returns the wrong
data, for every chunk, with no error and no warning.

```python
# WRONG on Neuron
pre, post, comb = projected.split([4, 4, 16], dim=-1)

# correct
pre  = projected[..., :4]
post = projected[..., 4:8]
comb = projected[..., 8:]
```

Measured on trn2 (neuronx-cc 2.27, torch/torch-xla 2.11) with
`tools/deepseek_v4/check_mhc_device.py --check-split`:

| Form | Result |
| --- | --- |
| `split([4,4,16], dim=-1)` | **all chunks wrong** |
| `split([8,8,8], dim=-1)` -- uniform sizes, but a list | **all chunks wrong** |
| `split([4,20], dim=1)` | **both chunks wrong** |
| `split(8, dim=-1)` -- int size | correct |
| `chunk(3, dim=-1)` | correct |
| `split([2,6], dim=0)` | correct |

So it is specifically the *list-of-sizes* form on a non-zero dim. An int size,
`chunk`, `narrow`, `index_select` and plain slicing are all correct.

The FX graph is **correct** (`split` -> `getitem`), so the defect lies below FX.
It can be localized exactly, because the compile cache keeps every stage next to
the NEFF: `fxgraph.txt`, `hlo_passes/step1_torch_xla_trace.hlo`, the later pass
outputs, `graph.hlo`, `command.txt` (the `neuronx-cc` invocation) and the
`.neff`.

Decoding the *first* HLO -- the one torch-xla's `PyLoweringContext` emits, before
any lite pass -- for `arange(8*24).reshape(8, 24).split([8,8,8], dim=-1)`:

```
parameter   8x24
reshape     192                                   <- flatten to 1-D
slice        64   [start=0  limit=64  stride=1]   <- 64 *contiguous* elements
reshape      8x8
reshape     192
slice        64   [start=8  limit=72  stride=1]
reshape      8x8
reshape     192
slice        64   [start=16 limit=80 stride=1]
reshape      8x8
```

The column offsets 0/8/16 are used as **flat** offsets, and each chunk takes one
contiguous run of 64 rather than 8 elements from each of 8 rows at stride 24. So
the stride-ignoring read model below is not a hypothesis fitted to the outputs;
it is what the IR literally says.

**This exonerates `neuronx-cc`**, which faithfully compiles the HLO it is given
-- and the same compiler version is used by the stack that gets this right. The
defect is in torch-xla's lowering of `split`. Decode any cached stage with
`libtorch_neuronx_lite.pyhlo.hlo_pb2.HloModuleProto`.

How it presented in practice (DeepSeek-V4, `hyperconnection_reference`): the
un-split `F.linear` output was correct to `2.4e-4`, while a chunk of that same
correct tensor, in the same graph, was off by **717**. Downstream, the model
generated entirely different text on device than on CPU, with step-0 logits off
by 20.69. Full account:
[`deepseek-v4-real-weight-validation.md`](deepseek-v4-real-weight-validation.md).

**Mechanism.** The lowering computes the right *starting offset* for each chunk
and then reads a **contiguous run** of `rows * width` elements from there, as if
the source were flat 1-D -- it ignores the row stride. So row 0 is always
correct, and every later row is filled with whatever sits next in memory. On
`arange(48).reshape(2, 24).split([4, 4, 16], dim=-1)`:

| Chunk | Row | Expected | Got |
| --- | --- | --- | --- |
| 0 | 0 | `0 1 2 3` | `0 1 2 3` |
| 0 | 1 | `24 25 26 27` | **`4 5 6 7`** |
| 1 | 1 | `28 29 30 31` | **`8 9 10 11`** |
| 2 | 1 | `32 … 47` | **`24 … 39`** |

Predicting the output from that stride-ignoring model reproduces the device
result exactly, so the diagnosis is verified rather than inferred. It also
explains why `dim=0` survives: rows *are* contiguous there, so a flat read
happens to be right.

Note what the bad data is: **real, valid, adjacent values**, not garbage or
uninitialised memory. That is why nothing downstream ever complains, and why a
model carrying this bug looks statistically healthy while being wrong.

Still unfixed upstream. `tools/repro_neuron_split_lowering.py` is a
self-contained reproducer -- torch plus a `vllm_neuron` import for the backend,
nothing else -- suitable for filing with AWS. It exits non-zero while the defect
is present, so it starts passing if the lowering is fixed. Run it with
`VLLM_NEURON_CPU_MODE=1` for the passing control.

**What decides this is whether the backend decomposes `split`, not the torch
version.** Measured on all three stacks available here:

| stack | `split([4,4,16], dim=-1)` |
| --- | --- |
| torch 2.11 / torch-xla 2.11 / lite 2.11 | **wrong chunks: [0, 1, 2]** |
| torch 2.12 / torch-xla 2.12 / lite 2.12 | **wrong chunks: [0, 1, 2]** |
| torch 2.12 / from-source torch-neuronx 2.12.3 | correct (eager and compiled) |

Both claims you will hear are therefore true, and they are about different
stacks: "torch 2.12 on Neuron lowers `split` correctly" (the third row) and
"upgrading vllm-neuron to torch 2.12 does not fix it" (the second). On lite,
`split` survives Dynamo as one node and is mis-lowered below FX; on the
from-source backend it is decomposed to three `aten.slice.Tensor` before the
backend ever sees it -- the same explicit-slice form as the manual workaround --
so the buggy path is never taken. Same `neuronx-cc` (2.27.5334.0) in all three;
only the IR differs.

vLLM runs only on the first two stacks, so the sliced workaround stays. Full
procedure, and the evidence to demand of any such claim, in
[neuron-lowering-stacks.md](neuron-lowering-stacks.md).

### `torch.cat` of a small rotary suffix onto a rank-4 tensor

Neuron lowered the small rotated suffix as a dead/zero concat operand, zeroing
the two rotary channels -- while the otherwise identical rank-3 KV path was
correct. Use a functional overwrite instead:

```python
return torch.index_copy(x, -1, rotary_indices, rotated)
```

`vllm_neuron/model/deepseek_v4/attention.py:36-44`, commit `15e548c`.

### Wide gathers above the DVE free-dimension limit

The multi-group form corrupts results on hardware (NKILIB-1592); wide gathers
must be tiled to `max_free_dim` (`2**14`).
`vllm_neuron/functional/vendored_kernels/rotational_topk/rotational_topk_utils.py:56-67`.

### Dtypes that do not survive XLA lowering

Compute in `int32` and cast to `uint32` as the *final* op, or the dtype is lost:
`vllm_neuron/functional/moe/build_all_gatherv_metadata.py:95`. Scatter needs its
mask width padded to at least K: `build_all2all_dispatch_metadata.py:190`.

## Fails loudly at graph capture or on the device

These at least tell you something is wrong.

### Data-dependent branching

Reading a tensor's *values* to decide control flow (`if t.min() < 0:`,
`if t.any():`) makes Dynamo raise `Unsupported: Data-dependent branching` -- it
does not graph-break, it fails. Guard validation with
`torch.compiler.is_compiling()` so eager keeps the check:

```python
if not torch.compiler.is_compiling():
    if input_ids.min() < 0 or input_ids.max() >= table.shape[0]:
        raise ValueError("token id is outside the table")
```

Examples: `vllm_neuron/model/deepseek_v4/moe.py`,
`vllm_neuron/model/deepseek_v4/mhc.py::sinkhorn_positive`. Shape-based checks
are safe and should stay outside the guard.

### Non-contiguous tensors into `copy_` / `index_put_`

Neuron rejects a non-contiguous source. Worse, at *load* time it can also refuse
to run the `.contiguous()` that would fix it -- so build on CPU and let `copy_`
cross the device boundary:

```python
identity = torch.eye(head_dim, dtype=param.dtype)          # CPU
param.copy_(identity.unsqueeze(0).expand(heads, -1, -1).contiguous())
```

`vllm_neuron/model/deepseek_v4/model.py`, and
`vllm_neuron/functional/attention/attention_decode.py:610-615` for the
`index_put_` case. Also note a stride-2 partial store does not lower at all
(`attention_decode.py:562-576`); write whole pairs instead.

### `.to(device=..., dtype=...)` in one step

Neuron rejects a copy that changes device *and* dtype together ("Expected
self.dtype() == dst.dtype()"); CPU performs it silently. Cast on the host first,
leaving a pure transfer for the device. This bites weight loading whenever
parameters and checkpoint differ in dtype, and stock
`transformers.modeling_rope_utils` hits it via
`torch.arange(...).to(device=device, dtype=torch.float)` -- so pass a CPU device
to RoPE init and copy the result across.

### `torch.topk` returns **unsigned** indices

On Neuron `torch.topk(...)[1]` comes back as `uint32`; CPU returns `int64`. The
values are correct, so anything that only *gathers* with them is fine. What
breaks is the near-universal idiom of marking "no selection" with `-1`:

```python
chosen = torch.topk(scores, k, dim=-1)[1]          # uint32 on device
chosen = torch.where(invalid, torch.full_like(chosen, -1), chosen)
valid  = chosen >= 0                               # ALWAYS TRUE on uint32
```

`full_like(chosen, -1)` wraps to 4294967295 and `>= 0` is vacuously true on an
unsigned type, so the sentinel is never recognised and that value is handed to
the next `scatter`/`gather` as a real index. Measured on trn2 with a 33-wide
scatter buffer: CPU produced `[-1, -1]`, the device produced
`[4294967295, 4294967295]`.

The device does detect it, but reports it against the whole NEFF with no
instruction and no op name:

```
scatter/gather (indirect memory copy via vector DGE) out-of-bound access.
model name = .../graph_<hash>.neff, neff instruction index = unknown
[ND 0][NC 4] Out of bounds access on model .../graph_<hash>.neff
RuntimeError: nrta status=1006
```

The traceback surfaces wherever the deferred error is next observed -- for us,
inside tensor capture, several frames from the cause.

Cast immediately, and treat out-of-range as "no selection" rather than clamping
it into range (clamping turns a meaningless pick into a confident selection of
the last entry):

```python
chosen = torch.topk(scores, k, dim=-1)[1].long()
invalid = (chosen >= threshold) | (chosen < 0) | (chosen >= entries)
```

**This is stack-specific, and the fix stays regardless.** Measured with
`tools/deepseek_v4/check_indexer_device.py`, which now prints the raw top-k
index dtype and takes `--stack` to pick the backend:

| stack | `topk(...)[1]` dtype | pre-fix `select` at an all-`-inf` row |
| --- | --- | --- |
| torch 2.11 / torch-xla 2.11 | **uint32** | returns **4294967295** |
| torch 2.12 / torch-xla 2.12 | **uint32** | returns **4294967295** |
| from-source torch-neuronx 2.12.3 / torch 2.12.1 | int64 | returns `-1`, as CPU does |

Same split as `Tensor.split` above, and the same lesson: the torch version is
not what changes the answer, the lowering stack is. The `.long()` therefore
stays -- both torch-xla stacks need it, and it is free on the third. All three
pass with the cast in place, at the tiny shapes and at the real Flash config
(`entries=1024, topk=512, heads=64, head_dim=128`).

Two things made this expensive to find, both worth copying:

* A device probe of the same functions **passed**, because it happened to test
  only rows with real values to rank. The bug needs a row that is entirely
  `-inf` -- a query with nothing yet eligible -- which is exactly the
  early-sequence case a synthetic probe skips. Probe the degenerate regime, not
  just the representative one.
* The FX-graph op histogram of the failing capture against a passing control
  named the culprit in one step: the only new indirect-memory op in the whole
  graph was `scatter_`. `tools/deepseek_v4/check_indexer_device.py` is the probe;
  the histogram is just `call_\w+\[target=...\]` counted per graph.

`vllm_neuron/model/deepseek_v4/indexer.py`.

## How to find one of these

The full model is the wrong place to look -- 7-13 minutes per compile at a tiny
slice's shape. Compile the **suspect module alone**, following
`examples/vllm_neuron/basics/helloworld.py`:

```python
import vllm_neuron                          # registers the dynamo backends
from vllm_neuron.envs import get_compile_backend_name

module, x = module.to("neuron:0"), x.to("neuron:0")
compiled = torch.compile(module, backend=get_compile_backend_name())
torch.testing.assert_close(expected_cpu, compiled(x).to("cpu"))
```

That compiles in about **1.5 seconds**, and because you own the harness the
module can simply *return* its intermediates -- no `tensor_capture`, which keeps
only element `[0]` of a tuple return. `tools/deepseek_v4/check_mhc_device.py` is
a worked example.

Set `NEURON_RT_VISIBLE_CORES` alongside `NEURON_VISIBLE_DEVICES`; the runtime
rejects one without the other, and a standalone harness has no vLLM worker to
set it for you.

Then bisect by **rewriting the expression into an equivalent op sequence and
diffing the device output byte-for-byte**. This is the discriminator that works:
`cat` -> `index_copy` changed the result and was a real bug; `expand` ->
`repeat` was byte-identical and was correctly ruled out.

Two habits worth keeping:

- **Measure intermediates; do not reason about which op looks riskiest.** The
  `split` bug was chased for a while as a suspected reduction defect, on the
  strength of the reduction width being exactly `2**14`. One standalone compile
  returning the intermediate showed the reduction accurate to `9.3e-10`.
- **Compare per position, not just in aggregate.** A defect that is exact at
  most positions and wrong at a few reads as floating-point noise in any
  summary statistic.

## When you find one

Land the workaround in model code and comment it with the observed symptom, the
rank/shape it occurred at, and the CPU-correct control -- otherwise it gets
"simplified" back into the broken form. Then add it here.
