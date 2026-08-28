# Tiny DeepSeek-V4 TP1 correctness deployment

Build the deterministic three-layer, 64-token checkpoint and tokenizer:

```bash
.venv/bin/python tools/deepseek_v4/build_tiny_checkpoint.py /tmp/deepseek-v4-tiny
```

Both the command above and the launcher below assume the virtual environment is
`.venv` at the repository root. If yours is named differently, do not edit the
launcher -- point it at your interpreter with the two override variables it
already honors:

```bash
export VLLM_NEURON_PYTHON=$PWD/.venv-neuron/bin/python
export VLLM_NEURON_VLLM=$PWD/.venv-neuron/bin/vllm
```

Without these the launcher fails immediately on a missing `.venv/bin/python`.

Start the OpenAI-compatible server with synchronous CPU sampling and an
isolated compile cache. Optional final arguments select the port and logical
core; otherwise the launcher finds the first free core with `neuron-ls -j`:

```bash
tools/deepseek_v4/run_tiny_tp1.sh /tmp/deepseek-v4-tiny /tmp/deepseek-v4-cache 8001
```

The diagnostic metadata validator is enabled by the launcher. It validates
the actual heterogeneous cache metadata immediately before graph capture and
is intentionally unsuitable for performance measurements because it performs
device-to-host checks. The launcher also writes `device-preflight.json` into
the isolated cache directory with Python, compiler, runtime, driver, instance,
Git revision, and initial cache inventory information.

The launcher prepends the virtual environment to `PATH` so Lite invokes its
matching `neuronx-cc`, sets `VLLM_CACHE_ROOT` (artifacts live below
`neuron/compile_cache`), disables CPU emulation, and skips optional EFA affinity
when EFA is unavailable. `PJRT_DEVICE=CPU` is expected on this stack: Neuron
compilation uses `neuron_libtorch` capture and `neuronx-cc`, not
`torch_xla.device()`.

The run writes NKI kernel caches into the working directory as 17-hex-char
directories of `.colz` files. They are regenerable and `.gitignore`d, but note
they accumulate across runs.

A cold accepted run has exactly three HLO/NEFF entries: prefill 8, prefill 64,
and decode `(1,64)`. A restart must report three cache hits and zero submitted
HLOs. Reject PJRT faults, DGE/scatter/gather out-of-bounds messages, allocator
assertions, tracebacks, and worker crashes. This gate does not qualify the full
checkpoint, TP>1, longer contexts, concurrency, or performance.

A cold run of this gate costs about 15 minutes, roughly 80 percent of it a
single graph (prefill 64). For iteration, not for the gate itself, the launcher
accepts `VLLM_NEURON_TINY_FAST=1`, which serves at length 16 with a smaller KV
cache. Measured cold, that is 196 seconds against about 870. It deliberately
does not satisfy the signature above -- it compiles prefill 8, prefill 16, and
decode `(1,16)`. See
"Where the time actually goes, and how to cut it" in
`deepseek-v4-trn2-compilation.md` for the measured breakdown and the reason
the graph is large.
