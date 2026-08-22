# Tiny DeepSeek-V4 TP1 correctness deployment

Build the deterministic three-layer, 64-token checkpoint and tokenizer:

```bash
.venv/bin/python tools/deepseek_v4/build_tiny_checkpoint.py /tmp/deepseek-v4-tiny
```

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

A cold accepted run has exactly three HLO/NEFF entries: prefill 8, prefill 64,
and decode `(1,64)`. A restart must report three cache hits and zero submitted
HLOs. Reject PJRT faults, DGE/scatter/gather out-of-bounds messages, allocator
assertions, tracebacks, and worker crashes. This gate does not qualify the full
checkpoint, TP>1, longer contexts, concurrency, or performance.
