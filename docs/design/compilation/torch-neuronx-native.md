# TorchNeuron Native backend

vLLM Neuron now uses the Native backend supplied by the external
`torch-neuronx` package. The package name and the backend name are intentionally
different: install `torch-neuronx`, then compile with
`torch.compile(..., backend="neuron", fullgraph=True)`.

The tested development environment is:

- `torch==2.12.1`
- `torch-neuronx==2.12.3.0.0+aa8779f4.dev`
- `vllm==0.24.0`

Install Torch and TorchNeuron first. Install vLLM and this project's remaining
dependencies without allowing dependency resolution to replace Torch.
`torch-neuronx` is deliberately not declared in `requirements/core.txt`: it is
provided by the Neuron SDK environment.

## Compilation and cache lifecycle

There is no graph-capture sidecar, forked trace pool, or parallel HLO compiler.
The text model and each vision component are wrapped once with the Native
backend. Startup then calls every configured prefill, decode, and vision bucket
in a deterministic order. Those calls lower through Dynamo/AOTAutograd and
materialize NEFFs; subsequent calls reuse TorchNeuron's HLO and NEFF caches.

Compiler caches are rooted at the existing vLLM Neuron cache directory and
isolated as `rank_<rank>/{inductor,hlo,neff}`. A cold run is expected to log
compile/cache-miss activity. A second run with the same cache root must report
cache hits and produce identical outputs.

`VLLM_NEURON_CPU_COMPILE=1` enables TorchNeuron Native compile-only mode via
`TORCH_NEURONX_COMPILE_ONLY=1`. `VLLM_NEURON_CPU_MODE=1` remains the CPU eager
development mode. The two modes are mutually exclusive.

## Configuration migration

An unset `VLLM_NEURON_BACKEND` selects Native. `neuron_native` is temporarily
accepted as an alias. `vllm_neuron` is rejected because the lite/XLA route has
been removed.

Native compile options are limited to a stable `model_name` and the tensorizer
toggle. Optimization level is translated to `NEURON_COMPILER_OPT_LEVEL`, and
compiler flags use `NEURON_CC_FLAGS`. Lite remote-cache, graph-capture, and
parallel-trace settings are rejected with migration guidance.

DeepSeek-V4 tiny TP1 is the device acceptance configuration for this migration.
Other registered model families retain import and graph-construction support,
but are not device-certified by this change. Existing lowering-divergence
workarounds remain until Native characterization tests explicitly clear them.
