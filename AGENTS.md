# Repository agent notes

## DeepSeek-V4 development

Before changing, compiling, profiling, or testing DeepSeek-V4 code, read both:

- `docs/model-dev/deepseek-v4-development-environment-handoff.md` for the exact
  host and Python/Neuron stacks, executable paths, device mapping, cache
  discipline, retained checkpoints/results, and development test ladder;
- `docs/model-dev/deepseek-v4-q512-mla-compile-explosion.md` for the open Q512
  MLA compiler graph-explosion defect, reproduction evidence, unsafe full cold
  run, recommended implementation, and acceptance criteria.

Use `/home/ssm-user/.venv-torch-neuronx-dev` for the recorded TorchNeuron Native
baseline. Do not mix it with the packaged `/opt` XLA environment. Preserve the
dirty worktree, use isolated local caches for cold measurements, and do not
repeat the known broken full Q512 TP2/EP2 cold compile on the 124 GiB host until
the MLA structural checks show that static query expansion and query-by-history
materializations have been removed.

## This host's dev environment (trn2.48xlarge, 2026-08-28)

Working TorchNeuron Native venv: `/home/ubuntu/vllm-neuron/.venv` (Python 3.12).
Put `.venv/bin` on `PATH`; never mix with the `/opt` XLA venv. Snapshot in
`.venv/setup-snapshot/`; full build recipe in
`/home/ubuntu/.claude/plans/reactive-snacking-nest.md`.

Stack: torch 2.12.1+cpu, torch-neuronx 2.12.3.0.0 (built editable from
`/home/ubuntu/torch-neuronx-main`), vllm 0.24.0, transformers 5.15.1,
neuronx-cc 2.27.5334.0, nki 0.6.0.

Two fixes were required and must persist:
- `torch-neuronx-main/torch_neuronx/csrc/core/streams/StreamImpl.cpp` does not
  compile as shipped; 2 lines patched to match its headers
  (`recorded_stream_id() != stream_id`, `IsSameStreamWaitBypass(op.get(), stream_id)`).
  Reapply if that tree is re-extracted.
- Pin `islpy==2026.1`; `2026.2.1` crashes `neuronx-cc` (`NCC_ISMP902 is_subset`).

Install torch-neuronx with `pip install -e` (a plain wheel drops `_C.so`).
Expected-benign import noise: `neg_kernel` UserWarning, `aten::div_.Tensor`
override warning. `pip check` still flags vllm's torch==2.11.0 pin — benign.
