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
