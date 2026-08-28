# Model development

Onboard new model architectures, debug accuracy issues, and develop without hardware. For developers adding or validating models on vLLM Neuron.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Onboard a new model
:link: onboarding-models
:link-type: doc

Implement and register a new architecture with vLLM.
:::

:::{grid-item-card} Onboard a vision-language model
:link: onboarding-vlm-models
:link-type: doc

Add a vision encoder tower on top of the text-decoder flow.
:::

:::{grid-item-card} Optimizing a vision-language model
:link: optimizing-vlm-models
:link-type: doc

Roofline, sharding, and profiling to optimize a vision-language model.
:::

:::{grid-item-card} Neuron lowering pitfalls
:link: neuron-lowering-pitfalls
:link-type: doc

Torch ops that are correct on CPU and wrong on Neuron, and how to find more.
:::

:::{grid-item-card} CPU/Neuron op divergences
:link: neuron-cpu-op-divergences
:link-type: doc

Quick-reference inventory of every op known to differ, and the safe form.
:::

:::{grid-item-card} Which Neuron stack?
:link: neuron-lowering-stacks
:link-type: doc

Two things are called "torch 2.12" and they disagree; how to reproduce each.
:::

:::{grid-item-card} CPU development workflow
:link: cpu-development
:link-type: doc

Develop and test without Neuron hardware.
:::

:::{grid-item-card} NKI CPU simulator
:link: nki_cpu_simulator
:link-type: doc

Validate NKI kernel correctness on CPU.
:::

:::{grid-item-card} Debugging model code
:link: debugging
:link-type: doc

Use pdb and print statements to inspect model execution.
:::

:::{grid-item-card} Debugging accuracy issues
:link: accuracy-debugging-guide
:link-type: doc

Methodology for isolating where accuracy drift is introduced.
:::

:::{grid-item-card} Accuracy debugger tools
:link: how-to-use-accuracy-debugger
:link-type: doc

Run the automated debugger pipeline and interpret results.
:::

::::

:::{toctree}
:maxdepth: 1
:hidden:

Onboarding a model <onboarding-models>
Onboarding a vision-language model <onboarding-vlm-models>
Optimizing a vision-language model <optimizing-vlm-models>
CPU development workflow <cpu-development>
NKI CPU simulator <nki_cpu_simulator>
Debugging model code <debugging>
Neuron lowering pitfalls <neuron-lowering-pitfalls>
CPU/Neuron op divergences <neuron-cpu-op-divergences>
Which Neuron stack? <neuron-lowering-stacks>
Debugging accuracy issues <accuracy-debugging-guide>
Accuracy debugger tools <how-to-use-accuracy-debugger>
:::
