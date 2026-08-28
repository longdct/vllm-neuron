#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

model_path=${1:?usage: run_tiny_tp1.sh CHECKPOINT_DIR [CACHE_DIR] [PORT] [LOGICAL_CORE]}
cache_path=${2:-/tmp/deepseek-v4-tiny-neuron-cache}
port=${3:-8001}
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
python_bin=${VLLM_NEURON_PYTHON:-$repo_dir/.venv/bin/python}
vllm_bin=${VLLM_NEURON_VLLM:-$repo_dir/.venv/bin/vllm}
export PATH="$(dirname -- "$python_bin"):$PATH"
logical_core=${4:-}
if [[ -z "$logical_core" ]]; then
  logical_core=$($python_bin "$script_dir/select_free_neuron_core.py")
fi

export VLLM_NEURON_ENABLE_DEEPSEEK_V4=1
export VLLM_NEURON_VALIDATE_CACHE_METADATA=1
export VLLM_CACHE_ROOT="$cache_path"
export NEURON_VISIBLE_DEVICES="$logical_core"
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
# The prefill-64 graph takes 689-835s in neuronx-cc, longer than the 600s
# default a lock waiter will poll for another process's compile. That does not
# bite the single-rank in-process path, but it does for TP>1, concurrent runs,
# and the warmup path -- so raise it rather than leaving a latent timeout.
export NEURON_LIBTORCH_COMPILATION_TIMEOUT=${NEURON_LIBTORCH_COMPILATION_TIMEOUT:-1800}
unset VLLM_NEURON_CPU_MODE VLLM_NEURON_CPU_COMPILE

# Fast-iteration profile. Off by default: the gate's accepted-run signature
# (three NEFFs cold, three hits / zero submitted HLOs warm) is defined against
# the values below it, so this must never change what a plain invocation runs.
#
# The accepted compile-time profile uses the portable packed-attention,
# de-duplicated MoE, direct shared-latent MLA, and indexed cache-update graph.
# These overrides keep the official acceptance geometry reproducible:
#
#   max-model-len 16       fewer unrolled token bodies. Measured 16min -> ~3min
#                          in docs/model-dev/deepseek-v4-tiny-tp1-neuron-investigation.md,
#                          which also records that length 16 still reproduced
#                          the original fourth-token mismatch exactly.
#   num-gpu-blocks 32      32 is the accepted cache geometry and smallest safe
#                          value: the runner logs
#                          max_num_blocks_per_req=[1, 2, 2, 2, 8, 16], so the
#                          override must clear 16 plus a null block. Re-read
#                          that log line before lowering it further.
if [[ ${VLLM_NEURON_TINY_FAST:-0} == 1 ]]; then
  max_model_len=16
  num_gpu_blocks=32
  batched_tokens_buckets='[8,16]'
else
  max_model_len=64
  num_gpu_blocks=256
  batched_tokens_buckets='[8,64]'
fi

mkdir -p "$cache_path"
"$python_bin" "$script_dir/write_device_preflight.py" \
  "$cache_path/device-preflight.json" --cache-root "$cache_path"

exec "$vllm_bin" serve "$model_path" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len "$max_model_len" \
  --max-num-seqs 1 \
  --block-size 32 \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --num-gpu-blocks-override "$num_gpu_blocks" \
  --load-format dummy \
  --port "$port" \
  --additional-config "{\"neuron_config\":{\"num_batched_tokens_buckets\":$batched_tokens_buckets,\"on_device_sampling_config\":null}}"
