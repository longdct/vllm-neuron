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
unset VLLM_NEURON_CPU_MODE VLLM_NEURON_CPU_COMPILE
mkdir -p "$cache_path"
"$python_bin" "$script_dir/write_device_preflight.py" \
  "$cache_path/device-preflight.json" --cache-root "$cache_path"

exec "$vllm_bin" serve "$model_path" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 64 \
  --max-num-seqs 1 \
  --block-size 32 \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --num-gpu-blocks-override 256 \
  --load-format dummy \
  --port "$port" \
  --additional-config '{"neuron_config":{"num_batched_tokens_buckets":[8,64],"on_device_sampling_config":null}}'
