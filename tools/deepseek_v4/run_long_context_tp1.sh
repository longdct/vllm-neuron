#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

model_path=${1:?usage: run_long_context_tp1.sh CHECKPOINT_DIR ARTIFACT_DIR [LOGICAL_CORE]}
artifact_dir=${2:?usage: run_long_context_tp1.sh CHECKPOINT_DIR ARTIFACT_DIR [LOGICAL_CORE]}
logical_core=${3:-}
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
python_bin=${VLLM_NEURON_PYTHON:-$repo_dir/.venv/bin/python}

mkdir -p "$artifact_dir"
if [[ -z "$logical_core" ]]; then
  logical_core=$($python_bin "$script_dir/select_free_neuron_core.py")
fi

run_target() {
  name=$1
  prefill=$2
  decode=$3
  cache_dir=$artifact_dir/cache-$name
  output=$artifact_dir/$name.json
  log=$artifact_dir/$name.log
  if [[ -e "$cache_dir" ]]; then
    printf 'refusing non-cold target cache: %s\n' "$cache_dir" >&2
    return 2
  fi
  mkdir -p "$cache_dir"
  start=$SECONDS
  /usr/bin/time -v env VLLM_CACHE_ROOT="$cache_dir" \
  NEURON_VISIBLE_DEVICES="$logical_core" \
  VLLM_NEURON_ENABLE_DEEPSEEK_V4=1 \
  NEURON_SKIP_EFA_AFFINITY=1 \
  "$python_bin" "$script_dir/generate_tiny_tp1.py" "$model_path" \
    --output "$output" --load-format auto \
    --max-model-len 131072 --max-num-batched-tokens "$prefill" \
    --prefill-segment-buckets "$prefill" \
    --decode-context-buckets "$decode" "${@:4}" 2>&1 | tee "$log"
  printf 'target wall: %s s\n' "$((SECONDS - start))" >> "$log"
}

# Each shape gets an isolated local-NVMe cache before accepting their union.
for prefill in 512 2048 4096; do
  run_target "prefill-$prefill" "$prefill" 4096 \
    --prompt-length "$prefill" --max-tokens 1
done
for decode in 4096 32768 131072; do
  run_target "decode-$decode" 512 "$decode" \
    --prompt-length "$((decode - 1))" --max-tokens 1
done

combined_cache=$artifact_dir/cache-combined
if [[ -e "$combined_cache" ]]; then
  printf 'refusing non-cold combined cache: %s\n' "$combined_cache" >&2
  exit 2
fi
for pass in cold warm; do
  /usr/bin/time -v env VLLM_CACHE_ROOT="$combined_cache" \
  NEURON_VISIBLE_DEVICES="$logical_core" \
  VLLM_NEURON_ENABLE_DEEPSEEK_V4=1 \
  NEURON_SKIP_EFA_AFFINITY=1 \
  "$python_bin" "$script_dir/generate_tiny_tp1.py" "$model_path" \
    --output "$artifact_dir/combined-$pass.json" --load-format auto \
    --max-model-len 131072 --max-num-batched-tokens 4096 \
    --prefill-segment-buckets 512,2048,4096 \
    --decode-context-buckets 4096,32768,131072 \
    --workload-lengths 512,2048,4096,4095,32767,131071 --max-tokens 1 \
    2>&1 | tee "$artifact_dir/combined-$pass.log"
done

# Preserve the short-context [8,16] cold regression as its own cache.
regression_cache=$artifact_dir/cache-regression-8-16
if [[ -e "$regression_cache" ]]; then
  printf 'refusing non-cold regression cache: %s\n' "$regression_cache" >&2
  exit 2
fi
mkdir -p "$regression_cache"
/usr/bin/time -v env VLLM_CACHE_ROOT="$regression_cache" \
NEURON_VISIBLE_DEVICES="$logical_core" \
VLLM_NEURON_ENABLE_DEEPSEEK_V4=1 \
NEURON_SKIP_EFA_AFFINITY=1 \
"$python_bin" "$script_dir/generate_tiny_tp1.py" "$model_path" \
  --output "$artifact_dir/regression-8-16.json" --load-format auto \
  --max-model-len 16 --max-num-batched-tokens 16 \
  --prefill-segment-buckets 8,16 --decode-context-buckets 16 \
  --prompt-length 8 --max-tokens 4 \
  2>&1 | tee "$artifact_dir/regression-8-16.log"

"$python_bin" "$script_dir/analyze_compile_artifacts.py" \
  --cache-root "$combined_cache" \
  --log "$artifact_dir/combined-cold.log" \
  --log "$artifact_dir/combined-warm.log" \
  > "$artifact_dir/compile-report.json"
