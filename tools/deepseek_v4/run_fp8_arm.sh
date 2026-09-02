#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Run one DeepSeek-V4 decode-benchmark arm in BF16 or FP8 on TRN2.
#
# This exists because the FP8 path needs four pieces of environment that are
# not discoverable from the errors they produce, and getting any one of them
# wrong fails in a way that points somewhere else:
#
#   UNSAFE_FP8FNCAST=1
#       torch only has e4m3fn; TRN2 implements e4m3 (max 240). Without this the
#       NKI tracer refuses the dtype. Safe here because the converter quantizes
#       against 240, so no value can leave the legacy format's range.
#
#   NEURON_CC_FLAGS=--internal-hlo2tensorizer-options=--experimental-unsafe-fp8e4m3fn-as-fp8e4m3
#       neuronx-cc rejects F8E4M3FN in the HLO (NCC_EVRF051). Its own error
#       recommends the bare flag, which this compiler build then rejects as
#       unrecognized -- it is only reachable through the pass-through.
#
#   PYTHONPATH=<repo root>
#       The editable install points at the primary checkout, so vLLM's worker
#       processes import THAT tree unless this names the one under test. A run
#       from a worktree otherwise silently measures different code; that
#       mistake invalidated an entire A/B before it was caught.
#
#   --enable-expert-parallel above TP8
#       FP8 experts run on `shard_on_i`, which needs I_TP >= 256. At TP16 with
#       no EP, I_TP is 128 and decode graph extraction dies with NCC_IBIR243
#       "Access pattern out of bounds. Pattern: [[256,128],[1,256]]".
#
# Sampling defaults to cpu: device sampling on this branch emits token IDs
# outside [0, vocab) and kills the request in _validate_token_ids. It
# reproduces in BF16, so it is not an FP8 defect -- but both arms of a
# comparison must use the same backend regardless.
#
# Usage: run_fp8_arm.sh <output-dir> <checkpoint> <bf16|fp8> [extra benchmark args...]
set -euo pipefail

OUT=${1:?output directory}; CKPT=${2:?checkpoint directory}; QUANT=${3:?bf16 or fp8}
shift 3
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${VENV:-/home/ubuntu/vllm-neuron/.venv}

mkdir -p "$OUT"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_CACHE_ROOT="$OUT/cache"
export VLLM_NEURON_ENABLE_DEEPSEEK_V4=1
export NEURON_SKIP_EFA_AFFINITY=1
# Logical CORE ids, not device indices, and exactly TP of them.
export NEURON_VISIBLE_DEVICES=${CORES:-12-19}
# The normal CSA selection graph still does not make forward progress; every
# measurement below is diagnostic, not an acceptance result.
export VLLM_NEURON_DSV4_FIXED_CSA_SELECTION=${FIXED_CSA:-1}
if [ "$QUANT" = "fp8" ]; then
  export UNSAFE_FP8FNCAST=1
  export NEURON_CC_FLAGS="--internal-hlo2tensorizer-options=--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"
fi

cd "$REPO"
exec "$VENV/bin/python" tools/deepseek_v4/benchmark_decode.py "$CKPT" \
  --output "$OUT/result.json" --load-format "${LOAD_FORMAT:-dummy}" \
  --tensor-parallel-size "${TP:-8}" --ep-degree "${EP:-4}" \
  --sampling-backend "${SAMPLING:-cpu}" \
  --workload sustained --query-bucket 512 --max-model-len 512 \
  --decode-context-buckets 512 --block-size 256 \
  --num-gpu-blocks-override 512 --gpu-memory-utilization 0.9 \
  --warmups 2 --repetitions 3 --max-output-tokens 128 \
  --quantization "$QUANT" "$@"
