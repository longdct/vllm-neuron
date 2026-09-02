#!/bin/bash
# Run the sampler-only probe across a TP group, one process per rank.
#
# Usage: launch_device_sampling_probe.sh <tp> <cores> <outdir> [extra probe args]
#   tp     number of ranks
#   cores  NEURON_VISIBLE_DEVICES value, e.g. "12-19" -- must be whole devices
#          (four logical cores each on trn2); a group straddling a device
#          boundary fails the runtime barrier before anything is compiled.
#
# Every rank writes its own answer. The ranks disagreeing IS the defect this
# probe exists to catch: a reduction spanning the group cannot return different
# results on different ranks.
set -u
TP=$1; CORES=$2; OUT=$3; shift 3
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$OUT"

expand_cores() {  # "12-19,24" -> "12 13 ... 19 24"
  local out=()
  IFS=',' read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then
      for ((c=${part%-*}; c<=${part#*-}; c++)); do out+=("$c"); done
    else
      out+=("$part")
    fi
  done
  echo "${out[@]}"
}
read -ra CORE_LIST <<< "$(expand_cores "$CORES")"
if [ "${#CORE_LIST[@]}" -ne "$TP" ]; then
  echo "cores '$CORES' expand to ${#CORE_LIST[@]} entries, need $TP" >&2
  exit 2
fi

export PATH=/home/ubuntu/vllm-neuron/.venv/bin:$PATH
export PYTHONPATH=$WT${PYTHONPATH:+:$PYTHONPATH}
export NEURON_SKIP_EFA_AFFINITY=1
export NEURON_VISIBLE_DEVICES=$CORES
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29650}
export WORLD_SIZE=$TP

pids=()
for ((r=0; r<TP; r++)); do
  RANK=$r LOCAL_RANK=$r NEURON_RT_VISIBLE_CORES=${CORE_LIST[$r]} \
  VLLM_CACHE_ROOT="$OUT/cache" \
    python "$WT/tools/deepseek_v4/probe_device_sampling.py" \
      --tensor-parallel-size "$TP" --output "$OUT/probe.json" "$@" \
      > "$OUT/rank$r.log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done

python - "$OUT" "$TP" <<'PY'
import json, sys
from pathlib import Path
out, tp = Path(sys.argv[1]), int(sys.argv[2])
reports = []
for r in range(tp):
    f = out / f"probe-rank{r}.json"
    if not f.exists():
        print(f"rank {r}: NO RESULT (see {out}/rank{r}.log)")
        continue
    reports.append(json.loads(f.read_text()))
if not reports:
    sys.exit("no ranks produced a result")
expected = reports[0]["expected_tokens"]
answers = {tuple(r["device_tokens"]) for r in reports}
print(f"ranks reporting : {len(reports)}/{tp}")
print(f"expected tokens : {expected}")
for r in reports:
    print(f"  rank {r['rank']:2d}: {r['device_tokens']}  shards={r['owning_shard_of_each_token']}"
          f"  {'ok' if r['matches_expected'] else 'MISMATCH'}")
print(f"ranks agree     : {len(answers) == 1}")
print(f"all correct     : {all(r['matches_expected'] for r in reports)}")
sys.exit(0 if len(answers) == 1 and all(r["matches_expected"] for r in reports) else 1)
PY
rc=$?
[ $fail -ne 0 ] && echo "(at least one rank exited non-zero)"
exit $rc
