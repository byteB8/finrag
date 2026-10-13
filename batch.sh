#!/usr/bin/env bash
# The whole MVP pass on the server. Each stage is skipped when its output already exists (FORCE=1 reruns
# everything), so the script can be restarted after a failure. Both GPUs are used where stages are independent:
#   GPU 0: index -> search -> gen retrieved (v1, v2)        GPU 1: gen gold (v1, v2)
#   then grading on both GPUs once the judge weights are downloaded, the hand-grading sheet, the report.
#     setsid nohup ./batch.sh > logs/batch.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
CPUS=${CPUS:-0-9}   # at most 10 cores on this server
V2=configs/v2.toml
log() { echo "$(date +%H:%M:%S) $*"; }

stage() {   # stage <output> <gpu> <run.py args...>: run unless <output> exists, log to logs/<args>.log
  local out=$1 gpu=$2 name
  shift 2
  if [ -e "$out" ] && [ -z "${FORCE:-}" ]; then log "skip  $* ($out exists)"; return 0; fi
  name=$(IFS=_; echo "$*" | tr -c 'a-zA-Z0-9_\n-' '_')
  log "start $* (GPU $gpu)"
  if GPU=$gpu GPUS=${GPUS:-} taskset -c "$CPUS" venv/bin/python run.py "$@" > "logs/$name.log" 2>&1; then
    log "done  $*"
  else
    log "FAILED $* (logs/$name.log)"
    return 1
  fi
}

if venv/bin/python test.py > logs/test.log 2>&1; then log "unit tests ok"; else log "unit tests FAILED"; fi
stage data/chunks.parquet 0 prep || exit 1

( stage results/raw/gen_gold.jsonl 1 gen --ctx gold
  stage results/raw/gen_gold_v2.jsonl 1 gen --ctx gold --cfg $V2 ) &
gold=$!
stage data/index/dense.npy 0 index &&
  stage results/raw/search.jsonl 0 search &&
  stage results/raw/gen_retrieved.jsonl 0 gen --ctx retrieved &&
  stage results/raw/gen_retrieved_v2.jsonl 0 gen --ctx retrieved --cfg $V2
wait "$gold"
stage results/review/sheet_retrieved.html 0 sheet --ctx retrieved

until grep -q "^model: Qwen/Qwen3-14B-AWQ" logs/fetch.log 2>/dev/null; do sleep 30; done
( stage results/raw/grade_gold.jsonl 1 grade --ctx gold
  stage results/raw/grade_gold_v2.jsonl 1 grade --ctx gold --cfg $V2 ) &
stage results/raw/grade_retrieved.jsonl 0 grade --ctx retrieved
stage results/raw/grade_retrieved_v2.jsonl 0 grade --ctx retrieved --cfg $V2
wait
FORCE=1 stage results/RESULTS.md 0 report
log "all done"
