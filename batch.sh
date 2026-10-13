#!/usr/bin/env bash
# The whole pass on the server. Each stage is skipped when its output already exists (FORCE=1 reruns
# everything), so the script can be restarted after a failure. Both GPUs are used where stages are independent:
#   GPU 1: question vectors for the unanswerable set, gold-context answers (v1, v2), then answers (year+hybrid)
#   GPU 0: search (FinanceBench + unanswerable), retrieved-context answers (v1, v2, dense+year)
#   then judge grading on both GPUs, the judge check against the hand grades, abstention, the report.
#     setsid nohup ./batch.sh > logs/batch.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
CPUS=${CPUS:-0-9}   # at most 10 cores on this server
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
R=results/raw

if venv/bin/python test.py > logs/test.log 2>&1; then log "unit tests ok"; else log "unit tests FAILED"; fi
stage data/chunks.parquet 0 prep || exit 1
stage data/index/dense.npy 0 index || exit 1
stage data/index/queries_unans.npy 1 qembed --set unans || exit 1

( stage $R/gen_gold.jsonl 1 gen --ctx gold
  stage $R/gen_gold_v2.jsonl 1 gen --ctx gold --cfg configs/v2.toml
  until [ -e $R/search_unans.jsonl ]; do sleep 10; done
  stage $R/gen_unans_retrieved_hyr.jsonl 1 gen --set fb,unans --cfg configs/hyr.toml ) &
gpu1=$!
stage $R/search_unans.jsonl 0 search --set fb,unans &&
  stage $R/gen_unans_retrieved.jsonl 0 gen --set fb,unans &&
  stage $R/gen_unans_retrieved_v2.jsonl 0 gen --set fb,unans --cfg configs/v2.toml &&
  stage $R/gen_unans_retrieved_dyr.jsonl 0 gen --set fb,unans --cfg configs/dyr.toml
wait "$gpu1"

( stage $R/grade_gold.jsonl 1 grade --ctx gold
  stage $R/grade_gold_v2.jsonl 1 grade --ctx gold --cfg configs/v2.toml
  stage $R/grade_retrieved_hyr.jsonl 1 grade --cfg configs/hyr.toml
  stage $R/grade_retrieved_prefix.jsonl 1 grade --cfg configs/prefix.toml ) &
stage $R/grade_retrieved.jsonl 0 grade
stage $R/grade_retrieved_v2.jsonl 0 grade --cfg configs/v2.toml
stage $R/grade_retrieved_dyr.jsonl 0 grade --cfg configs/dyr.toml
wait

stage results/agreement_retrieved_prefix.json 0 agree --cfg configs/prefix.toml
for c in mvp v2 hyr dyr; do
  tag=$([ $c = mvp ] && echo "" || echo "_$c")
  stage results/abstain_retrieved$tag.json 0 abstain --cfg configs/$c.toml
done
FORCE=1 stage results/RESULTS.md 0 report
log "all done"
