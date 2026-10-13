#!/usr/bin/env bash
# The whole MVP pass on the server (prep first if it has not run), using both GPUs where stages are independent:
#   GPU 0: index -> search -> gen (retrieved chunks)      GPU 1: gen (labelled evidence pages)
#   then grading on both GPUs once the judge weights are downloaded, then the report.
#     setsid nohup ./batch.sh > logs/batch.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
CPUS=${CPUS:-0-9}   # at most 10 cores on this server
log() { echo "$(date +%H:%M:%S) $*"; }

stage() {   # stage <gpu> <run.py args...>: one stage in the foreground, its output in logs/<args>.log
  local gpu=$1 name
  shift
  name=$(IFS=_; echo "$*" | tr -c 'a-zA-Z0-9_\n-' '_')
  log "start $* (GPU $gpu)"
  if GPU=$gpu taskset -c "$CPUS" venv/bin/python run.py "$@" > "logs/$name.log" 2>&1; then
    log "done  $*"
  else
    log "FAILED $* (logs/$name.log)"
    return 1
  fi
}

if venv/bin/python test.py > logs/test.log 2>&1; then log "unit tests ok"; else log "unit tests FAILED"; fi
[ -f data/chunks.parquet ] || stage 0 prep || exit 1
( stage 1 gen --ctx gold ) &
gold=$!
stage 0 index && stage 0 search && stage 0 gen --ctx retrieved
wait "$gold"

until grep -q "^model: Qwen/Qwen3-14B-AWQ" logs/fetch.log 2>/dev/null; do sleep 30; done
( stage 1 grade --ctx gold ) &
stage 0 grade --ctx retrieved
wait
stage 0 report
log "all done"
