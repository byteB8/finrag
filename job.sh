#!/usr/bin/env bash
# Start a run.py stage detached on the server, pinned to the allowed CPUs, with one log per stage.
#   ./job.sh <stage> [args...]    e.g. ./job.sh prep   ./job.sh gen --ctx gold
#   ./job.sh status               running jobs and the end of the newest log
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs tmp
CPUS=${CPUS:-0-9}   # at most 10 cores on this server
if [ "${1:-}" = status ]; then
  pgrep -u "$(id -u)" -af "python run.py" || echo "no job running"
  f=$(ls -t logs/*.log 2>/dev/null | head -1 || true)
  if [ -n "$f" ]; then echo "== $f"; tail -n 15 "$f"; fi
  exit 0
fi
log=logs/$(IFS=_; echo "$*" | tr -c 'a-zA-Z0-9_\n-' '_').log
setsid nohup taskset -c "$CPUS" venv/bin/python run.py "$@" > "$log" 2>&1 < /dev/null &
echo "started: run.py $* (pid $!, CPUs $CPUS) -> $log"
