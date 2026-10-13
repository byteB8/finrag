#!/usr/bin/env bash
# Start/stop the service: the vLLM model servers of configs/serve.toml (one at a time, since two engines
# starting on one GPU together miscount each other's memory), then the HTTP API. Everything listens on
# 127.0.0.1 only.
#   ./serve.sh up | down | status
# From a laptop:  ssh -N -L 8010:127.0.0.1:8010 <server>   then open http://127.0.0.1:8010/docs
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs tmp
CPUS=${CPUS:-0-9}   # at most 10 cores on this server

servers() {   # "name gpu port" per model server, in start order
  venv/bin/python -c 'from core.config import load_cfg; c = load_cfg("configs/serve.toml")
for n, s in c["servers"].items(): print(n, s["gpu"], s["port"])'
}
api_port() { venv/bin/python -c 'from core.config import load_cfg; print(load_cfg("configs/serve.toml")["api"]["port"])'; }
alive() { [ -f "logs/$1.pid" ] && kill -0 "$(cat "logs/$1.pid")" 2>/dev/null; }

wait_ready() {   # wait_ready <name> <url>
  for _ in $(seq 1 120); do
    if curl -sf "$2" >/dev/null; then return 0; fi
    alive "$1" || { echo "FAILED: $1 exited"; tail -20 "logs/$1.log"; return 1; }
    sleep 5
  done
  echo "FAILED: $1 not ready in 10 min"; return 1
}

up() {
  while read -r name gpu port; do
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; then echo "$name already up (port $port)"; continue; fi
    CUDA_VISIBLE_DEVICES=$gpu SRV_NAME=$name setsid nohup taskset -c "$CPUS" venv/bin/python main.py \
      > "logs/srv_$name.log" 2>&1 < /dev/null &
    echo $! > "logs/srv_$name.pid"
    wait_ready "srv_$name" "http://127.0.0.1:$port/v1/models" || exit 1
    echo "$name up on GPU $gpu, port $port"
  done < <(servers)
  local port
  port=$(api_port)
  if ! curl -sf "http://127.0.0.1:$port/health" >/dev/null; then
    setsid nohup taskset -c "$CPUS" venv/bin/python api.py > logs/srv_api.log 2>&1 < /dev/null &
    echo $! > logs/srv_api.pid
    wait_ready srv_api "http://127.0.0.1:$port/health" || exit 1
  fi
  curl -s "http://127.0.0.1:$port/health"; echo
}

down() {
  for f in logs/srv_*.pid; do
    [ -f "$f" ] || continue
    pid=$(cat "$f")
    kill "$pid" 2>/dev/null && echo "stopped $(basename "$f" .pid) ($pid)"
    rm -f "$f"
  done
}

status() {
  for f in logs/srv_*.pid; do
    [ -f "$f" ] || { echo "nothing running"; return; }
    name=$(basename "$f" .pid)
    if kill -0 "$(cat "$f")" 2>/dev/null; then echo "$name: running"; else echo "$name: dead"; fi
  done
}

case "${1:-status}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) echo "usage: $0 up|down|status"; exit 1 ;;
esac
