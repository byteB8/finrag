#!/usr/bin/env bash
# The server holds everything (venv, model weights, data, runs); this folder holds the code and a copy
# of the results.
#   ./sync.sh push    code -> server
#   ./sync.sh pull    results <- server (files over 200 MB stay on the server)
# Remote is read from .remote (gitignored), e.g.:  REMOTE=host  PORT=22  DIR=work/exp
#
# `push` sends an explicit whitelist of code paths and deletes only inside the code directories, so
# server-side state can never be removed by a sync.
set -euo pipefail
cd "$(dirname "$0")"
source .remote
SSH="ssh -p ${PORT:-22}"
case "${1:-push}" in
  push) ssh -p "${PORT:-22}" "$REMOTE" "mkdir -p ${DIR}"
        rsync -az -e "$SSH" --delete --exclude __pycache__ core configs tests "${REMOTE}:${DIR}/"
        rsync -az -e "$SSH" ./*.py ./*.sh requirements.txt "${REMOTE}:${DIR}/" ;;
  pull) rsync -az -e "$SSH" --max-size=200m "${REMOTE}:${DIR}/results/" results/ ;;
  *) echo "usage: $0 push|pull"; exit 1 ;;
esac
