#!/usr/bin/env bash
# One-time server setup, entirely under this directory (no root): uv, Python 3.11, a venv with vLLM,
# then model weights and the FinanceBench data.
#   ./setup.sh            (run on the server; logs to logs/setup.log when started by sync.sh users)
#
# The host driver (570) speaks CUDA 12.8, so vLLM's default CUDA 13 wheels cannot run here. The +cu129
# build with torch from the cu129 index runs through CUDA minor-version compatibility. On this network
# GitHub release assets and Hugging Face's Xet transfers can stall, while vLLM's own wheel index and
# plain HTTPS from Hugging Face (both CloudFront) stay fast; PyPI is slow but usable with parallel downloads.
set -euo pipefail
cd "$(dirname "$0")"
BASE=$(pwd)
mkdir -p tools tmp logs hf cache data
export UV_CACHE_DIR=$BASE/cache/uv UV_PYTHON_INSTALL_DIR=$BASE/tools/python TMPDIR=$BASE/tmp
export UV_CONCURRENT_DOWNLOADS=32 UV_HTTP_TIMEOUT=300
UV=$BASE/tools/uv

# 1. uv, Python 3.11, venv
if [ ! -x "$UV" ]; then
  curl -sfL https://github.com/astral-sh/uv/releases/download/0.8.0/uv-x86_64-unknown-linux-gnu.tar.gz \
    | tar xz -C tools --strip-components=1
fi
"$UV" python install 3.11
[ -x venv/bin/python ] || "$UV" venv venv --python 3.11

# 2. weights and data download in the background while the packages install
"$UV" pip install --python venv/bin/python huggingface_hub
HF_HOME=$BASE/hf HF_HUB_DISABLE_XET=1 nohup venv/bin/python fetch.py > logs/fetch.log 2>&1 &
FETCH=$!

# 3. vLLM (+cu129) and torch (cu129), then this project's few extra packages
"$UV" pip install --python venv/bin/python "vllm==0.31.0+cu129" "transformers==5.17.0" \
  --extra-index-url https://wheels.vllm.ai/0.31.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
"$UV" pip install --python venv/bin/python -r requirements.txt
# torchcodec is built for CUDA 13 and fails with OSError (not ImportError) on import; vLLM only needs it for audio
"$UV" pip uninstall --python venv/bin/python torchcodec || true

venv/bin/python -c "import torch, vllm; assert torch.cuda.is_available(); \
print('ok torch', torch.__version__, 'vllm', vllm.__version__, torch.cuda.get_device_name(0))"
wait "$FETCH" && echo "fetch done"
