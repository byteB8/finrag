"""Download model weights (into $HF_HOME) and the FinanceBench repository (into data/financebench).

    HF_HOME=hf python fetch.py

Small models first, so retrieval work can start while the large ones are still downloading.
"""
import subprocess
import sys
import time
from pathlib import Path

MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3-14B-AWQ",
]
DATA_REPO = "https://github.com/patronus-ai/financebench.git"


def main() -> None:
    from huggingface_hub import snapshot_download

    dest = Path("data/financebench")
    if not dest.exists():
        t = time.time()
        subprocess.run(["git", "clone", "-q", "--depth", "1", DATA_REPO, str(dest)], check=True)
        print(f"data: {dest} in {time.time() - t:.0f}s", flush=True)
    for name in MODELS:
        t = time.time()
        path = snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        print(f"model: {name} -> {path} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
