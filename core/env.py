"""Paths, environment defaults and GPU choice. Imported by run.py before torch or vLLM."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FB = DATA / "financebench"
INDEX = DATA / "index"
RESULTS = BASE / "results"
RAW = RESULTS / "raw"
MAX_CPUS = 10


def set_defaults() -> None:
    """Every cache under BASE, weights read offline, vLLM's sampler and torch's compiler kept in-process."""
    d = os.environ.setdefault
    d("HF_HOME", str(BASE / "hf"))
    d("HF_HUB_OFFLINE", "1")
    d("TOKENIZERS_PARALLELISM", "false")
    d("VLLM_USE_FLASHINFER_SAMPLER", "0")   # FlashInfer's top-p kernel JIT-builds and needs nvcc (none here)
    d("VLLM_CACHE_ROOT", str(BASE / "cache" / "vllm"))
    d("TORCHINDUCTOR_CACHE_DIR", str(BASE / "cache" / "inductor"))
    d("TORCHINDUCTOR_COMPILE_THREADS", "1")  # no compile-worker subprocesses
    d("TRITON_CACHE_DIR", str(BASE / "cache" / "triton"))
    d("XDG_CACHE_HOME", str(BASE / "cache"))
    d("TMPDIR", str(BASE / "tmp"))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        d(var, str(MAX_CPUS))


def gpus() -> list[dict]:
    """Every GPU with its free and total memory (MiB), most free first."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    rows = [dict(zip(("index", "free", "total"), map(int, line.split(",")))) for line in out.strip().splitlines()]
    return sorted(rows, key=lambda g: -g["free"])


def pick_gpu(need_mib: int, margin_mib: int = 768, cap: float = 0.92) -> tuple[int, float]:
    """GPU with the most free memory (or the one named by $GPU), and the share of its total memory this
    process may take.

    Other people's processes may share the GPU, so the share is bounded by what is free now (less a
    margin), not by the total. Raises if the GPU cannot hold `need_mib`."""
    all_gpus = gpus()
    want = os.environ.get("GPU")
    g = next(x for x in all_gpus if x["index"] == int(want)) if want else all_gpus[0]
    usable = g["free"] - margin_mib
    if usable < need_mib:
        raise RuntimeError(f"no GPU has {need_mib} MiB free (best: GPU {g['index']} with {g['free']} MiB)")
    return g["index"], round(min(cap, usable / g["total"]), 3)


def use_gpu(index: int) -> None:
    """Must run before torch initialises CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
