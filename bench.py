"""Latency of the HTTP API (started by serve.sh) on the 150 FinanceBench questions.

    python bench.py [--clients 8] [--limit 0]

First one question at a time (per-stage timings from the API, plus wall time), then --clients concurrent
clients over the same questions (throughput and wall-time percentiles). Writes results/latency.json.
"""
import argparse
import asyncio
import json
import time

import httpx
import numpy as np

from core import data, env
from core.config import load_cfg


def pct(values, q) -> float:
    return round(float(np.percentile(values, q)), 1)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{load_cfg(env.BASE / 'configs/serve.toml')['api']['port']}/ask"
    qs = [q["question"] for q in data.load_questions(env.FB)][: args.limit or None]

    async with httpx.AsyncClient(timeout=600) as client:
        async def one(q):
            t = time.perf_counter()
            r = await client.post(url, json={"question": q})
            r.raise_for_status()
            return r.json(), 1000 * (time.perf_counter() - t)

        await one(qs[0])   # warm-up (prefix cache, CUDA graphs)
        seq = [await one(q) for q in qs]
        stages = {k: [res["timing_ms"][k] for res, _ in seq] for k in seq[0][0]["timing_ms"]}
        walls = [w for _, w in seq]

        sem = asyncio.Semaphore(args.clients)

        async def limited(q):
            async with sem:
                return await one(q)

        t = time.perf_counter()
        conc = await asyncio.gather(*(limited(q) for q in qs))
        elapsed = time.perf_counter() - t

    out = {"questions": len(qs),
           "sequential": {"p50_ms": {k: pct(v, 50) for k, v in stages.items()},
                          "p95_ms": {k: pct(v, 95) for k, v in stages.items()},
                          "wall_p50_ms": pct(walls, 50), "wall_p95_ms": pct(walls, 95)},
           "concurrent": {"clients": args.clients, "seconds": round(elapsed, 1),
                          "questions_per_s": round(len(qs) / elapsed, 2),
                          "wall_p50_ms": pct([w for _, w in conc], 50), "wall_p95_ms": pct([w for _, w in conc], 95)},
           "abstained": sum(r["abstained"] for r, _ in seq),
           "valid_citations": sum(r["checks"]["valid_citations"] for r, _ in seq)}
    (env.RESULTS / "latency.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
