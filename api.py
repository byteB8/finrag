"""HTTP API for cited answers over the filings.

Retrieval runs in this process (BM25, dense vectors and the company/year filters, all in memory); embedding,
reranking and generation are calls to the vLLM servers that serve.sh starts (configs/serve.toml).

    POST /ask          {"question": "...", "company": null, "prompt": "v1"}  ->  answer, citations, timings
    POST /ask/stream   same request; Server-Sent Events: retrieval, then the answer JSON as it is generated
                       ("token" events), then the full result
    GET  /health       which model servers answer

Started by serve.sh as `python api.py` (uvicorn on 127.0.0.1, port from configs/serve.toml).
"""
import os


def _pin_process_title() -> None:
    title = os.environ.get("PROC_TITLE", "python api.py")
    try:
        import setproctitle
    except ImportError:
        return
    setproctitle.setproctitle(title)


_pin_process_title()

import json  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from core import answer, data, env, llm  # noqa: E402
from core.config import load_cfg  # noqa: E402
from core.search import BM25, EOT, RERANK_PREFIX, RERANK_SUFFIX, entity_masks, rrf, top_k  # noqa: E402

env.set_defaults()
SERVE = load_cfg(env.BASE / "configs/serve.toml")
CFG = load_cfg(env.BASE / SERVE["api"]["cfg"])
URL = {name: f"http://127.0.0.1:{s['port']}" for name, s in SERVE["servers"].items()}
S: dict = {}   # in-memory index and clients, filled at start-up


class Ask(BaseModel):
    question: str
    company: str | None = None   # restrict to one company's filings (a corpus company name), like a UI filter
    prompt: str = "v1"


@asynccontextmanager
async def lifespan(_app):
    from transformers import AutoTokenizer
    chunks = pd.read_parquet(env.DATA / "chunks.parquet")
    tok = AutoTokenizer.from_pretrained(CFG["embed"]["model"])
    S.update(chunks=chunks, texts=data.chunk_texts(chunks), company=chunks.company.to_numpy(),
             period=chunks.period.astype(str).to_numpy(), dense=np.load(env.INDEX / "dense.npy").astype(np.float32),
             bm=BM25.load(env.INDEX / "bm25"), tok=tok, eot=tok.convert_tokens_to_ids(EOT),
             keys=[f"{d}:{p}:{k}" for d, p, k in zip(chunks.doc_name, chunks.page, chunks.part)],
             http=httpx.AsyncClient(timeout=180))
    yield
    await S["http"].aclose()


app = FastAPI(title="finrag", lifespan=lifespan)


async def _post(server: str, path: str, body: dict) -> dict:
    r = await S["http"].post(URL[server] + path, json=body)
    if r.status_code != 200:
        raise HTTPException(502, f"{server} server: {r.status_code} {r.text[:300]}")
    return r.json()


async def embed(question: str) -> np.ndarray:
    """Same input as the offline index: instruction prefix, then the end-of-text token that is pooled."""
    e = CFG["embed"]
    ids = S["tok"](f"Instruct: {e['query_instruction']}\nQuery:{question}", add_special_tokens=False)["input_ids"]
    out = await _post("embed", "/v1/embeddings", {"model": "embed", "input": [ids[: e["max_len"] - 1] + [S["eot"]]]})
    v = np.asarray(out["data"][0]["embedding"], dtype=np.float32)
    return v / np.linalg.norm(v)


def candidates(question: str, qv: np.ndarray, company: str | None):
    s, mode = CFG["search"], SERVE["api"]["search"]
    found, years, co, co_yr = entity_masks(question, S["company"], S["period"])
    if company:   # an explicit company scope replaces the detected one; the year filter still applies
        co = S["company"] == company
        m = co & np.isin(S["period"], years) if years else co
        co_yr = m if m.any() else co
    mask = co_yr if "year" in mode else co if "company" in mode else None
    dense = S["dense"] @ qv
    if mode.startswith("dense"):
        return top_k(dense, s["depth"], mask), found, years
    return rrf([top_k(S["bm"].scores(question), s["depth"], mask), top_k(dense, s["depth"], mask)], s["rrf_k"]), found, years


async def rerank(question: str, cands: list[int]) -> list[int]:
    r = CFG["rerank"]
    head = cands[: r["top_n"]]
    query = f"{RERANK_PREFIX}<Instruct>: {r['instruction']}\n<Query>: {question}\n"
    docs = [f"<Document>: {S['texts'][i]}{RERANK_SUFFIX}" for i in head]
    out = await _post("rerank", "/score", {"model": "rerank", "text_1": query, "text_2": docs})
    score = {d["index"]: d["score"] for d in out["data"]}
    return [head[k] for k in sorted(range(len(head)), key=lambda k: -score[k])] + cands[r["top_n"]:]


def gen_body(question: str, passages: list[tuple[str, str]], prompt: str, stream: bool) -> dict:
    g = CFG["gen"]
    return {"model": "gen", "messages": answer.build_messages(question, passages, prompt), "temperature": 0,
            "max_tokens": g["max_tokens"], "stream": stream,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "answer", "schema": llm.with_separators(answer.SCHEMA)}},
            "chat_template_kwargs": {"enable_thinking": g.get("thinking", False)}}


async def retrieve(req: Ask, t: dict):
    t0 = time.perf_counter()
    qv = await embed(req.question)
    t1 = time.perf_counter()
    cands, found, years = candidates(req.question, qv, req.company)
    t2 = time.perf_counter()
    top = (await rerank(req.question, cands))[: CFG["gen"]["contexts"]]
    t3 = time.perf_counter()
    t.update(embed_ms=1000 * (t1 - t0), search_ms=1000 * (t2 - t1), rerank_ms=1000 * (t3 - t2))
    passages = [(S["chunks"].header.iat[i], S["chunks"].text.iat[i]) for i in top]
    return top, passages, {"detected_companies": found, "years": years, "chunks": [S["keys"][i] for i in top]}


def result(req: Ask, text: str, top: list[int], passages, retrieval: dict, t: dict) -> dict:
    out = llm.parse_json(text)
    check = answer.check(out, len(passages), passages)
    ans = (out or {}).get("answer", text)
    chunks = S["chunks"]
    return {"question": req.question, "answer": ans, "answerable": (out or {}).get("answerable"),
            "abstained": out is None or out.get("answerable") is False or answer.says_missing(ans),
            "reasoning": (out or {}).get("reasoning"),
            "citations": [{"n": c, "company": chunks.company.iat[top[c - 1]], "filing": chunks.doc_name.iat[top[c - 1]],
                           "page": int(chunks.page.iat[top[c - 1]]) + 1, "excerpt": chunks.text.iat[top[c - 1]][:400]}
                          for c in check["cited"] if 1 <= c <= len(top)],
            "checks": {"valid_citations": check["valid_citations"], "ungrounded_numbers": check["ungrounded_numbers"]},
            "retrieval": retrieval, "timing_ms": {k: round(v, 1) for k, v in t.items()}}


@app.post("/ask")
async def ask(req: Ask) -> dict:
    t: dict = {}
    t0 = time.perf_counter()
    top, passages, retrieval = await retrieve(req, t)
    t1 = time.perf_counter()
    out = await _post("gen", "/v1/chat/completions", gen_body(req.question, passages, req.prompt, False))
    t["generate_ms"] = 1000 * (time.perf_counter() - t1)
    t["total_ms"] = 1000 * (time.perf_counter() - t0)
    res = result(req, out["choices"][0]["message"]["content"], top, passages, retrieval, t)
    res["tokens"] = out.get("usage")
    return res


@app.post("/ask/stream")
async def ask_stream(req: Ask) -> StreamingResponse:
    async def events():
        t: dict = {}
        t0 = time.perf_counter()
        top, passages, retrieval = await retrieve(req, t)
        yield f"event: retrieval\ndata: {json.dumps(retrieval)}\n\n"
        text, t1 = "", time.perf_counter()
        body = gen_body(req.question, passages, req.prompt, True)
        async with S["http"].stream("POST", URL["gen"] + "/v1/chat/completions", json=body) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
                if delta:
                    if not text:
                        t["first_token_ms"] = 1000 * (time.perf_counter() - t0)
                    text += delta
                    yield f"event: token\ndata: {json.dumps(delta)}\n\n"
        t["generate_ms"] = 1000 * (time.perf_counter() - t1)
        t["total_ms"] = 1000 * (time.perf_counter() - t0)
        yield f"event: result\ndata: {json.dumps(result(req, text, top, passages, retrieval, t))}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict:
    up = {}
    for name, url in URL.items():
        try:
            up[name] = (await S["http"].get(url + "/v1/models", timeout=5)).status_code == 200
        except httpx.HTTPError:
            up[name] = False
    return {"ok": all(up.values()), "servers": up, "chunks": len(S["texts"]), "search": SERVE["api"]["search"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=SERVE["api"]["port"], log_level="warning")
