"""Pipeline entry point: one stage per call, settings from a TOML config (configs/mvp.toml by default).

    python run.py prep                  parse the PDFs, chunk every page, check the evidence labels
    python run.py index                 BM25 index and dense embeddings of every chunk (one GPU process per shard)
    python run.py search                retrieval for each question under each configuration, scored
    python run.py gen --ctx retrieved   answers from the top retrieved chunks (--ctx gold: the labelled pages)
    python run.py grade --ctx retrieved verdicts from an LLM judge against the reference answers
    python run.py sheet --ctx retrieved blind hand-grading page for a random sample (--n, default 30)
    python run.py agree --ctx retrieved judge vs hand grades (from results/review/labels_<ctx>.json)
    python run.py report                results/RESULTS.md from the saved scores

Started detached by job.sh. Every process title is pinned to PROC_TITLE (default `python run.py`), so on
the shared server ps and nvidia-smi show a generic name instead of vLLM's EngineCore/worker titles.
"""
import os


def _pin_process_title() -> None:
    title = os.environ.get("PROC_TITLE", "python run.py")
    try:
        import setproctitle
    except ImportError:
        return
    original = setproctitle.setproctitle
    setproctitle.setproctitle = lambda *_args, **_kwargs: original(title)
    original(title)


# Top level on purpose: spawned child processes re-import this module (as __mp_main__) before vLLM
# retitles them, so the patch is in place in the engine core too.
_pin_process_title()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import tomllib  # noqa: E402

from core import env  # noqa: E402


def write_json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_name(cfg: dict, args) -> str:
    """Output name of a gen/grade run: the context plus the config's optional tag (e.g. retrieved_v2)."""
    return args.ctx + cfg.get("tag", "")


def gold_pages(q: dict) -> list[tuple[str, int]]:
    return sorted({(e.get("doc_name") or e.get("evidence_doc_name"), int(e["evidence_page_num"]))
                   for e in q["evidence"]})


# ---------------------------------------------------------------- prep

def stage_prep(cfg: dict, args) -> None:
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd
    from transformers import AutoTokenizer

    from core import data

    t0 = time.time()
    info = data.load_docinfo(env.FB)
    pdfs = sorted((env.FB / "pdfs").glob("*.pdf"))
    with ProcessPoolExecutor(max_workers=env.MAX_CPUS) as ex:
        parsed = list(ex.map(data.parse_pdf, map(str, pdfs), chunksize=2))
    rows = []
    for pdf, pages in zip(pdfs, parsed):
        meta = data.doc_meta(pdf.stem, info)
        rows += [{"doc_name": pdf.stem, "page": p, **meta, "text": text} for p, text in enumerate(pages)]
    pages_df = pd.DataFrame(rows)
    pages_df.to_parquet(env.DATA / "pages.parquet", index=False)
    print(f"parsed {len(pdfs)} PDFs, {len(pages_df)} pages in {time.time() - t0:.0f}s", flush=True)

    # Chunk sizes in tokens of the embedding model (digits tokenize one by one, so tables run long in words).
    tok = AutoTokenizer.from_pretrained(cfg["embed"]["model"])

    def sizes_of(lines):
        return [len(ids) for ids in tok(lines, add_special_tokens=False)["input_ids"]] if lines else []

    c = cfg["chunk"]
    chunks = []
    for r in pages_df.itertuples(index=False):
        meta = {"company": r.company, "doc_type": r.doc_type, "period": r.period}
        for part, text in enumerate(data.chunk_page(r.text, c["max_tokens"], c["overlap_tokens"], sizes_of)):
            chunks.append({"doc_name": r.doc_name, "page": r.page, "part": part, **meta,
                           "header": data.chunk_header(meta, r.page), "text": text})
    chunks_df = pd.DataFrame(chunks)
    chunks_df.to_parquet(env.DATA / "chunks.parquet", index=False)
    print(f"{len(chunks_df)} chunks in {time.time() - t0:.0f}s", flush=True)

    # Does each labelled evidence passage occur on its labelled page of these PDFs?
    text_of = {(r.doc_name, r.page): r.text for r in pages_df.itertuples(index=False)}
    items = []
    for q in data.load_questions(env.FB):
        for e in q["evidence"]:
            doc, page = e.get("doc_name") or e.get("evidence_doc_name"), int(e["evidence_page_num"])
            here = data.evidence_overlap(e["evidence_text"], text_of.get((doc, page), ""))
            near = max((data.evidence_overlap(e["evidence_text"], text_of.get((doc, page + o), "")), o)
                       for o in (-2, -1, 1, 2))
            items.append({"id": q["financebench_id"], "doc": doc, "page": page, "doc_found": (doc, 0) in text_of,
                          "overlap": round(here, 3), "best_neighbour": {"offset": near[1], "overlap": round(near[0], 3)}})
    n = len(items)
    ok = sum(i["overlap"] >= 0.8 for i in items)
    shifted = sum(i["overlap"] < 0.8 and i["best_neighbour"]["overlap"] >= 0.8 for i in items)
    write_json(env.RESULTS / "data_check.json", {
        "pdfs": len(pdfs), "pages": len(pages_df), "chunks": len(chunks_df),
        "chunk_tokens": {"max": c["max_tokens"], "overlap": c["overlap_tokens"]},
        "evidence_items": n, "on_labelled_page": ok, "on_neighbouring_page": shifted,
        "missing_doc": sum(not i["doc_found"] for i in items), "items": items})
    print(f"evidence on its labelled page: {ok}/{n}; on a neighbouring page instead: {shifted}", flush=True)


# ---------------------------------------------------------------- index

def chunk_texts(chunks) -> list[str]:
    return (chunks["header"] + "\n" + chunks["text"]).tolist()


def stage_index(cfg: dict, args) -> None:
    """BM25 on the CPU, then dense embeddings sharded over every GPU with room (one `run.py embed` process
    per GPU), merged into data/index/dense.npy. Shard 0 also embeds the questions (data/index/queries.npy)."""
    import os
    import subprocess
    import sys

    import numpy as np
    import pandas as pd

    from core.search import BM25

    chunks = pd.read_parquet(env.DATA / "chunks.parquet")
    texts = chunk_texts(chunks)
    env.INDEX.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    BM25.build(texts).save(env.INDEX / "bm25")
    print(f"bm25: {len(texts)} chunks in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    need = cfg["embed"]["need_mib"]
    want = os.environ.get("GPUS")   # e.g. "0,1"; default: every GPU with `need` MiB free
    ids = [int(x) for x in want.split(",")] if want else [g["index"] for g in env.gpus() if g["free"] - 768 >= need]
    if not ids:
        raise RuntimeError("no GPU has room for the embedding model")
    procs = [subprocess.Popen([sys.executable, "run.py", "embed", "--cfg", args.cfg, "--shard", str(k),
                               "--shards", str(len(ids))], env={**os.environ, "GPU": str(gpu)})
             for k, gpu in enumerate(ids)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"embedding shards failed: exit codes {codes}")
    parts = [env.INDEX / f"dense_{k}.npy" for k in range(len(ids))]
    vecs = np.concatenate([np.load(f) for f in parts])
    assert len(vecs) == len(texts), (len(vecs), len(texts))
    np.save(env.INDEX / "dense.npy", vecs)
    for f in parts:
        f.unlink()
    print(f"dense: {vecs.shape} on GPUs {ids} in {time.time() - t0:.0f}s", flush=True)


def stage_embed(cfg: dict, args) -> None:
    """One shard of the chunk embeddings on one GPU (started by `index`); contiguous slices keep the order."""
    import numpy as np
    import pandas as pd

    from core import data
    from core.search import Embedder

    e = cfg["embed"]
    gpu, util = env.pick_gpu(need_mib=e["need_mib"])
    util = min(util, e["gpu_util"])
    env.use_gpu(gpu)
    texts = chunk_texts(pd.read_parquet(env.DATA / "chunks.parquet"))
    size = -(-len(texts) // args.shards)
    part = texts[args.shard * size:(args.shard + 1) * size]
    t0 = time.time()
    log = (lambda m: print(f"[shard {args.shard}] {m}", flush=True))
    embed = Embedder(e["model"], util, max_tokens=e["max_len"])
    vecs = embed(part, log=log)
    np.save(env.INDEX / f"dense_{args.shard}.npy", vecs)
    log(f"{len(part)} chunks on GPU {gpu} in {time.time() - t0:.0f}s")
    if args.shard == 0:
        qs = data.load_questions(env.FB)
        np.save(env.INDEX / "queries.npy", embed([x["question"] for x in qs], instruction=e["query_instruction"]))


# ---------------------------------------------------------------- search

def stage_search(cfg: dict, args) -> None:
    """Candidates per question and configuration on the CPU; every reranker pair scored in one vLLM run."""
    import numpy as np
    import pandas as pd

    from core import data, metrics
    from core.search import BM25, rerank_scores, rrf, top_k

    s, rr_cfg = cfg["search"], cfg["rerank"]
    chunks = pd.read_parquet(env.DATA / "chunks.parquet")
    texts = chunk_texts(chunks)
    keys = [f"{d}:{p}:{k}" for d, p, k in zip(chunks.doc_name, chunks.page, chunks.part)]
    page_of = list(zip(chunks.doc_name, chunks.page.astype(int)))
    company = chunks.company.to_numpy()
    doc = chunks.doc_name.to_numpy()
    period = chunks.period.astype(str).to_numpy()
    dense = np.load(env.INDEX / "dense.npy").astype(np.float32)
    bm = BM25.load(env.INDEX / "bm25")
    qs = data.load_questions(env.FB)[: args.limit or None]
    qvecs = np.load(env.INDEX / "queries.npy").astype(np.float32)[: len(qs)]
    K, depth, top_n = max(s["k"]), s["depth"], rr_cfg["top_n"]

    t0 = time.time()
    cands, extra = [], []
    for q, qv in zip(qs, qvecs):
        bm_s, de_s = bm.scores(q["question"]), dense @ qv
        found, years = data.detect_companies(q["question"]), data.detect_years(q["question"])
        co = np.isin(company, found) if found else None
        co_yr = co
        if co is not None and years:
            m = co & np.isin(period, years)
            co_yr = m if m.any() else co

        def hybrid(mask):
            return rrf([top_k(bm_s, depth, mask), top_k(de_s, depth, mask)], s["rrf_k"])

        cands.append({
            "bm25": top_k(bm_s, K), "dense": top_k(de_s, K), "hybrid": hybrid(None), "hybrid+company": hybrid(co),
            "hybrid+company+rerank": hybrid(co), "hybrid+company+year+rerank": hybrid(co_yr),
            # scoped settings use the question's labels, as if the user had picked the company or filing
            "scoped-company: hybrid+rerank": hybrid(company == q["company"]),
            "scoped-filing: hybrid+rerank": hybrid(doc == q["doc_name"])})
        extra.append({"detected": found, "years": years})
    print(f"candidates for {len(qs)} questions in {time.time() - t0:.0f}s", flush=True)

    reranked = [n for n in cands[0] if n.endswith("rerank")]
    pairs = sorted({(qi, i) for qi, c in enumerate(cands) for n in reranked for i in c[n][:top_n]})
    gpu, util = env.pick_gpu(need_mib=rr_cfg["need_mib"])
    util = min(util, rr_cfg["gpu_util"])
    env.use_gpu(gpu)
    t1 = time.time()
    scores = rerank_scores(rr_cfg["model"], [(qs[qi]["question"], texts[i]) for qi, i in pairs],
                           rr_cfg["instruction"], util, max_tokens=rr_cfg["max_len"])
    score = dict(zip(pairs, scores))
    print(f"reranked {len(pairs)} pairs on GPU {gpu} in {time.time() - t1:.0f}s", flush=True)

    rows = []
    for qi, (q, c, x) in enumerate(zip(qs, cands, extra)):
        runs = {n: (sorted(r[:top_n], key=lambda i: -score[(qi, i)]) + r[top_n:]) if n in reranked else r
                for n, r in c.items()}
        gold = gold_pages(q)
        rows.append({"id": q["financebench_id"], "question": q["question"], "company": q["company"],
                     "doc_name": q["doc_name"], **x, "gold": gold,
                     "page_rank": {n: metrics.first_hit([page_of[i] for i in r[:K]], set(map(tuple, gold)))
                                   for n, r in runs.items()},
                     "doc_rank": {n: metrics.first_hit([doc[i] for i in r[:K]], {g[0] for g in gold})
                                  for n, r in runs.items()},
                     "top": {n: [keys[i] for i in r[:K]] for n, r in runs.items()}})
    write_jsonl(env.RAW / "search.jsonl", rows)

    summary = {"questions": len(rows), "rerank_pairs": len(pairs), "seconds": round(time.time() - t0, 1), "configs": {}}
    for name in rows[0]["top"]:
        pr = [r["page_rank"][name] for r in rows]
        dr = [r["doc_rank"][name] for r in rows]
        summary["configs"][name] = {
            **{f"page@{k}": metrics.bootstrap_ci(metrics.hits_at(pr, k)) for k in s["k"]},
            **{f"doc@{k}": metrics.bootstrap_ci(metrics.hits_at(dr, k)) for k in s["k"]},
            f"mrr@{K}": metrics.bootstrap_ci(metrics.reciprocal_ranks(pr, K))}
    det = [r["detected"] for r in rows]
    gold_co = [r["company"] for r in rows]
    summary["company_detection"] = {
        "exact": sum(d == [g] for d, g in zip(det, gold_co)),
        "includes_gold": sum(g in d for d, g in zip(det, gold_co)),
        "none": sum(not d for d in det),
        "wrong_only": sum(bool(d) and g not in d for d, g in zip(det, gold_co))}
    write_json(env.RESULTS / "search.json", summary)
    for name, m in summary["configs"].items():
        print(f"{name:34s} page@5 {m['page@5'][0]:.3f}  page@10 {m['page@10'][0]:.3f}  doc@5 {m['doc@5'][0]:.3f}",
              flush=True)


# ---------------------------------------------------------------- gen and grade

def contexts(cfg: dict, ctx: str, qs: list[dict]) -> list[list[tuple[str, str]]]:
    """(header, text) passages per question: the top retrieved chunks, or the labelled evidence pages."""
    import pandas as pd

    from core import data
    if ctx == "gold":
        pages = pd.read_parquet(env.DATA / "pages.parquet")
        text_of = {(r.doc_name, r.page): (r, r.text) for r in pages.itertuples(index=False)}
        out = []
        for q in qs:
            ps = []
            for d, p in gold_pages(q):
                r, text = text_of[(d, p)]
                ps.append((data.chunk_header({"company": r.company, "doc_type": r.doc_type, "period": r.period}, p), text))
            out.append(ps)
        return out
    chunks = pd.read_parquet(env.DATA / "chunks.parquet")
    by_key = {f"{d}:{p}:{k}": (h, t) for d, p, k, h, t in
              zip(chunks.doc_name, chunks.page, chunks.part, chunks.header, chunks.text)}
    found = {r["id"]: r for r in read_jsonl(env.RAW / "search.jsonl")}
    g = cfg["gen"]
    return [[by_key[k] for k in found[q["financebench_id"]]["top"][g["search"]][: g["contexts"]]] for q in qs]


def stage_gen(cfg: dict, args) -> None:
    from core import answer, data, llm

    g = cfg["gen"]
    qs = data.load_questions(env.FB)[: args.limit or None]
    if args.ids:
        qs = [q for q in qs if q["financebench_id"] in args.ids.split(",")]
    passages = contexts(cfg, args.ctx, qs)
    gpu, util = env.pick_gpu(need_mib=g["need_mib"])
    env.use_gpu(gpu)
    model = llm.make_llm(g["model"], g["max_model_len"], util, max_num_seqs=g["max_num_seqs"],
                         limit_mm_per_prompt={"image": 0, "video": 0})
    params = llm.json_params(answer.SCHEMA, g["max_tokens"])
    t0 = time.time()
    prompt = g.get("prompt", "v1")
    outs = llm.chat(model, [answer.build_messages(q["question"], p, prompt) for q, p in zip(qs, passages)], params,
                    thinking=g.get("thinking", False))
    secs = time.time() - t0
    rows = []
    for q, p, o in zip(qs, passages, outs):
        parsed = llm.parse_json(o["text"])
        rows.append({"id": q["financebench_id"], "question": q["question"], "reference": q["answer"],
                     "answer": (parsed or {}).get("answer", o["text"]), "answerable": (parsed or {}).get("answerable"),
                     "reasoning": (parsed or {}).get("reasoning"), "check": answer.check(parsed, len(p), p),
                     "passages": [h for h, _ in p], **{k: o[k] for k in ("finish", "prompt_tokens", "output_tokens")}})
    write_jsonl(env.RAW / f"gen_{run_name(cfg, args)}.jsonl", rows)
    print(f"{len(rows)} answers on GPU {gpu} in {secs:.0f}s "
          f"({sum(r['output_tokens'] for r in rows) / secs:.0f} output tok/s)", flush=True)


def stage_grade(cfg: dict, args) -> None:
    from core import data, grade, llm, metrics

    j = cfg["grade"]
    rows = read_jsonl(env.RAW / f"gen_{run_name(cfg, args)}.jsonl")
    refs = {q["financebench_id"]: q for q in data.load_questions(env.FB)}
    gpu, util = env.pick_gpu(need_mib=j["need_mib"])
    env.use_gpu(gpu)
    model = llm.make_llm(j["model"], j["max_model_len"], util, max_num_seqs=j["max_num_seqs"])
    convs = [grade.build_messages(r["question"], r["reference"], refs[r["id"]].get("justification"), r["answer"])
             for r in rows]
    outs = llm.chat(model, convs, llm.json_params(grade.SCHEMA, j["max_tokens"]))
    graded = []
    for r, o in zip(rows, outs):
        parsed = llm.parse_json(o["text"])
        graded.append({"id": r["id"], "verdict": grade.verdict_of(parsed), "reason": (parsed or {}).get("reason"),
                       "judge_parsed": parsed is not None})
    write_jsonl(env.RAW / f"grade_{run_name(cfg, args)}.jsonl", graded)
    summary = {"questions": len(graded), "judge": j["model"], "judge_unparsed": sum(not g["judge_parsed"] for g in graded)}
    for v in grade.VERDICTS:
        summary[v] = metrics.bootstrap_ci([g["verdict"] == v for g in graded])
    checks = [r["check"] for r in rows]
    summary["answers_unparsed"] = sum(not c["parsed"] for c in checks)
    summary["valid_citations"] = sum(c["valid_citations"] for c in checks)
    summary["with_ungrounded_numbers"] = sum(bool(c["ungrounded_numbers"]) for c in checks)
    write_json(env.RESULTS / f"grade_{run_name(cfg, args)}.json", summary)
    print(json.dumps({k: v for k, v in summary.items()}, indent=1), flush=True)


# ---------------------------------------------------------------- hand grading

def stage_sheet(cfg: dict, args) -> None:
    from core import data, review

    rows = {r["id"]: r for r in read_jsonl(env.RAW / f"gen_{run_name(cfg, args)}.jsonl")}
    qs = {q["financebench_id"]: q for q in data.load_questions(env.FB)}
    ids = review.sample_ids(sorted(rows), args.n)
    passages = contexts(cfg, args.ctx, [qs[i] for i in ids])
    items = [{"id": i, "question": rows[i]["question"], "reference": rows[i]["reference"],
              "justification": qs[i].get("justification"), "answer": rows[i]["answer"],
              "reasoning": rows[i].get("reasoning"), "passages": [list(p) for p in ps]}
             for i, ps in zip(ids, passages)]
    out = env.RESULTS / "review" / f"sheet_{run_name(cfg, args)}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(review.render_sheet(items, run_name(cfg, args)))
    print(f"wrote {out.relative_to(env.BASE)} ({len(items)} answers)")


def stage_agree(cfg: dict, args) -> None:
    from core import review

    labels = json.loads((env.RESULTS / "review" / f"labels_{run_name(cfg, args)}.json").read_text())["labels"]
    judge = {g["id"]: g["verdict"] for g in read_jsonl(env.RAW / f"grade_{run_name(cfg, args)}.jsonl")}
    res = review.agreement(labels, judge)
    write_json(env.RESULTS / f"agreement_{run_name(cfg, args)}.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements"}, indent=1))


# ---------------------------------------------------------------- report

def stage_report(cfg: dict, args) -> None:
    from core import report
    (env.RESULTS / "RESULTS.md").write_text(report.render(env.RESULTS, env.FB / "results"))
    print("wrote results/RESULTS.md")


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(path) -> dict:
    """A config may name a `base` config (relative to the project root) and override only what differs."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if "base" in cfg:
        cfg = merge(load_cfg(env.BASE / cfg.pop("base")), cfg)
    return cfg


STAGES = {"prep": stage_prep, "index": stage_index, "embed": stage_embed, "search": stage_search, "gen": stage_gen,
          "grade": stage_grade, "sheet": stage_sheet, "agree": stage_agree, "report": stage_report}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--cfg", default="configs/mvp.toml")
    ap.add_argument("--ctx", default="retrieved", choices=["retrieved", "gold"])
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke runs)")
    ap.add_argument("--n", type=int, default=30, help="sample size for the hand-grading sheet")
    ap.add_argument("--ids", default="", help="gen: only these comma-separated question ids (debugging)")
    ap.add_argument("--shard", type=int, default=0, help="embed: which slice of the chunks")
    ap.add_argument("--shards", type=int, default=1, help="embed: how many slices")
    args = ap.parse_args()
    env.set_defaults()
    cfg = load_cfg(env.BASE / args.cfg)
    t0 = time.time()
    STAGES[args.stage](cfg, args)
    print(f"[{args.stage}] done in {time.time() - t0:.0f}s", flush=True)


def _reap_children(timeout: float = 15) -> None:
    """Terminate (then kill) every process this one started, e.g. vLLM's engine core."""
    try:
        import psutil
    except ImportError:
        return
    kids = psutil.Process().children(recursive=True)
    for k in kids:
        try:
            k.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(kids, timeout=timeout)
    for k in alive:
        try:
            k.kill()
        except psutil.Error:
            pass


if __name__ == "__main__":
    import sys
    import traceback
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:
        traceback.print_exc()
        code = 1
    # Outputs are written. vLLM's own teardown can hang a finished process, so stop its engine process
    # directly and exit without waiting on it.
    sys.stdout.flush()
    sys.stderr.flush()
    _reap_children()
    os._exit(code)
