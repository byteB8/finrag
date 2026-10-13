"""results/RESULTS.md from the JSON summaries the stages write."""
from __future__ import annotations

import json
from pathlib import Path


def _pct(ci) -> str:
    m, lo, hi = ci
    return f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


PAPER_SETTINGS = {"sharedStore": "one index over all filings", "singleStore": "the question's filing only",
                  "oracle": "labelled evidence pages given"}


def paper_baselines(fb_results: Path) -> list[dict]:
    """Correct / incorrect / refusal shares of the FinanceBench paper's human-graded runs, per model and setting."""
    out = []
    for f in sorted(fb_results.glob("*.jsonl")):
        model, _, mode = f.stem.partition("_")
        if mode not in PAPER_SETTINGS:
            continue
        labels = [json.loads(line)["label"] for line in f.read_text().splitlines() if line.strip()]
        n = len(labels)
        out.append({"model": model, "setting": mode, "n": n,
                    "correct": labels.count("Correct Answer") / n, "incorrect": labels.count("Incorrect Answer") / n,
                    "refusal": labels.count("Refusal") / n})
    return out


def render(results: Path, fb_results: Path | None = None) -> str:
    out = ["# Results", "", "FinanceBench open-source sample: 150 questions over 368 filings. Brackets are 95% "
           "bootstrap intervals over questions.", ""]
    path = results / "data_check.json"
    if path.exists():
        d = json.loads(path.read_text())
        out += ["## Data", "",
                f"- {d['pdfs']} PDFs, {d['pages']:,} pages, {d['chunks']:,} chunks of at most "
                f"{d['chunk_tokens']['max']} tokens ({d['chunk_tokens']['overlap']} overlap), never crossing a page.",
                f"- Labelled evidence found on its labelled page: {d['on_labelled_page']} of {d['evidence_items']} "
                f"items; on a neighbouring page instead: {d['on_neighbouring_page']}; document missing: "
                f"{d['missing_doc']}.", ""]
    path = results / "search.json"
    if path.exists():
        d = json.loads(path.read_text())
        out += ["## Retrieval", "",
                "A hit means a retrieved chunk lies on a labelled evidence page (page) or in the labelled filing (doc).", "",
                "| configuration | page hit@1 | page hit@5 | page hit@10 | doc hit@5 | MRR |",
                "|---|---|---|---|---|---|"]
        for name, m in d["configs"].items():
            mrr = next(v for k, v in m.items() if k.startswith("mrr@"))
            out.append(f"| {name} | {_pct(m['page@1'])} | {_pct(m['page@5'])} | {_pct(m['page@10'])} | "
                       f"{_pct(m['doc@5'])} | {mrr[0]:.3f} |")
        c = d["company_detection"]
        out += ["", f"Company detection from the question text: exactly right {c['exact']}, includes the right "
                f"company {c['includes_gold']}, none found {c['none']}, wrong {c['wrong_only']} "
                f"(of {d['questions']}).", ""]
    rows = []
    for ctx, label in (("retrieved", "retrieved chunks"), ("gold", "labelled evidence pages (oracle)")):
        path = results / f"grade_{ctx}.json"
        if path.exists():
            g = json.loads(path.read_text())
            rows.append(f"| {label} | {_pct(g['correct'])} | {_pct(g['incorrect'])} | {_pct(g['refusal'])} | "
                        f"{g['valid_citations']}/{g['questions']} | {g['with_ungrounded_numbers']} |")
    if rows:
        out += ["## Answers", "", "Verdicts from an LLM judge against the analysts' reference answers.", "",
                "| context | correct | incorrect | refusal | valid citations | with numbers not in cited text |",
                "|---|---|---|---|---|---|", *rows, ""]
    for ctx in ("retrieved", "gold"):
        path = results / f"agreement_{ctx}.json"
        if path.exists():
            a = json.loads(path.read_text())
            c = a["correct_rate"]
            out += ["## Judge check", "",
                    f"On {a['n']} randomly sampled answers ({ctx} context), graded blind by hand, the judge agreed on "
                    f"{a['agree']} (Cohen's kappa {a['kappa']}). Share graded correct on that sample: by hand "
                    f"{100 * c['human']:.1f}%, by the judge {100 * c['judge']:.1f}%.", ""]
    if fb_results is not None and fb_results.exists():
        base = paper_baselines(fb_results)
        if base:
            out += ["## FinanceBench paper baselines", "",
                    "Human-graded runs released with the benchmark (same 150 questions).", "",
                    "| model | setting | correct | incorrect | refusal |", "|---|---|---|---|---|"]
            out += [f"| {b['model']} | {PAPER_SETTINGS[b['setting']]} | {100 * b['correct']:.1f} | "
                    f"{100 * b['incorrect']:.1f} | {100 * b['refusal']:.1f} |" for b in base]
            out.append("")
    return "\n".join(out) + "\n"
