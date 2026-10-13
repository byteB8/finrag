"""results/RESULTS.md from the JSON summaries the stages write."""
from __future__ import annotations

import json
from pathlib import Path


def _pct(ci) -> str:
    m, lo, hi = ci
    return f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def render(results: Path) -> str:
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
    return "\n".join(out) + "\n"
