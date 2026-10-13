"""results/RESULTS.md from the JSON summaries the stages write."""
from __future__ import annotations

import json
from pathlib import Path

from core import metrics
from core.answer import says_missing


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


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _verdicts(results: Path, name: str) -> dict[str, str]:
    return {g["id"]: g["verdict"] for g in _jsonl(results / "raw" / f"grade_{name}.jsonl")}


def prompt_ab(results: Path) -> list[str]:
    """Paired comparison of prompt v2 against v1 on the same questions (share correct, bootstrap over questions)."""
    rows = []
    for ctx in ("gold", "retrieved"):
        a, b = _verdicts(results, ctx), _verdicts(results, f"{ctx}_v2")
        ids = sorted(set(a) & set(b))
        if not ids:
            continue
        ca = [a[i] == "correct" for i in ids]
        cb = [b[i] == "correct" for i in ids]
        d = metrics.paired_diff(cb, ca)
        rows.append(f"| {run_label(ctx).split(',')[0]} | {100 * sum(ca) / len(ids):.1f} | {100 * sum(cb) / len(ids):.1f} | "
                    f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] | {d['p']:.2f} |")
    return rows


def run_label(name: str) -> str:
    """retrieved -> 'retrieved chunks'; gold_v2 -> 'labelled evidence pages (oracle), prompt v2'."""
    ctx, _, tag = name.partition("_")
    label = {"retrieved": "retrieved chunks", "gold": "labelled evidence pages (oracle)"}.get(ctx, ctx)
    if tag == "prefix":
        return f"{label}, prompt v1, before the decoding-loop fixes"
    return f"{label}, prompt {tag}" if tag else f"{label}, prompt v1"


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
    for path in sorted(results.glob("grade_*.json")):
        g = json.loads(path.read_text())
        name = path.stem[len("grade_"):]
        gen = _jsonl(results / "raw" / f"gen_{name}.jsonl")
        missing = f"{sum(says_missing(r['answer']) for r in gen)}" if gen else "-"
        rows.append(f"| {run_label(name)} | {_pct(g['correct'])} | {_pct(g['incorrect'])} | {_pct(g['refusal'])} | "
                    f"{missing} | {g['valid_citations']}/{g['questions']} | {g['with_ungrounded_numbers']} |")
    if rows:
        out += ["## Answers", "", "Verdicts from an LLM judge (Qwen3-14B-AWQ) against the analysts' reference answers. "
                "The judge files most answers that say the information is missing under *incorrect*, so the "
                "*says info missing* column counts them from the answer text.", "",
                "| run | correct | incorrect | refusal | says info missing | valid citations | with numbers not in cited text |",
                "|---|---|---|---|---|---|---|", *rows, ""]
    ab = prompt_ab(results)
    if ab:
        out += ["## Prompt A/B (v2 - v1, same questions)", "",
                "Prompt v2 was written after reading v1's errors on these same questions (see the notebook), so it is "
                "tuned on the test set.", "",
                "| context | v1 correct % | v2 correct % | difference [95% CI] | p |", "|---|---|---|---|---|", *ab, ""]
    for path in sorted(results.glob("agreement_*.json")):
        if path.exists():
            a = json.loads(path.read_text())
            c = a["correct_rate"]
            b = a["binary"]
            out += ["## Judge check", "",
                    f"On {a['n']} randomly sampled answers ({run_label(path.stem[len('agreement_'):])}), graded blind by hand, "
                    f"the judge agreed on correct vs not correct for {b['agree']} (Cohen's kappa {b['kappa']}); on all "
                    f"three verdicts for {a['agree']} (kappa {a['kappa']}; the hand grades marked refusals as incorrect). "
                    f"Share graded correct on that sample: by hand {100 * c['human']:.1f}%, by the judge "
                    f"{100 * c['judge']:.1f}%.", ""]
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
