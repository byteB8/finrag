"""Blind hand-grading of a random sample of answers, and the judge's agreement with those hand grades.

The sheet is one self-contained HTML file: question, reference answer, the model's answer and the excerpts
it was given, with the judge's verdict left out so it cannot anchor the grader. Labels autosave in the
browser and are exported as JSON (results/review/labels_<ctx>.json), which `run.py agree` compares with
the judge.
"""
from __future__ import annotations

import hashlib
import json
import random

from core.grade import VERDICTS


def sample_ids(ids: list[str], n: int, seed: int = 7) -> list[str]:
    return random.Random(seed).sample(list(ids), min(n, len(ids)))


def cohen_kappa(a: list[str], b: list[str], labels=VERDICTS) -> float:
    """Agreement between two graders beyond what their label frequencies would give by chance."""
    n = len(a)
    if n == 0:
        return float("nan")
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    return 1.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)


def agreement(human: dict[str, str], judge: dict[str, str]) -> dict:
    ids = [i for i in human if i in judge and human[i] in VERDICTS]
    h = [human[i] for i in ids]
    j = [judge[i] for i in ids]
    n = len(ids)
    return {
        "n": n,
        "agree": sum(x == y for x, y in zip(h, j)),
        "kappa": round(cohen_kappa(h, j), 3) if n else None,
        "confusion": {hv: {jv: sum(x == hv and y == jv for x, y in zip(h, j)) for jv in VERDICTS} for hv in VERDICTS},
        "correct_rate": {"human": sum(x == "correct" for x in h) / n if n else None,
                         "judge": sum(y == "correct" for y in j) / n if n else None},
        "disagreements": [{"id": i, "human": human[i], "judge": judge[i]} for i in ids if human[i] != judge[i]],
    }


def render_sheet(items: list[dict], ctx: str) -> str:
    """`items`: dicts with id, question, reference, justification, answer, reasoning, passages [[header, text]].
    Saved labels are keyed by a hash of the answers, so a regenerated sheet never shows stale labels."""
    version = hashlib.sha1(json.dumps([i["answer"] for i in items]).encode()).hexdigest()[:10]
    data = json.dumps({"ctx": ctx, "version": version, "items": items}).replace("</", "<\\/")
    return _PAGE.replace("__CTX__", ctx).replace("__N__", str(len(items))).replace("__DATA__", data)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hand grading</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1b; --muted:#5f5f5a; --line:#deddd8; --accent:#1f5f8b; --ok:#2e7d32; --bad:#b3261e; --ref:#8a6d00; }
@media (prefers-color-scheme: dark) { :root { --bg:#151514; --card:#1f1f1d; --ink:#ecebe6; --muted:#a3a29b; --line:#34342f; --accent:#7cb4dc; --ok:#7bc47f; --bad:#f08a80; --ref:#e0c25a; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0 auto 0 0; }
button { font:inherit; padding:6px 12px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--ink); cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
main { max-width:900px; margin:0 auto; padding:16px; }
.rules { color:var(--muted); font-size:14px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:14px 0; }
.card.done { border-left:4px solid var(--accent); }
.q { font-weight:600; }
.label { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin-top:10px; }
.ref { color:var(--ref); white-space:pre-wrap; }
.ans { white-space:pre-wrap; }
.small { color:var(--muted); font-size:13px; white-space:pre-wrap; }
details { margin-top:8px; } summary { cursor:pointer; color:var(--accent); }
pre { white-space:pre-wrap; font:12px/1.4 ui-monospace, monospace; background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:8px; max-height:300px; overflow:auto; }
.choices { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
.choices label { border:1px solid var(--line); border-radius:6px; padding:5px 10px; cursor:pointer; }
.choices input { margin-right:6px; }
textarea { width:100%; margin-top:8px; font:inherit; font-size:13px; background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:6px; }
</style></head>
<body>
<header><h1>Hand grading: __N__ answers (__CTX__ context)</h1><span id="progress"></span>
<button id="copy">Copy JSON</button><button id="export" class="primary">Export labels</button></header>
<main>
<p class="rules">Compare each answer with the analyst's reference. <b>correct</b>: agrees on every key fact and number
(rounding or equivalent units are fine; extra correct detail is fine). <b>incorrect</b>: a key number or fact differs,
contradicts the reference, or answers a different question. <b>refusal</b>: declines or says the information is missing.
The judge's verdicts are hidden. Labels save in this browser as you go; export when done.</p>
<div id="cards"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const KEY = "finrag-labels-" + DATA.ctx + "-" + DATA.version;
let state = { labels: {}, notes: {} };
try { state = Object.assign(state, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} progress(); };
const el = (tag, cls, text) => { const x = document.createElement(tag); if (cls) x.className = cls; if (text !== undefined) x.textContent = text; return x; };
function progress() {
  document.getElementById("progress").textContent = Object.keys(state.labels).length + " / " + DATA.items.length + " graded";
}
const cards = document.getElementById("cards");
DATA.items.forEach((it, i) => {
  const c = el("div", "card" + (state.labels[it.id] ? " done" : ""));
  c.append(el("div", "label", "#" + (i + 1) + "  " + it.id));
  c.append(el("div", "q", it.question));
  c.append(el("div", "label", "Reference answer"), el("div", "ref", String(it.reference)));
  if (it.justification) { const d = el("details"); d.append(el("summary", "", "Analyst's justification"), el("div", "small", it.justification)); c.append(d); }
  c.append(el("div", "label", "Answer to grade"), el("div", "ans", it.answer || "(no answer)"));
  if (it.reasoning) c.append(el("div", "small", "Model's reasoning: " + it.reasoning));
  const d = el("details"); d.append(el("summary", "", "Excerpts the model was given (" + it.passages.length + ")"));
  it.passages.forEach((p, k) => { d.append(el("div", "label", "[" + (k + 1) + "] " + p[0]), el("pre", "", p[1])); });
  c.append(d);
  const ch = el("div", "choices");
  ["correct", "incorrect", "refusal"].forEach(v => {
    const lab = el("label"); const r = el("input"); r.type = "radio"; r.name = "v" + i; r.value = v;
    r.checked = state.labels[it.id] === v;
    r.onchange = () => { state.labels[it.id] = v; c.classList.add("done"); save(); };
    lab.append(r, document.createTextNode(v)); ch.append(lab);
  });
  c.append(ch);
  const t = el("textarea"); t.rows = 1; t.placeholder = "Note (optional): e.g. reference looks wrong, partly right ...";
  t.value = state.notes[it.id] || ""; t.oninput = () => { state.notes[it.id] = t.value; save(); };
  c.append(t);
  cards.append(c);
});
progress();
const payload = () => JSON.stringify({ ctx: DATA.ctx, version: DATA.version, labels: state.labels, notes: state.notes }, null, 1);
document.getElementById("export").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([payload()], { type: "application/json" }));
  a.download = "labels_" + DATA.ctx + ".json"; a.click();
};
document.getElementById("copy").onclick = () => { navigator.clipboard.writeText(payload()).then(() => alert("Copied")); };
</script>
</body></html>
"""
