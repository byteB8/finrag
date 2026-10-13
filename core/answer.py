"""Cited answers: the prompt, the output schema, and checks on what the model returned."""
from __future__ import annotations

import re

SYSTEM = """You answer questions about public companies using excerpts from their SEC filings and earnings reports.
Rules:
- Use only the numbered excerpts, never outside knowledge.
- In "reasoning", note the figures you use and any calculation, in at most three sentences.
- In "answer", give the final answer in one or two sentences, with units (for example USD millions) and the fiscal period.
- In "citations", list the numbers of the excerpts that contain the figures or statements you used.
- If the excerpts do not contain what the question needs, set "answerable" to false and say in "answer" what is missing."""

# v2: written after reading v1's errors on the labelled-evidence run (refusals on judgement questions, fiscal
# years that do not end in December, line items under other names). Results with v2 are therefore tuned on
# the same 150 questions and are reported as such.
SYSTEM_V2 = """You are a financial analyst answering questions about public companies from excerpts of their SEC filings and earnings reports.
Rules:
- Use only the numbered excerpts for company figures. Standard financial-analysis definitions and formulas may be applied to them.
- Fiscal years can end in any month: a balance sheet dated at a company's fiscal year end (for example June 30, 2020 for a June year end) is that fiscal year's year-end balance sheet.
- Line items go by several names: cost of sales = cost of goods sold; property, plant and equipment, net = net PP&E; free cash flow = cash from operations minus capital expenditures; accounts receivable, net = net AR.
- For judgement questions (for example "is the business capital-intensive?" or "is the margin improving?"), compute the relevant ratios from the figures, then give a clear yes or no with the numbers behind it.
- In "reasoning", note the figures you use and any calculation, in at most three sentences.
- In "answer", give the final answer in one or two sentences, with units (for example USD millions) and the fiscal period.
- In "citations", list the numbers of the excerpts that contain the figures or statements you used.
- Set "answerable" to false only when a figure the answer needs is absent from the excerpts, and then say in "answer" what is missing."""

PROMPTS = {"v1": SYSTEM, "v2": SYSTEM_V2}
MAX_PASSAGES = 8   # the most excerpts any prompt carries (8 retrieved chunks; at most 3 evidence pages)

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        # bounded: unbounded, greedy decoding sometimes looped ("1, 2, 1, 2, ..." or "164, 165, ...")
        "citations": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": MAX_PASSAGES},
                      "maxItems": MAX_PASSAGES},
        "answerable": {"type": "boolean"},
    },
    "required": ["reasoning", "answer", "citations", "answerable"],
    "additionalProperties": False,
}


def build_messages(question: str, passages: list[tuple[str, str]], prompt: str = "v1") -> list[dict]:
    """`passages` are (header, text) pairs, numbered from 1 in the prompt."""
    blocks = "\n\n".join(f"[{i}] {header}\n{text}" for i, (header, text) in enumerate(passages, 1))
    return [{"role": "system", "content": PROMPTS[prompt]},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n\n{blocks}"}]


_MISSING = re.compile(r"(not (contain|provide|include|available|specif|list|detail|explicitly)|cannot be "
                      r"(calculated|determined)|insufficient|missing|unable)", re.I)


def says_missing(text: str) -> bool:
    """The answer text itself says the excerpts lack what is needed (the `answerable` flag can disagree)."""
    return bool(_MISSING.search(text or ""))


_NUM = re.compile(r"\d[\d,]*\.?\d*")


def numbers(text: str) -> set[str]:
    """Numbers as written, without thousands separators or trailing zeros after a decimal point."""
    out = set()
    for m in _NUM.findall(text):
        n = m.replace(",", "").rstrip(".")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        if n:
            out.add(n)
    return out


def check(out: dict | None, n_passages: int, passages: list[tuple[str, str]]) -> dict:
    """Checks on one answer: parsed at all, citations in range, and which numbers in the reasoning and
    answer appear in no cited passage (computed values such as ratios legitimately appear nowhere)."""
    if out is None:
        return {"parsed": False, "valid_citations": False, "cited": [], "ungrounded_numbers": []}
    cited = list(dict.fromkeys(c for c in out.get("citations", []) if isinstance(c, int)))   # de-duplicated
    valid = bool(cited) and all(1 <= c <= n_passages for c in cited)
    source = " ".join(passages[c - 1][1] for c in cited if 1 <= c <= n_passages)
    have = numbers(source)
    claimed = numbers(out.get("reasoning", "") + " " + out.get("answer", ""))
    return {"parsed": True, "valid_citations": valid, "cited": cited,
            "ungrounded_numbers": sorted(claimed - have)}
