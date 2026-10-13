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

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "answerable": {"type": "boolean"},
    },
    "required": ["reasoning", "answer", "citations", "answerable"],
    "additionalProperties": False,
}


def build_messages(question: str, passages: list[tuple[str, str]]) -> list[dict]:
    """`passages` are (header, text) pairs, numbered from 1 in the prompt."""
    blocks = "\n\n".join(f"[{i}] {header}\n{text}" for i, (header, text) in enumerate(passages, 1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n\n{blocks}"}]


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
    cited = [c for c in out.get("citations", []) if isinstance(c, int)]
    valid = bool(cited) and all(1 <= c <= n_passages for c in cited)
    source = " ".join(passages[c - 1][1] for c in cited if 1 <= c <= n_passages)
    have = numbers(source)
    claimed = numbers(out.get("reasoning", "") + " " + out.get("answer", ""))
    return {"parsed": True, "valid_citations": valid, "cited": cited,
            "ungrounded_numbers": sorted(claimed - have)}
