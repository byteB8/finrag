"""Abstention: how often the system declines, on questions it can and cannot answer.

Two rules for "declined": the answer's own `answerable` flag, and the answer text saying the information
is missing. They can disagree (see the notebook), so both are reported.
"""
from __future__ import annotations

from core.answer import says_missing

RULES = {"flag": lambda r: r.get("answerable") is False,
         "text": lambda r: says_missing(r.get("answer") or "")}


def summarize(answerable: list[dict], verdicts: dict[str, str], unanswerable: list[dict]) -> dict:
    """`answerable`: gen rows for questions with a reference answer; `verdicts`: the judge's verdict per id;
    `unanswerable`: gen rows for questions the corpus cannot answer."""
    out = {}
    for name, declined in RULES.items():
        a = [declined(r) for r in answerable]
        answered = [r for r, d in zip(answerable, a) if not d]
        u = [declined(r) for r in unanswerable]
        out[name] = {
            "answerable": {
                "n": len(answerable), "declined": sum(a),
                "coverage": 1 - sum(a) / len(a) if a else None,
                "correct": sum(verdicts.get(r["id"]) == "correct" for r in answerable) / len(a) if a else None,
                "selective_accuracy": (sum(verdicts.get(r["id"]) == "correct" for r in answered) / len(answered)
                                       if answered else None)},
            "unanswerable": {"n": len(unanswerable), "declined": sum(u),
                             "declined_rate": sum(u) / len(u) if u else None}}
    return out
