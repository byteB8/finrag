"""LLM-judge grading against FinanceBench's reference answers (verdicts as in the FinanceBench paper)."""
from __future__ import annotations

VERDICTS = ("correct", "incorrect", "refusal")

SYSTEM = """You grade answers to questions about public companies against a reference answer written by a financial analyst.
Verdicts:
- "correct": the answer agrees with the reference on every key fact and number. Numbers may differ by rounding (about 1%) or by units that mean the same amount (1.5 billion = 1,500 million). Extra correct detail is fine. If the reference is a judgement (for example "yes, because ..."), the answer must reach the same judgement for a compatible reason.
- "incorrect": a key number or fact differs from the reference, the answer contradicts it, or it answers a different question.
- "refusal": the answer declines, says the information is missing, or does not answer the question.
Reply in JSON with a one-sentence reason."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
    },
    "required": ["reason", "verdict"],
    "additionalProperties": False,
}


def build_messages(question: str, reference: str, justification: str | None, candidate: str) -> list[dict]:
    ref = reference if not justification else f"{reference}\nAnalyst's justification: {justification}"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nReference answer: {ref}\n\nAnswer to grade: {candidate}"}]


def verdict_of(out: dict | None) -> str:
    """Judge output that failed to parse counts as incorrect (logged separately by the caller)."""
    v = (out or {}).get("verdict")
    return v if v in VERDICTS else "incorrect"
