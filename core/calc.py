"""Calculator tool, in two passes: the model lists the figures it needs and writes the arithmetic over them
(JSON); Python evaluates each expression exactly; the model then answers with the results in its prompt.

The evaluator walks the expression's syntax tree and allows only numbers, figure names, + - * / **,
unary minus and abs/min/max/round/sqrt, so nothing the model writes can run as code.
"""
from __future__ import annotations

import ast
import math
import operator

PLAN_SYSTEM = """You prepare the calculations a financial analyst needs to answer a question from numbered excerpts of SEC filings.
- In "figures", list every number from the excerpts that the answer needs: a short snake_case name (for example cogs_2017), the value exactly as printed in the excerpt's own units (no commas or currency signs; a number in parentheses is negative), and the excerpt number.
- In "calculations", write each value the question asks for as an arithmetic expression over those names (operators + - * / ** and parentheses; functions abs, min, max, round, sqrt), with a short snake_case label. A later calculation may use an earlier one by its label. Follow the question's own definition of a metric exactly when it gives one; otherwise use the standard definition.
- If the question needs no arithmetic, leave "calculations" empty."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "figures": {"type": "array", "maxItems": 12, "items": {
            "type": "object",
            "properties": {"name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,40}$"},
                           "value": {"type": "number"},
                           "excerpt": {"type": "integer", "minimum": 1, "maximum": 8}},
            "required": ["name", "value", "excerpt"], "additionalProperties": False}},
        "calculations": {"type": "array", "maxItems": 4, "items": {
            "type": "object",
            "properties": {"label": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,40}$"},
                           "expression": {"type": "string", "maxLength": 300}},
            "required": ["label", "expression"], "additionalProperties": False}},
    },
    "required": ["figures", "calculations"],
    "additionalProperties": False,
}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"abs": abs, "min": min, "max": max, "round": round, "sqrt": math.sqrt}


def safe_eval(expression: str, names: dict[str, float]) -> float:
    def ev(node) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"unknown figure '{node.id}'")
            return float(names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return float(_BIN[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return float(_UNARY[type(node.op)](ev(node.operand)))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
                and not node.keywords and node.args):
            return float(_FUNCS[node.func.id](*[ev(a) for a in node.args]))
        raise ValueError(f"not allowed in a calculation: {type(node).__name__}")
    return ev(ast.parse(expression, mode="eval"))


def _fmt(x: float) -> str:
    return f"{x:.6g}" if abs(x) < 1e15 else f"{x:.6e}"


def run(plan: dict | None) -> list[dict]:
    """Evaluate every planned calculation; each result carries either a value or an error message."""
    if not plan:
        return []
    names = {f["name"]: f["value"] for f in plan.get("figures", [])}
    out = []
    for c in plan.get("calculations", []):
        try:
            value = safe_eval(c["expression"], names)
            out.append({"label": c["label"], "expression": c["expression"], "value": value})
            names.setdefault(c["label"], value)          # later calculations may build on this one
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError) as e:
            out.append({"label": c["label"], "expression": c["expression"], "error": str(e) or type(e).__name__})
    return out


def results_block(plan: dict | None, results: list[dict]) -> str:
    """Text appended to the answer prompt; empty when nothing was calculated."""
    if not results:
        return ""
    figures = ", ".join(f"{f['name']} = {_fmt(f['value'])} (excerpt {f['excerpt']})" for f in (plan or {}).get("figures", []))
    lines = [f"- {r['label']}: {r['expression']} = {_fmt(r['value'])}" if "value" in r
             else f"- {r['label']}: {r['expression']} could not be computed ({r['error']})" for r in results]
    return ("\n\nCalculator results, computed exactly from figures taken from the excerpts (" + figures + "):\n"
            + "\n".join(lines) + "\nUse these results instead of redoing the arithmetic; check that the figures and "
            "formulas fit the question before relying on them.")
