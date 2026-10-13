"""Figures for the README, each in a light and a dark version (docs/figures/<name>-{light,dark}.png).

    python plots.py

Reads the summaries in results/ (and results/raw/ for the reasoning experiment). Colours: two categorical
slots checked for colour-vision deficiency on both surfaces, plus neutral greys for context; text always
uses the ink colours, never a series colour.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from core import metrics  # noqa: E402

BASE = Path(__file__).resolve().parent
RES = BASE / "results"
OUT = BASE / "docs" / "figures"
DPI = 200
PX = DPI / 100          # figure pixels per screen pixel: the PNGs are drawn at twice the size they are shown

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "s1": "#3987e5", "s2": "#d95926"},
}

# FinanceBench paper, gpt-4-1106-preview, human-graded, same 150 questions (results/RESULTS.md, paper baselines)
PAPER = {"all filings": 0.193, "own filing": 0.500, "evidence pages": 0.853}


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def verdicts(name: str) -> dict[str, str]:
    rows = (RES / "raw" / f"grade_{name}.jsonl").read_text().splitlines()
    return {r["id"]: r["verdict"] for r in map(json.loads, filter(str.strip, rows))}


def ms(value: float) -> str:
    return f"{value:.0f} ms" if value < 1000 else f"{value / 1000:.2f} s"


def minus(text: str) -> str:
    return text.replace("-", "−")


# ---------------------------------------------------------------- drawing helpers

def new_figure(t: dict, width: float, height: float, rect: tuple):
    """A figure with one axes for horizontal bars: vertical hairline grid, baseline at x = 0, no other spines."""
    fig = plt.figure(figsize=(width, height), dpi=DPI, facecolor=t["surface"])
    ax = fig.add_axes(rect, facecolor=t["surface"])
    style_axes(ax, t)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(t["axis"])
    ax.spines["left"].set_linewidth(0.75)
    return fig, ax


def style_axes(ax, t: dict) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=8.5, pad=5)
    ax.tick_params(axis="x", labelcolor=t["muted"], labelsize=8)
    ax.grid(axis="x", color=t["grid"], linewidth=0.75)
    ax.set_axisbelow(True)


def heading(fig, t: dict, text: str, sub: str) -> None:
    fig.text(0.025, 0.955, text, color=t["ink"], fontsize=12, fontweight="semibold", va="top")
    fig.text(0.025, 0.955 - 0.36 / fig.get_figheight(), sub, color=t["ink2"], fontsize=8.5, va="top")


def note(fig, t: dict, text: str) -> None:
    fig.text(0.025, 0.1 / fig.get_figheight(), text, color=t["muted"], fontsize=7.5, va="bottom")


def legend_row(fig, t: dict, items: list[tuple[str, str, str]], top: float) -> None:
    """Legend under the subtitle: a coloured key mark beside ink text, per series, left to right."""
    y = 1 - top / fig.get_figheight()
    x = 0.025
    renderer = fig.canvas.get_renderer()
    for color, marker, label in items:
        fig.lines.append(plt.Line2D([x + 0.007], [y], transform=fig.transFigure, marker=marker, markersize=7,
                                    color=color, markeredgecolor=t["surface"], markeredgewidth=0, linestyle=""))
        txt = fig.text(x + 0.02, y, label, color=t["ink2"], fontsize=8.5, va="center")
        x += 0.02 + txt.get_window_extent(renderer).width / fig.bbox.width + 0.035


def px_size(ax) -> tuple[float, float]:
    """Data units per figure pixel along x and y (the limits must be final)."""
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (x1 - x0) / box.width, abs(y1 - y0) / box.height


def hbar(ax, y: float, value: float, color: str, thick: float = 14) -> None:
    """Horizontal bar from the baseline: square there, 4px rounded at the data end, at most 24px thick."""
    if value <= 0:
        return
    dx, dy = px_size(ax)
    h = min(thick, 24) * PX * dy
    r = min(4 * PX * dx, value / 2)
    ax.add_patch(FancyBboxPatch((0, y - h / 2), value, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=dy / dx, facecolor=color, edgecolor="none", zorder=3, snap=False))
    ax.add_patch(Rectangle((0, y - h / 2), min(value, r), h, facecolor=color, edgecolor="none", zorder=3, snap=False))


def whisker(ax, y: float, lo: float, hi: float, color: str) -> None:
    ax.plot([lo, hi], [y, y], color=color, linewidth=1.1, solid_capstyle="butt", zorder=4)
    for x in (lo, hi):
        ax.plot([x, x], [y - 0.07, y + 0.07], color=color, linewidth=1.1, zorder=4)


def separator(ax, t: dict, y: float) -> None:
    ax.axhline(y, color=t["grid"], linewidth=0.75, zorder=1)


def percent_axis(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")


def save(fig, name: str, theme: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}-{theme}.png"
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figures

def fig_accuracy(t: dict):
    """FinRAG against GPT-4-Turbo from the FinanceBench paper, in the two settings both were run in."""
    ours = {"all filings": load("grade_retrieved_dyr.json")["correct"], "evidence pages": load("grade_gold.json")["correct"]}
    rows = [("evidence pages", "Labelled evidence pages given"), ("all filings", "Searching all 368 filings")]
    fig, ax = new_figure(t, 7.2, 3.3, rect=(0.29, 0.17, 0.64, 0.50))
    percent_axis(ax)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    for k, (key, _) in enumerate(rows):
        m, lo, hi = (100 * v for v in ours[key])
        hbar(ax, k + 0.17, m, t["s1"])
        whisker(ax, k + 0.17, lo, hi, t["ink2"])
        ax.text(hi + 1.5, k + 0.17, f"{m:.1f}%", color=t["ink"], fontsize=8.5, va="center", fontweight="semibold")
        p = 100 * PAPER[key]
        hbar(ax, k - 0.17, p, t["s2"])
        ax.text(p + 1.5, k - 0.17, f"{p:.1f}%", color=t["ink"], fontsize=8.5, va="center")
    heading(fig, t, "Correct answers on FinanceBench, 150 questions",
            "A local 4B model: its lead when searching comes from retrieval (company/year filters and reranking)")
    legend_row(fig, t, [(t["s1"], "s", "FinRAG, Qwen3.5-4B (95% CI)"),
                        (t["s2"], "s", "GPT-4-Turbo, FinanceBench paper")], top=0.84)
    note(fig, t, "Paper runs (2023) are graded by humans; FinRAG by an LLM judge that agrees with blind hand grades "
                 "on 27 of 30 answers (κ = 0.80).")
    return fig


RETRIEVAL = [
    ("bm25", "BM25"),
    ("hybrid+company", "Hybrid + company filter"),
    ("hybrid", "Hybrid: BM25 + dense, RRF"),
    ("dense", "Dense, Qwen3-Embedding-0.6B"),
    ("hybrid+company+rerank", "Hybrid + company + rerank"),
    ("dense+company+rerank", "Dense + company + rerank"),
    ("hybrid+company+year+rerank", "Hybrid + company + year + rerank"),
    ("dense+company+year+rerank", "Dense + company + year + rerank"),
]
SERVED = "dense+company+year+rerank"
REFERENCE = ("scoped-filing: hybrid+rerank", "Reference: the question's filing only")


def fig_retrieval(t: dict):
    """Labelled evidence page among the top 5 chunks, per retrieval configuration; the served one emphasised."""
    s = load("search.json")["configs"]
    rows = sorted(RETRIEVAL, key=lambda kv: s[kv[0]]["page@5"][0]) + [REFERENCE]
    fig, ax = new_figure(t, 7.2, 4.6, rect=(0.335, 0.12, 0.60, 0.66))
    percent_axis(ax)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    for k, (key, _) in enumerate(rows):
        m, lo, hi = (100 * v for v in s[key]["page@5"])
        served = key == SERVED
        hbar(ax, k, m, t["s1"] if served else t["muted"], thick=13)
        whisker(ax, k, lo, hi, t["ink2"])
        ax.text(hi + 1.5, k, f"{m:.1f}%", color=t["ink"] if served else t["ink2"], fontsize=8.5, va="center",
                fontweight="semibold" if served else "normal")
    for tick, (key, _) in zip(ax.get_yticklabels(), rows):
        if key == SERVED:
            tick.set_color(t["ink"])
            tick.set_fontweight("semibold")
        elif key == REFERENCE[0]:
            tick.set_color(t["muted"])
    separator(ax, t, len(rows) - 1.5)
    heading(fig, t, "Retrieval: the labelled evidence page among the top 5 chunks",
            "150 questions over 111,561 chunks from 368 filings. Blue: the configuration the API serves")
    note(fig, t, "Whiskers: 95% bootstrap CI. Rerank: Qwen3-Reranker-0.6B over the top 50. Company and year are read "
                 "from the question.")
    return fig


ARMS = [("calc", "Calculator tool"), ("calcr", "Calculator, planner reasons first"),
        ("unit", "Unit and rounding rule"), ("think", "Step-by-step reasoning")]
CONTEXTS = [("gold", "gold", "Labelled evidence pages given"), ("retrieved", "retrieved_dyr", "Searching all filings")]


def fig_experiment(t: dict):
    """Each reasoning arm's paired change against prompt v1: development half against the held-out half."""
    split = load("split.json")
    fig = plt.figure(figsize=(7.6, 4.5), dpi=DPI, facecolor=t["surface"])
    rows = list(reversed(ARMS))                    # first arm at the top
    for c, (ctx, base, ctx_label) in enumerate(CONTEXTS):
        ax = fig.add_axes((0.315 + c * 0.345, 0.16, 0.30, 0.55), facecolor=t["surface"])
        style_axes(ax, t)
        ax.set_xlim(-30, 30)
        ax.set_xticks(range(-20, 21, 10))
        ax.xaxis.set_major_formatter(lambda v, _: minus(f"{v:+.0f}") if v else "0")
        ax.set_ylim(-0.85, len(rows) - 0.45)
        ax.axvline(0, color=t["axis"], linewidth=1.0, zorder=2)
        ax.set_yticks(range(len(rows)), [label for _, label in rows] if c == 0 else [])
        ax.set_title(ctx_label, color=t["ink"], fontsize=9, loc="left", pad=8)
        ax.set_xlabel("points vs prompt v1", color=t["muted"], fontsize=7.5, labelpad=4)
        ref = verdicts(base)
        for k, (arm, _) in enumerate(rows):
            got = verdicts(f"{ctx}_{arm}")
            for half, color, marker, dy in (("dev", t["s1"], "o", 0.15), ("test", t["s2"], "D", -0.15)):
                ids = [i for i in split[half] if i in got and i in ref]
                d = metrics.paired_diff([got[i] == "correct" for i in ids], [ref[i] == "correct" for i in ids])
                m, lo, hi = 100 * d["diff"], 100 * d["lo"], 100 * d["hi"]
                ax.plot([lo, hi], [k + dy] * 2, color=color, linewidth=1.4, solid_capstyle="round", zorder=4)
                ax.plot(m, k + dy, marker=marker, markersize=6.5 if marker == "o" else 5.5, color=color,
                        markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=5)
                if arm == "think" and half == "test":
                    ax.text(m, k + dy - 0.36, minus(f"{m:+.1f} points, p = {d['p']:.2f}"), color=t["ink"],
                            fontsize=7.5, ha="center", va="center", fontweight="semibold")
        if c == 0:
            for tick, (arm, _) in zip(ax.get_yticklabels(), rows):
                if arm == "think":
                    tick.set_color(t["ink"])
                    tick.set_fontweight("semibold")
    heading(fig, t, "Reasoning variants: gains on the development half did not hold out",
            "Paired change in correct answers vs prompt v1 (95% CI). Split by company, fixed before any run")
    legend_row(fig, t, [(t["s1"], "o", "Development half, 75 questions"),
                        (t["s2"], "D", "Held-out half, 75 questions, scored once after the freeze")], top=0.80)
    note(fig, t, "Step-by-step reasoning won on the development half under a selection rule written in advance; "
                 "held out, it lost 13\u201315 points.")
    return fig


STAGES = [("embed_ms", "Embed the question"), ("search_ms", "Search and filter"), ("rerank_ms", "Rerank 50 chunks"),
          ("generate_ms", "Generate the cited answer"), ("total_ms", "Total")]


def fig_latency(t: dict):
    """Median and 95th-percentile time per stage of one /ask request."""
    lat = load("latency.json")
    seq, conc = lat["sequential"], lat["concurrent"]
    rows = list(reversed(STAGES))                  # pipeline order from the top, total last
    fig, ax = new_figure(t, 7.2, 3.6, rect=(0.29, 0.15, 0.63, 0.56))
    ax.set_xlim(0, 9000)
    ax.set_xticks(range(0, 8001, 2000))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f} s" if v else "0")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    for k, (key, _) in enumerate(rows):
        p50, p95 = seq["p50_ms"][key], seq["p95_ms"][key]
        total = key == "total_ms"
        hbar(ax, k, p50, t["s1"] if total else t["muted"], thick=13)
        ax.plot([p95, p95], [k - 0.22, k + 0.22], color=t["ink2"], linewidth=1.4, zorder=5)
        weight = "semibold" if total else "normal"
        if p95 - p50 > 1000:                       # room for the median between the bar and the tick
            ax.text(p50 + 110, k, ms(p50), color=t["ink"], fontsize=8.5, va="center", fontweight=weight)
            ax.text(p95 + 110, k, f"p95 {ms(p95)}", color=t["muted"], fontsize=7.5, va="center")
        else:
            ax.text(p95 + 110, k, ms(p50), color=t["ink"], fontsize=8.5, va="center", fontweight=weight)
    ax.get_yticklabels()[0].set_fontweight("semibold")
    ax.get_yticklabels()[0].set_color(t["ink"])
    separator(ax, t, 0.5)
    heading(fig, t, "Latency: one question through the HTTP API",
            f"Median per stage (bar) and 95th percentile (tick) over {lat['questions']} questions, sent one at a time")
    note(fig, t, f"Two RTX A4000 GPUs: generation on one, embedding and reranking on the other. "
                 f"With {conc['clients']} concurrent clients: {conc['questions_per_s']:.2f} questions/s.")
    return fig


FIGURES = {"accuracy": fig_accuracy, "retrieval": fig_retrieval, "experiment": fig_experiment, "latency": fig_latency}


def main() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = "Inter" if "Inter" in names else "DejaVu Sans"
    for name, make in FIGURES.items():
        for theme, t in THEMES.items():
            print(save(make(t), name, theme).relative_to(BASE))


if __name__ == "__main__":
    main()
