"""Retrieval metrics and bootstrap confidence intervals (numpy only)."""
from __future__ import annotations

import numpy as np


def first_hit(ranked: list, gold: set) -> int | None:
    """1-based rank of the first item in `gold`, or None."""
    for i, item in enumerate(ranked, 1):
        if item in gold:
            return i
    return None


def hits_at(ranks: list[int | None], k: int) -> np.ndarray:
    return np.array([r is not None and r <= k for r in ranks], dtype=float)


def reciprocal_ranks(ranks: list[int | None], k: int) -> np.ndarray:
    return np.array([1.0 / r if r is not None and r <= k else 0.0 for r in ranks])


def bootstrap_ci(values, n: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval, resampling questions."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff(a, b, n: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean of a - b over the same questions, its bootstrap interval, and a two-sided bootstrap p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)), "p": float(min(1.0, p))}
