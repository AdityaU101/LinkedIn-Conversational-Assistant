"""Maximal Marginal Relevance: argmax_d [λ·rel(d) − (1−λ)·max_{s∈S} sim(d, s)]."""
from __future__ import annotations

import numpy as np


def mmr(relevance: np.ndarray, sim: np.ndarray, k: int, lam: float = 0.7) -> list[int]:
    """Select ``k`` indices. ``relevance`` should be min-max normalised to [0,1];
    ``sim`` is the pairwise similarity matrix of the candidates."""
    n = len(relevance)
    if n == 0:
        return []
    selected: list[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best = max(remaining, key=lambda i: lam * relevance[i] - (1 - lam) * max(sim[i, j] for j in selected))
        selected.append(best)
        remaining.remove(best)
    return selected


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)
