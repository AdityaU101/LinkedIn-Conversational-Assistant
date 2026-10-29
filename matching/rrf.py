"""Reciprocal Rank Fusion (Cormack et al., 2009): RRF(d) = Σ_r 1 / (k + rank_r(d))."""
from __future__ import annotations


def rrf(rankings: dict[str, list[int]], k: int = 60) -> tuple[list[int], dict[int, float], dict[int, dict[str, int]]]:
    """Fuse ranked lists of document positions.

    Returns (fused order, fused score per doc, per-component 1-based rank per doc)
    so component ranks stay available for evaluation and debugging.
    """
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, docs in rankings.items():
        for r, d in enumerate(docs, start=1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + r)
            ranks.setdefault(d, {})[name] = r
    order = sorted(scores, key=lambda d: (-scores[d], d))
    return order, scores, ranks
