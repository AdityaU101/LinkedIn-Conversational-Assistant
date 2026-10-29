"""Jaccard overlap between normalised skill-ID sets."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class JaccardRetriever:
    """Inverted index skill -> docs so only overlapping docs are scored."""

    def __init__(self, skill_sets: list[frozenset[str]]):
        self.sets = skill_sets
        self.inv: dict[str, list[int]] = {}
        for i, s in enumerate(skill_sets):
            for sid in s:
                self.inv.setdefault(sid, []).append(i)

    def scores(self, query: Iterable[str]) -> np.ndarray:
        q = frozenset(query)
        out = np.zeros(len(self.sets), dtype=np.float32)
        cand: set[int] = set()
        for sid in q:
            cand.update(self.inv.get(sid, ()))
        for i in cand:
            out[i] = jaccard(q, self.sets[i])
        return out
