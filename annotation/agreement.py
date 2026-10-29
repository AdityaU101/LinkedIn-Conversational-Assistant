"""Inter-annotator agreement: Cohen's kappa (nominal / linear / quadratic weights),
Krippendorff's alpha (nominal / ordinal) and span-level F1 between annotators."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Hashable, Iterable, Optional

import numpy as np


def cohen_kappa(a: list[Hashable], b: list[Hashable], weights: Optional[str] = None) -> Optional[float]:
    """weights: None (nominal), 'linear' or 'quadratic' (ordinal labels must be numeric)."""
    if len(a) != len(b) or not a:
        return None
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    if k == 1:
        return 1.0
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    if weights is None:
        W = 1 - np.eye(k)
    else:
        vals = np.array(cats, dtype=float)
        d = np.abs(vals[:, None] - vals[None, :]) / (vals.max() - vals.min())
        W = d if weights == "linear" else d ** 2
    denom = (W * E).sum()
    return float(1 - (W * O).sum() / denom) if denom else 1.0


def krippendorff_alpha(units: dict[Hashable, list], level: str = "nominal") -> Optional[float]:
    """``units``: item -> list of labels from different annotators (missing allowed by omission)."""
    pairs: list[tuple] = []
    for labels in units.values():
        labels = [l for l in labels if l is not None]
        m = len(labels)
        if m < 2:
            continue
        for i, j in combinations(range(m), 2):
            pairs.append((labels[i], labels[j], 1.0 / (m - 1)))
            pairs.append((labels[j], labels[i], 1.0 / (m - 1)))
    if not pairs:
        return None
    vals = sorted({p[0] for p in pairs})
    o = defaultdict(float)
    for x, y, w in pairs:
        o[(x, y)] += w
    n_c = Counter()
    for (x, _), w in o.items():
        n_c[x] += w
    n = sum(n_c.values())

    if level == "nominal":
        delta = lambda x, y: 0.0 if x == y else 1.0  # noqa: E731
    else:  # ordinal (Krippendorff 2011)
        order = {v: i for i, v in enumerate(vals)}

        def delta(x, y):
            lo, hi = sorted((order[x], order[y]))
            s = sum(n_c[vals[g]] for g in range(lo, hi + 1)) - (n_c[x] + n_c[y]) / 2
            return s ** 2

    Do = sum(w * delta(x, y) for (x, y), w in o.items())
    De = sum(n_c[x] * n_c[y] * delta(x, y) for x in vals for y in vals) / (n - 1)
    return float(1 - Do / De) if De else 1.0


def span_f1(spans_a: Iterable[tuple], spans_b: Iterable[tuple]) -> dict:
    """Exact-match F1 between two annotators' (doc, label, start, end) span sets."""
    a, b = set(spans_a), set(spans_b)
    tp = len(a & b)
    p = tp / len(b) if b else 0.0
    r = tp / len(a) if a else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "a": len(a), "b": len(b)}
