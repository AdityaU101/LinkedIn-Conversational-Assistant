"""Ranking metrics over graded relevance (0/1/2) and per-method evaluation."""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


def dcg(rels: Iterable[float], k: int) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(list(rels)[:k]))


def ndcg_at_k(ranked_rels: list[float], k: int = 10, all_rels: Optional[list[float]] = None) -> float:
    """nDCG@k. ``all_rels`` = every judged label for the query (ideal ranking);
    defaults to the ranked list itself."""
    ideal = sorted(all_rels if all_rels is not None else ranked_rels, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(ranked_rels, k) / idcg if idcg > 0 else 0.0


def mrr(ranked_rels: list[float], threshold: float = 1) -> float:
    for i, r in enumerate(ranked_rels, start=1):
        if r >= threshold:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked_rels: list[float], k: int = 5, threshold: float = 1) -> float:
    top = ranked_rels[:k]
    return sum(1 for r in top if r >= threshold) / k if k else 0.0


def evaluate_rankings(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], k_ndcg: int = 10,
                      k_p: int = 5, judged_only: bool = True) -> dict:
    """``rankings``: query -> ranked doc ids; ``qrels``: query -> {doc: label}.

    With ``judged_only`` unjudged documents are removed (condensed lists,
    Sakai 2007) instead of being assumed irrelevant — appropriate for pooled,
    incomplete judgments.
    """
    nd, rr, pk, cov = [], [], [], []
    for q, docs in rankings.items():
        if q not in qrels:
            continue
        judged = qrels[q]
        seq = [d for d in docs if d in judged] if judged_only else docs
        rels = [judged.get(d, 0) for d in seq]
        cov.append(sum(1 for d in docs[:k_ndcg] if d in judged) / max(1, min(k_ndcg, len(docs))))
        nd.append(ndcg_at_k(rels, k_ndcg, list(judged.values())))
        rr.append(mrr(rels))
        pk.append(precision_at_k(rels, k_p))
    if not nd:
        return {"queries": 0}
    return {"queries": len(nd), f"ndcg@{k_ndcg}": round(float(np.mean(nd)), 4), "mrr": round(float(np.mean(rr)), 4),
            f"precision@{k_p}": round(float(np.mean(pk)), 4), f"judged@{k_ndcg}": round(float(np.mean(cov)), 4)}
