"""Experiment D — ranking methods against human relevance judgments.

Without judgments the experiment reports ``not_evaluated`` and only
descriptive statistics (latency, overlap between methods, skill coverage of
the top-10) that are explicitly *not* relevance measures.
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np

from evaluation.ranking import evaluate_rankings
from evaluation.registry import write_result
from matching import lambdamart
from matching.pipeline import METHODS, get_pipeline


def _candidates(ids: list[str]):
    from backend.repositories import candidates as repo

    return [c for c in (repo.get_candidate(i) for i in ids) if c]


def descriptive(cands, k: int = 10) -> dict:
    p = get_pipeline()
    lat = {m: [] for m in METHODS}
    lists = {m: [] for m in METHODS}
    cov = {m: [] for m in METHODS}
    for c in cands:
        for m in METHODS:
            if m == "dense" and p.dense is None:
                continue
            t0 = time.perf_counter()
            res = p.recommend(c, method=m, k=k, with_gap=True, use_cross_encoder=(m == "final"))
            lat[m].append((time.perf_counter() - t0) * 1000)
            lists[m].append([r.job_id for r in res.results])
            cv = [r.gap.coverage_required for r in res.results if r.gap and r.gap.coverage_required is not None]
            cov[m].append(float(np.mean(cv)) if cv else 0.0)
    overlap = {}
    for a, b in combinations([m for m in METHODS if lists[m]], 2):
        overlap[f"{a}~{b}"] = round(float(np.mean([len(set(x) & set(y)) / k for x, y in zip(lists[a], lists[b])])), 3)
    return {"NOT_RELEVANCE_METRICS": True, "candidates": len(cands),
            "latency_ms_mean": {m: round(float(np.mean(v)), 1) for m, v in lat.items() if v},
            "top10_overlap": overlap,
            "mean_required_skill_coverage@10": {m: round(float(np.mean(v)), 4) for m, v in cov.items() if v}}


def run_silver(n: int = 60) -> dict:
    from evaluation import silver

    res = silver.run_ranking(silver.test_sample(n))
    res["labels"] = ("silver: a job is relevant when its title matches one of the resume's resume_corpus expert "
                     "occupation classes (evaluation/silver.py TITLE_RULES); complete labels over all indexed jobs")
    return res


def run(candidate_ids: list[str], n_silver: int = 60) -> dict:
    j = lambdamart.load_judgments()
    cands = _candidates(candidate_ids)
    desc = descriptive(cands[:20]) if cands else {}
    silver_res = run_silver(n_silver)
    if j.empty:
        return write_result("D_ranking", "partially_evaluated", silver=silver_res,
                            reason="No human relevance judgments yet (Data/annotations/relevance/judgments.csv): "
                                   "human nDCG/MRR/P@5 not evaluated, LambdaMART awaits annotations (fallback ranker "
                                   "used). Silver relevance from resume_corpus expert labels is reported separately.",
                            descriptive=desc, details={"lambdamart": lambdamart.status()})
    qrels: dict[str, dict[str, int]] = {}
    for r in j.itertuples():
        qrels.setdefault(r.candidate_id, {})[r.job_id] = int(r.label)
    judged = _candidates(list(qrels))
    p = get_pipeline()
    metrics = {}
    for m in METHODS:
        if m == "dense" and p.dense is None:
            continue
        runs = {c.candidate_id: [r.job_id for r in p.recommend(c, method=m, k=50, with_gap=False,
                                                              use_cross_encoder=(m == "final")).results] for c in judged}
        metrics[m] = evaluate_rankings(runs, qrels)
    for name, kw in (("final_no_cross_encoder", {"use_cross_encoder": False}), ("final_no_mmr", {"use_mmr": False}),
                     ("final_no_graph_features", {"drop_graph_features": True})):
        runs = {c.candidate_id: [r.job_id for r in p.recommend(c, method="final", k=50, with_gap=False, **kw).results]
                for c in judged}
        metrics[name] = evaluate_rankings(runs, qrels)
    return write_result("D_ranking", "evaluated", metrics, descriptive=desc, silver=silver_res,
                        details={"judged_pairs": int(len(j)), "judged_candidates": len(qrels),
                                 "lambdamart": lambdamart.status(),
                                 "protocol": "condensed lists (unjudged documents removed), graded nDCG with 2^rel-1 gain"})
