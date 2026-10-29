"""Matching, skill-gap and explanation services."""
from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Optional

from backend.database.sqlite import log_activity
from backend.repositories import candidates as repo
from common.schemas import Job
from explanations.llm_explainer import explain
from explanations.template import template_explanation
from explanations.verifier import verify
from matching.pipeline import METHODS, get_pipeline
from matching.skill_gap import compute_gap, related_occupations
from matching.store import JobFilters, get_store

_lock = threading.Lock()


def job_card(job: Job) -> dict:
    return {k: getattr(job, k) for k in (
        "job_id", "source", "source_job_id", "title", "company", "location", "work_mode", "employment_type",
        "seniority", "salary_min", "salary_max", "salary_currency", "salary_period", "posting_date", "source_url",
        "attribution", "is_historical", "industry")} | {
        "occupation": job.occupation.canonical_title if job.occupation else None,
        "occupation_code": job.occupation.code if job.occupation else None}


def recommend(candidate_id: str, filters: Optional[dict] = None, method: str = "final", k: int = 10,
              explain_with_llm: bool = False, use_cross_encoder: bool = True) -> dict:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    cand = repo.get_candidate(candidate_id)
    if cand is None:
        raise KeyError(candidate_id)
    f = JobFilters(**(filters or {}))
    with _lock:  # models are shared; serialise heavy inference
        resp = get_pipeline().recommend(cand, f, method=method, k=k, use_cross_encoder=use_cross_encoder)
    store = get_store()
    results = []
    for r in resp.results:
        job = store.by_id(r.job_id)
        exp = explain(r.gap, prefer_llm=explain_with_llm) if r.gap else None
        results.append({
            "rank": r.rank, "score": r.score, "job": job_card(job),
            "component_scores": r.component_scores, "component_ranks": r.component_ranks, "features": r.features,
            "matched_skills": [m.model_dump() for m in (r.gap.matched_required + r.gap.matched_preferred)],
            "missing_required": [m.model_dump() for m in r.gap.missing_required],
            "missing_preferred": [m.model_dump() for m in r.gap.missing_preferred],
            "transferable": [m.model_dump() for m in r.gap.transferable],
            "coverage_required": r.gap.coverage_required,
            "explanation": exp.model_dump() if exp else None,
        })
    run_id = repo.save_match_run(candidate_id, method, asdict(f), resp.ranker, results, resp.stages, resp.timings_ms)
    log_activity("recommendation", f"{method}: {len(results)} jobs ({resp.ranker})", candidate_id)
    return {"run_id": run_id, "candidate_id": candidate_id, "method": method, "ranker": resp.ranker,
            "filtered_pool": resp.filtered_pool, "stages": resp.stages, "timings_ms": resp.timings_ms,
            "results": results}


def skill_gap(candidate_id: str, job_id: Optional[str] = None) -> dict:
    cand = repo.get_candidate(candidate_id)
    if cand is None:
        raise KeyError(candidate_id)
    store = get_store()
    if job_id is None:
        run = repo.latest_match_run(candidate_id)
        if not run or not run["results"]:
            raise LookupError("No job specified and no recommendations yet")
        job_id = run["results"][0]["job"]["job_id"]
    job = store.by_id(job_id)
    if job is None:
        raise KeyError(job_id)
    gap = compute_gap(cand, job)
    codes = [o.code for o in cand.occupations if o.code]
    return {"gap": gap.model_dump(), "job": job_card(job),
            "candidate_skills": [{"skill_id": s.canonical_id, "name": s.canonical_name or s.mention,
                                  "confidence": s.confidence, "method": s.method} for s in cand.skills],
            "related_occupations": [r.model_dump() for r in related_occupations(codes)]}


def explanation(candidate_id: str, job_id: str, method: str = "auto") -> dict:
    cand = repo.get_candidate(candidate_id)
    job = get_store().by_id(job_id)
    if cand is None or job is None:
        raise KeyError(candidate_id if cand is None else job_id)
    gap = compute_gap(cand, job)
    if method == "template":
        exp = template_explanation(gap)
        v = verify(exp.text, exp.registry)
        exp.verification = {k: v[k] for k in ("n_sentences", "n_factual", "n_supported", "citation_validity",
                                              "faithfulness", "hallucination_rate")}
    else:
        exp = explain(gap, prefer_llm=method in ("auto", "llm"))
    return exp.model_dump()
