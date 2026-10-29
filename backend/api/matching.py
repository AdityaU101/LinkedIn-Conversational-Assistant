from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.repositories import candidates as repo
from backend.schemas.api import ExplainIn, JobCandidatesIn, RecommendIn
from backend.services import matching_service as svc

router = APIRouter(prefix="/api", tags=["matching"])


@router.post("/matching/recommend")
def recommend(body: RecommendIn):
    try:
        return svc.recommend(body.candidate_id, body.filters.model_dump(), body.method, body.k, body.explain_with_llm,
                             body.use_cross_encoder)
    except KeyError:
        raise HTTPException(404, "Candidate not found")
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("/matching/{candidate_id}/results")
def results(candidate_id: str, method: str | None = Query(None)):
    run = repo.latest_match_run(candidate_id, method)
    if run is None:
        raise HTTPException(404, "No recommendations stored for this candidate yet")
    return run


@router.post("/matching/job-candidates")
def job_candidates(body: JobCandidatesIn):
    from matching.pipeline import get_pipeline
    from matching.store import get_store

    job = get_store().by_id(body.job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    cands = repo.list_candidates(body.source)
    out = get_pipeline().recommend_candidates(job, cands, body.k)
    for o in out:
        o["gap"] = o["gap"].model_dump()
    return {"job": svc.job_card(job), "pool": len(cands), "results": out}


@router.get("/skills/{candidate_id}/gaps")
def gaps(candidate_id: str, job_id: str | None = None):
    try:
        return svc.skill_gap(candidate_id, job_id)
    except KeyError:
        raise HTTPException(404, "Candidate or job not found")
    except LookupError as e:
        raise HTTPException(409, str(e))


@router.post("/explanations")
def explanation(body: ExplainIn):
    try:
        return svc.explanation(body.candidate_id, body.job_id, body.method)
    except KeyError:
        raise HTTPException(404, "Candidate or job not found")
