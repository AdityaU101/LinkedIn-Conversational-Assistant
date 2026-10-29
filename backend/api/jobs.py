from __future__ import annotations

from dataclasses import asdict

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.schemas.api import JobSearchIn
from backend.services.matching_service import job_card
from matching.store import JobFilters, get_store

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), source: str | None = None,
              work_mode: str | None = None, location: str | None = None, title: str | None = None):
    store = get_store()
    f = JobFilters(sources=[source] if source else [], work_modes=[work_mode] if work_mode else [], location=location,
                   title_contains=title)
    idx = np.flatnonzero(store.mask(f))
    page = idx[offset:offset + limit]
    return {"items": [job_card(store.job(int(i))) for i in page], "total": int(len(idx)), "offset": offset,
            "limit": limit, "sources": store.source_counts()}


@router.get("/{job_id:path}")
def get_job(job_id: str):
    job = get_store().by_id(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {"job": job.model_dump(exclude={"raw"}) | {"raw": job.raw}, "card": job_card(job)}


@router.post("/search")
def search(body: JobSearchIn):
    from matching.pipeline import get_pipeline

    store = get_store()
    mask = store.mask(JobFilters(**body.filters.model_dump()))
    if body.query.strip():
        scores = get_pipeline().bm25.scores(body.query)
        scores = np.where(mask, scores, -np.inf)
        order = [int(i) for i in np.argsort(-scores) if np.isfinite(scores[i]) and scores[i] > 0]
    else:
        order = [int(i) for i in np.flatnonzero(mask)]
    page = order[body.offset:body.offset + body.limit]
    return {"items": [job_card(store.job(i)) for i in page], "total": len(order), "offset": body.offset,
            "limit": body.limit, "ranking": "bm25" if body.query.strip() else "none"}
