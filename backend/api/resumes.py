from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.repositories import candidates as repo
from backend.schemas.api import ProfileIn
from backend.services import resume_service
from common.config import get_settings

router = APIRouter(prefix="/api", tags=["resumes"])


@router.post("/resumes/upload")
async def upload(file: UploadFile = File(...), use_llm: str = Query("auto", pattern="^(auto|true|false)$")):
    max_bytes = int(float(get_settings().get("parsing.max_upload_mb", 5)) * 1024 * 1024)
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {max_bytes // (1024 * 1024)} MB")
    flag = {"auto": "auto", "true": True, "false": False}[use_llm]
    try:
        return resume_service.handle_upload(data, file.filename or "resume", use_llm=flag)
    except resume_service.UploadError as e:
        raise HTTPException(422, str(e))


@router.get("/resumes")
def list_resumes(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), source: str | None = None):
    items, total = repo.candidate_summaries(source, limit, offset)
    return {"items": items, "total": total, "offset": offset, "limit": limit}


@router.get("/resumes/{resume_id}")
def get_resume(resume_id: str):
    r = repo.get_resume(resume_id)
    c = repo.get_candidate(resume_id)
    if c is None:
        raise HTTPException(404, "Resume not found")
    meta = {k: v for k, v in (r or {}).items() if k != "stored_path"}
    return {"resume": meta or {"resume_id": resume_id, "status": "imported"}, "text": c.raw_text}


@router.get("/resumes/{resume_id}/analysis")
def analysis(resume_id: str):
    c = repo.get_candidate(resume_id)
    if c is None:
        raise HTTPException(404, "Resume not found")
    return {"candidate": c.model_dump(), "resume": {k: v for k, v in (repo.get_resume(resume_id) or {}).items()
                                                    if k != "stored_path"},
            "stats": {"skills": len(c.skills), "experiences": len(c.experiences), "education": len(c.education),
                      "credentials": len(c.credentials), "projects": len(c.projects),
                      "occupations": len(c.occupations), "evidence_spans": len(c.all_evidence())}}


@router.put("/resumes/{resume_id}/profile")
def update_profile(resume_id: str, body: ProfileIn):
    try:
        c = resume_service.update_profile(resume_id, body.profile)
    except KeyError:
        raise HTTPException(404, "Resume not found")
    return {"candidate": c.model_dump(), "saved": True}


@router.post("/resumes/{resume_id}/reparse")
def reparse(resume_id: str, use_llm: str = Query("auto", pattern="^(auto|true|false)$"), force: bool = False):
    flag = {"auto": "auto", "true": True, "false": False}[use_llm]
    try:
        return resume_service.reparse_resume(resume_id, use_llm=flag, force=force)
    except KeyError:
        raise HTTPException(404, "Resume not found (only uploaded resumes can be re-parsed)")
    except resume_service.ReparseConflict as e:
        raise HTTPException(409, str(e))
    except resume_service.UploadError as e:
        raise HTTPException(422, str(e))


@router.delete("/resumes/{resume_id}")
def delete(resume_id: str):
    if repo.get_candidate(resume_id) is None:
        raise HTTPException(404, "Resume not found")
    return resume_service.delete_resume(resume_id)
