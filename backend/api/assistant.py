from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agents.critic_agent import render, strip_citations
from agents.graph import run_agents
from backend.repositories import candidates as repo
from backend.schemas.api import ChatIn, FollowupIn, FollowupStatusIn, OptimizeIn, OutreachIn

router = APIRouter(prefix="/api", tags=["assistant"])


def _public(state: dict) -> dict:
    return {"answer": state.get("answer", ""), "body": state.get("body", state.get("answer", "")),
            "intent": state.get("intent"), "citations": state.get("citations", []),
            "critic": state.get("critic"), "requires_approval": state.get("requires_approval", False),
            "actions": state.get("actions", []), "injection_flags": state.get("injection_flags", []),
            "errors": state.get("errors", []), "trace": state.get("trace", []), "job_id": state.get("job_id")}


@router.post("/chat")
def chat(body: ChatIn):
    if body.candidate_id and repo.get_candidate(body.candidate_id) is None:
        raise HTTPException(404, "Candidate not found")
    repo.add_chat(body.candidate_id, "user", body.message)
    state = run_agents("chat", body.candidate_id, body.message, body.job_id)
    out = _public(state)
    out["suggestions"] = state.get("suggestions") or []
    out["outreach"] = state.get("data", {}).get("outreach")
    repo.add_chat(body.candidate_id, "assistant", out["answer"], {"intent": out["intent"], "critic": out["critic"],
                                                                  "citations": out["citations"]})
    return out


@router.get("/chat/{candidate_id}/history")
def history(candidate_id: str):
    return {"messages": repo.chat_history(candidate_id)}


@router.post("/resume/optimize")
def optimize(body: OptimizeIn):
    if repo.get_candidate(body.candidate_id) is None:
        raise HTTPException(404, "Candidate not found")
    state = run_agents("optimize", body.candidate_id)
    out = _public(state) | {"suggestions": state.get("suggestions", [])}
    if body.rewrite_bullet:
        from agents.profile_optimizer_agent import rewrite_bullet

        out["rewrite"] = rewrite_bullet(body.rewrite_bullet)
    return out


@router.post("/outreach/draft")
def outreach(body: OutreachIn):
    if repo.get_candidate(body.candidate_id) is None:
        raise HTTPException(404, "Candidate not found")
    try:
        state = run_agents("outreach", body.candidate_id, job_id=body.job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")
    out = _public(state)
    out["draft"] = out["answer"]
    out["unverified_draft"] = strip_citations(render(state.get("sentences") or []))
    out["outreach"] = state.get("data", {}).get("outreach")
    out["sent"] = False
    return out


@router.post("/followups")
def create_followup(body: FollowupIn):
    if repo.get_candidate(body.candidate_id) is None:
        raise HTTPException(404, "Candidate not found")
    state = run_agents("followup", body.candidate_id, message=body.when, job_id=body.job_id,
                       params={"note": body.note})
    return _public(state)


@router.get("/followups")
def list_followups(candidate_id: str | None = None):
    return {"items": repo.list_followups(candidate_id)}


@router.patch("/followups/{followup_id}")
def set_status(followup_id: int, body: FollowupStatusIn):
    if not repo.set_followup_status(followup_id, body.status):
        raise HTTPException(404, "Follow-up not found")
    return {"followup_id": followup_id, "status": body.status}
