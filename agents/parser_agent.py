"""Parser Agent: obtain a structured candidate (parse new text or load stored profile)."""
from __future__ import annotations

import uuid

from agents.state import AgentState
from backend.repositories import candidates as repo
from common.privacy import detect_injection
from parsing.resume_parser import parse_resume_text


def parser_agent(state: AgentState) -> AgentState:
    trace = state.get("trace", []) + ["parser"]
    flags = detect_injection(state.get("message", ""))
    if state.get("resume_text"):
        cid = state.get("candidate_id") or "cand-" + uuid.uuid4().hex[:12]
        res = parse_resume_text(state["resume_text"], cid, "agent_input")
        repo.save_candidate(res.candidate, source="upload")
        return {**state, "candidate_id": cid, "candidate": res.candidate.model_dump(), "trace": trace,
                "injection_flags": flags + res.injection_flags, "data": {**state.get("data", {}), "new_profile": True}}
    cid = state.get("candidate_id")
    cand = repo.get_candidate(cid) if cid else None
    errors = state.get("errors", [])
    if cid and cand is None:
        errors = errors + [f"Unknown candidate {cid}"]
    return {**state, "candidate": cand.model_dump() if cand else None, "trace": trace, "injection_flags": flags,
            "errors": errors}
