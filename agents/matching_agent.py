"""Matching Agent: retrieve and rank jobs (reuses the latest stored run when possible)."""
from __future__ import annotations

from agents.state import AgentState
from backend.repositories import candidates as repo


def matching_agent(state: AgentState) -> AgentState:
    trace = state.get("trace", []) + ["matching"]
    cid = state.get("candidate_id")
    if not cid or not state.get("candidate"):
        return {**state, "trace": trace}
    run = None if state.get("params", {}).get("refresh") else repo.latest_match_run(cid, "final")
    if run is None:
        from backend.services.matching_service import recommend

        run = recommend(cid, state.get("filters") or {}, "final", k=state.get("params", {}).get("k", 10))
    return {**state, "recommendations": run, "trace": trace}
