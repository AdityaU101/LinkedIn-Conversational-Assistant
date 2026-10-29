"""KG-Build Agent: normalise and upsert the candidate into the knowledge graph."""
from __future__ import annotations

from agents.state import AgentState
from common.schemas import Candidate


def kg_builder_agent(state: AgentState) -> AgentState:
    trace = state.get("trace", []) + ["kg_builder"]
    if not state.get("candidate") or not state.get("data", {}).get("new_profile"):
        return {**state, "trace": trace}
    from backend.services.resume_service import sync_graph

    status = sync_graph(Candidate.model_validate(state["candidate"]))
    return {**state, "trace": trace, "data": {**state.get("data", {}), "graph": status}}
