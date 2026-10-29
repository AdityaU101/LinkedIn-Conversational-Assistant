"""LangGraph orchestration of the seven agents.

  START -> parser -> kg_builder -> route
     route=chat:      intent -> [matching] -> {compose | optimizer | outreach | scheduler} -> critic -> END
     route=pipeline:  matching -> critic -> END
     route=optimize:  matching -> profile_optimizer -> critic -> END
     route=outreach:  research_outreach -> critic -> END
     route=followup:  scheduler -> critic -> END
External actions (sending messages, calendar writes) are never executed:
outreach and follow-ups end in ``requires_approval=True`` drafts.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from agents.career_assistant import classify, compose
from agents.critic_agent import critic_agent
from agents.kg_builder_agent import kg_builder_agent
from agents.matching_agent import matching_agent
from agents.parser_agent import parser_agent
from agents.profile_optimizer_agent import profile_optimizer_agent
from agents.research_outreach_agent import research_outreach_agent
from agents.scheduler_agent import scheduler_agent
from agents.state import AgentState


def intent_node(state: AgentState) -> AgentState:
    return {**state, "intent": classify(state.get("message", "")), "trace": state.get("trace", []) + ["intent"]}


def _route_request(state: AgentState) -> str:
    req = state.get("request", "chat")
    if state.get("errors") and not state.get("candidate"):
        return "critic"
    return {"chat": "intent", "pipeline": "matching", "optimize": "matching", "outreach": "research_outreach",
            "followup": "scheduler"}.get(req, "intent")


def _after_intent(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "outreach":
        from agents.career_assistant import resolve_job

        return "matching" if not state.get("job_id") else "research_outreach"
    if intent == "followup":
        return "scheduler"
    if intent in ("recommend", "why_match", "missing_skills", "transferable", "improve_resume"):
        return "matching"
    return "compose"


def _after_matching(state: AgentState) -> str:
    if state.get("request") == "pipeline":
        return "compose"
    if state.get("request") == "optimize" or state.get("intent") == "improve_resume":
        return "profile_optimizer"
    if state.get("intent") == "outreach":
        return "pick_job"
    return "compose"


def pick_job(state: AgentState) -> AgentState:
    from agents.career_assistant import resolve_job

    return {**state, "job_id": resolve_job(state.get("message", ""), state), "trace": state.get("trace", []) + ["pick_job"]}


def _after_pick(state: AgentState) -> str:
    return "research_outreach" if state.get("job_id") else "compose"


def _after_optimizer(state: AgentState) -> str:
    return "compose" if state.get("request") == "chat" else "critic"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("parser", parser_agent)
    g.add_node("kg_builder", kg_builder_agent)
    g.add_node("intent", intent_node)
    g.add_node("matching", matching_agent)
    g.add_node("pick_job", pick_job)
    g.add_node("compose", lambda s: compose({**s, "intent": s.get("intent") or ("recommend" if s.get("request") == "pipeline" else None)}))
    g.add_node("profile_optimizer", profile_optimizer_agent)
    g.add_node("research_outreach", research_outreach_agent)
    g.add_node("scheduler", scheduler_agent)
    g.add_node("critic", critic_agent)
    g.add_edge(START, "parser")
    g.add_edge("parser", "kg_builder")
    g.add_conditional_edges("kg_builder", _route_request,
                            {"intent": "intent", "matching": "matching", "research_outreach": "research_outreach",
                             "scheduler": "scheduler", "critic": "critic"})
    g.add_conditional_edges("intent", _after_intent, {"matching": "matching", "research_outreach": "research_outreach",
                                                      "scheduler": "scheduler", "compose": "compose"})
    g.add_conditional_edges("matching", _after_matching, {"compose": "compose", "profile_optimizer": "profile_optimizer",
                                                          "pick_job": "pick_job"})
    g.add_conditional_edges("pick_job", _after_pick, {"research_outreach": "research_outreach", "compose": "compose"})
    g.add_conditional_edges("profile_optimizer", _after_optimizer, {"compose": "compose", "critic": "critic"})
    g.add_edge("compose", "critic")
    g.add_edge("research_outreach", "critic")
    g.add_edge("scheduler", "critic")
    g.add_edge("critic", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_agent_graph():
    return build_graph()


def run_agents(request: str, candidate_id: str | None = None, message: str = "", job_id: str | None = None,
               resume_text: str | None = None, filters: dict | None = None, params: dict | None = None) -> AgentState:
    state: AgentState = {"request": request, "candidate_id": candidate_id, "message": message, "job_id": job_id,
                         "resume_text": resume_text, "filters": filters or {}, "params": params or {}, "trace": [],
                         "errors": [], "actions": [], "data": {}, "requires_approval": False}
    return get_agent_graph().invoke(state)
