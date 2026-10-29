"""Typed LangGraph state shared by all agents."""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # request
    request: str                       # chat | pipeline | optimize | outreach | followup
    message: str
    candidate_id: Optional[str]
    job_id: Optional[str]
    resume_text: Optional[str]
    filters: dict
    params: dict
    # working data
    intent: str
    candidate: Optional[dict]
    recommendations: Optional[dict]
    gap: Optional[dict]
    registry: Optional[dict]
    draft: Optional[str]
    sentences: list[dict]              # {text, citations}
    suggestions: list[dict]
    # outputs
    answer: str
    body: str            # the answer without its preface (the part meant to be copied)
    citations: list[dict]
    critic: Optional[dict]
    requires_approval: bool
    actions: list[dict]
    injection_flags: list[str]
    errors: list[str]
    trace: list[str]
    data: dict[str, Any]
