"""Scheduler Agent — application follow-up reminders (stored locally, pending approval)."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

from agents.state import AgentState
from backend.repositories import candidates as repo


def parse_due(text: str, today: Optional[date] = None) -> date:
    today = today or date.today()
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text or "")
    if m:
        return date.fromisoformat(m.group(1))
    m = re.search(r"\bin\s+(\d{1,3})\s*(day|days|week|weeks)\b", text or "", re.I)
    if m:
        n = int(m.group(1)) * (7 if m.group(2).lower().startswith("week") else 1)
        return today + timedelta(days=n)
    if re.search(r"\btomorrow\b", text or "", re.I):
        return today + timedelta(days=1)
    if re.search(r"\bnext week\b", text or "", re.I):
        return today + timedelta(days=7)
    return today + timedelta(days=7)  # documented default


def schedule(candidate_id: str, job_id: Optional[str], when: str, note: str = "") -> dict:
    due = parse_due(when)
    fid = repo.add_followup(candidate_id, job_id, due.isoformat(), note or "Follow up on application")
    return {"followup_id": fid, "due_date": due.isoformat(), "status": "pending_approval",
            "note": "Reminder saved locally. Approve it to activate; no external calendar or email is used."}


def scheduler_agent(state: AgentState) -> AgentState:
    trace = state.get("trace", []) + ["scheduler"]
    cid = state.get("candidate_id")
    if not cid:
        return {**state, "trace": trace, "errors": state.get("errors", []) + ["No candidate"]}
    res = schedule(cid, state.get("job_id"), state.get("message", ""), state.get("params", {}).get("note", ""))
    return {**state, "trace": trace, "requires_approval": True,
            "actions": state.get("actions", []) + [{"type": "followup", **res}],
            "sentences": [{"text": f"I drafted a follow-up reminder for {res['due_date']}; it is pending your approval.",
                           "citations": []}],
            "data": {**state.get("data", {}), "allowed_numbers": [res["due_date"][:4], res["due_date"][5:7],
                                                                  res["due_date"][8:10]]}}
