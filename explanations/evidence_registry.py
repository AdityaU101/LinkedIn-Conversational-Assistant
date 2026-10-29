"""Numbered evidence items for one candidate–job pair.

Explanations may cite only these IDs ([C1] candidate resume span, [J1] job
posting span, [G1] graph/taxonomy relation). The verifier checks every claim
against the cited items.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from matching.skill_gap import SkillGap


class EvidenceItem(BaseModel):
    cite_id: str
    kind: Literal["candidate", "job", "graph"]
    text: str
    skill_ids: list[str] = Field(default_factory=list)
    skill_names: list[str] = Field(default_factory=list)
    evidence_id: Optional[str] = None          # Evidence node id (resume/job spans)
    start: Optional[int] = None
    end: Optional[int] = None
    meta: dict = Field(default_factory=dict)


class EvidenceRegistry(BaseModel):
    candidate_id: str
    job_id: str
    items: list[EvidenceItem] = Field(default_factory=list)

    def get(self, cite_id: str) -> Optional[EvidenceItem]:
        return next((i for i in self.items if i.cite_id == cite_id), None)

    def ids(self) -> set[str]:
        return {i.cite_id for i in self.items}


def build_registry(gap: SkillGap, max_per_skill: int = 1) -> EvidenceRegistry:
    reg = EvidenceRegistry(candidate_id=gap.candidate_id, job_id=gap.job_id)
    counters = {"candidate": 0, "job": 0, "graph": 0}
    seen: dict[str, str] = {}

    def add(kind, text, skill_id, skill_name, ev=None, meta=None) -> str:
        key = f"{kind}|{ev.evidence_id if ev else text}"
        if key in seen:
            item = reg.get(seen[key])
            if skill_id and skill_id not in item.skill_ids:
                item.skill_ids.append(skill_id)
                item.skill_names.append(skill_name)
            return seen[key]
        counters[kind] += 1
        cid = {"candidate": "C", "job": "J", "graph": "G"}[kind] + str(counters[kind])
        reg.items.append(EvidenceItem(cite_id=cid, kind=kind, text=text[:400], skill_ids=[skill_id] if skill_id else [],
                                      skill_names=[skill_name] if skill_name else [],
                                      evidence_id=ev.evidence_id if ev else None,
                                      start=ev.start_offset if ev else None, end=ev.end_offset if ev else None,
                                      meta=meta or {}))
        seen[key] = cid
        return cid

    for m in gap.matched_required + gap.matched_preferred:
        for ev in m.candidate_evidence[:max_per_skill]:
            add("candidate", ev.source_text, m.skill_id, m.name, ev)
        if not m.candidate_evidence:  # skill added by the user in the profile editor
            add("candidate", f"{m.name} (added by the user in the profile editor; not found in the resume text)",
                m.skill_id, m.name, meta={"user_assertion": True})
        for ev in m.job_evidence[:max_per_skill]:
            add("job", ev.source_text, m.skill_id, m.name, ev, {"importance": m.importance})
    for ms in gap.missing_required + gap.missing_preferred:
        for ev in ms.job_evidence[:max_per_skill]:
            add("job", ev.source_text, ms.skill_id, ms.name, ev, {"importance": ms.importance, "missing": True})
        for b in ms.bridges[:1]:
            add("graph", f"{b.candidate_skill_name} is related to {ms.name} (inferred: {b.method}, "
                         f"confidence {b.confidence:.2f})", ms.skill_id, ms.name,
                meta={"related_skill_id": b.candidate_skill_id, "related_skill_name": b.candidate_skill_name,
                      "inferred": True, "method": b.method, "confidence": b.confidence})
    return reg
