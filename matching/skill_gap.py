"""Skill-gap analysis between a candidate and a job.

Matching is by canonical skill ID only. Related skills are reported
separately as *transferable* bridges (taxonomy RELATED_TO edges, inferred,
with method and confidence) — they are never counted as matches.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from common.schemas import Candidate, Evidence, Job
from normalization.taxonomy import Taxonomy, load_taxonomy


class SkillMatch(BaseModel):
    skill_id: str
    name: str
    importance: str
    candidate_evidence: list[Evidence] = Field(default_factory=list)
    job_evidence: list[Evidence] = Field(default_factory=list)


class Bridge(BaseModel):
    candidate_skill_id: str
    candidate_skill_name: str
    confidence: float
    method: str
    inferred: bool = True


class MissingSkill(BaseModel):
    skill_id: str
    name: str
    importance: str
    job_evidence: list[Evidence] = Field(default_factory=list)
    bridges: list[Bridge] = Field(default_factory=list)


class RelatedOccupation(BaseModel):
    code: str
    title: str
    tier: Optional[str] = None
    via: Optional[str] = None


class OccupationTransfer(BaseModel):
    skill_id: str
    name: str
    candidate_importance: float
    target_importance: float
    source: str = "O*NET importance ratings (IM scale)"


class SkillGap(BaseModel):
    candidate_id: str
    job_id: str
    job_title: Optional[str] = None
    matched_required: list[SkillMatch] = Field(default_factory=list)
    matched_preferred: list[SkillMatch] = Field(default_factory=list)
    missing_required: list[MissingSkill] = Field(default_factory=list)
    missing_preferred: list[MissingSkill] = Field(default_factory=list)
    transferable: list[MissingSkill] = Field(default_factory=list)   # missing skills with ≥1 bridge
    occupation_transferable: list[OccupationTransfer] = Field(default_factory=list)
    coverage_required: Optional[float] = None
    coverage_all: Optional[float] = None
    candidate_occupations: list[str] = Field(default_factory=list)
    job_occupation: Optional[str] = None
    experience_required_years: Optional[float] = None
    candidate_experience_years: Optional[float] = None


def candidate_skill_evidence(c: Candidate) -> dict[str, list[Evidence]]:
    ev: dict[str, list[Evidence]] = {}
    for s in c.skills:
        if s.canonical_id:
            ev.setdefault(s.canonical_id, []).extend(s.evidence)
    for group in (c.experiences, c.projects):
        for item in group:
            for s in item.skills:
                if s.canonical_id:
                    lst = ev.setdefault(s.canonical_id, [])
                    have = {e.evidence_id for e in lst}
                    lst.extend(e for e in s.evidence if e.evidence_id not in have)
    return ev


def related_occupations(codes: list[str], tax: Optional[Taxonomy] = None, limit: int = 10) -> list[RelatedOccupation]:
    tax = tax or load_taxonomy()
    rel = tax.related_occupations
    out, seen = [], set(codes)
    tier_rank = {"Primary-Short": 0, "Primary-Long": 1, "Supplemental": 2}
    rows = rel[rel["code"].isin(codes)].copy()
    rows["tr"] = rows["tier"].map(tier_rank).fillna(3)
    for r in rows.sort_values(["tr", "rank"]).itertuples():
        if r.related_code in seen:
            continue
        seen.add(r.related_code)
        out.append(RelatedOccupation(code=r.related_code, title=tax.occupation_title(r.related_code), tier=r.tier,
                                     via=tax.occupation_title(r.code)))
        if len(out) >= limit:
            break
    return out


def compute_gap(c: Candidate, job: Job, tax: Optional[Taxonomy] = None, max_bridges: int = 3) -> SkillGap:
    tax = tax or load_taxonomy()
    cev = candidate_skill_evidence(c)
    cand_ids = set(cev)
    gap = SkillGap(candidate_id=c.candidate_id, job_id=job.job_id, job_title=job.title,
                   candidate_occupations=[o.code for o in c.occupations if o.code],
                   job_occupation=job.occupation.code if job.occupation else None,
                   candidate_experience_years=c.total_experience_years)
    seen: set[str] = set()
    for r in job.requirements:
        if r.kind == "experience" and r.years is not None and r.importance == "required":
            gap.experience_required_years = min(gap.experience_required_years or 99, r.years)
        if r.kind != "skill" or not r.skill or not r.skill.canonical_id:
            continue
        sid = r.skill.canonical_id
        if sid in seen:
            continue
        seen.add(sid)
        name = tax.skill_name(sid)
        if sid in cand_ids:
            m = SkillMatch(skill_id=sid, name=name, importance=r.importance, candidate_evidence=cev[sid][:3],
                           job_evidence=r.evidence[:3])
            (gap.matched_required if r.importance == "required" else gap.matched_preferred).append(m)
        else:
            bridges = []
            for other, conf in sorted(tax.related_skills.get(sid, []), key=lambda x: -x[1]):
                if other in cand_ids:
                    bridges.append(Bridge(candidate_skill_id=other, candidate_skill_name=tax.skill_name(other),
                                          confidence=conf, method=tax.relation_method.get((sid, other), "related")))
                if len(bridges) >= max_bridges:
                    break
            ms = MissingSkill(skill_id=sid, name=name, importance=r.importance, job_evidence=r.evidence[:3],
                              bridges=bridges)
            (gap.missing_required if r.importance == "required" else gap.missing_preferred).append(ms)
            if bridges:
                gap.transferable.append(ms)
    n_req = len(gap.matched_required) + len(gap.missing_required)
    n_all = n_req + len(gap.matched_preferred) + len(gap.missing_preferred)
    gap.coverage_required = round(len(gap.matched_required) / n_req, 4) if n_req else None
    gap.coverage_all = round((len(gap.matched_required) + len(gap.matched_preferred)) / n_all, 4) if n_all else None
    if gap.job_occupation and gap.candidate_occupations:
        target = {k: v for k, v in tax.skills_by_occ.get(gap.job_occupation, {}).items() if k.startswith("onet:")}
        mine: dict[str, float] = {}
        for code in gap.candidate_occupations:
            for k, v in tax.skills_by_occ.get(code, {}).items():
                if k.startswith("onet:"):
                    mine[k] = max(mine.get(k, 0), v)
        shared = sorted(set(target) & set(mine), key=lambda k: -target[k])
        gap.occupation_transferable = [OccupationTransfer(skill_id=k, name=tax.skill_name(k),
                                                          candidate_importance=round(mine[k] * 5, 2),
                                                          target_importance=round(target[k] * 5, 2))
                                       for k in shared[:8]]
    return gap
