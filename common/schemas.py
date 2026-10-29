"""Canonical internal representations shared by every component.

Unknown values are always ``None`` — loaders must never invent values.
Every extracted fact carries :class:`Evidence` pointing back to an exact span
of the source document.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from common.text import sha1

WorkMode = Literal["remote", "hybrid", "onsite"]
EmploymentType = Literal["full_time", "part_time", "contract", "temporary", "internship", "volunteer", "other"]
Seniority = Literal["internship", "entry", "associate", "mid_senior", "director", "executive"]
Importance = Literal["required", "preferred"]
RequirementKind = Literal["skill", "experience", "education", "credential", "responsibility", "other"]

SENIORITY_ORDER = {"internship": 0, "entry": 1, "associate": 2, "mid_senior": 3, "director": 4, "executive": 5}
EDUCATION_ORDER = {"high_school": 1, "associate": 2, "bachelor": 3, "master": 4, "doctorate": 5}


class Evidence(BaseModel):
    """A verbatim span of a source document supporting a fact."""

    model_config = ConfigDict(frozen=False)

    evidence_id: str = ""
    source_document_id: str
    source_text: str
    start_offset: int
    end_offset: int
    extractor: str
    extractor_version: str
    confidence: float = Field(ge=0.0, le=1.0)

    def model_post_init(self, __context) -> None:  # stable, content-derived ID
        if not self.evidence_id:
            self.evidence_id = "ev:" + sha1(
                f"{self.source_document_id}|{self.start_offset}|{self.end_offset}|{self.source_text}"
            )[:16]


class SkillMention(BaseModel):
    mention: str
    canonical_id: Optional[str] = None
    canonical_name: Optional[str] = None
    taxonomy: Optional[str] = None
    method: Optional[str] = None
    confidence: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return self.canonical_id or "raw:" + self.mention.lower()


class OccupationMention(BaseModel):
    mention: str
    code: Optional[str] = None           # O*NET-SOC code, e.g. 15-2051.00
    canonical_title: Optional[str] = None
    method: Optional[str] = None
    confidence: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)


class Experience(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    description: Optional[str] = None
    skills: list[SkillMention] = Field(default_factory=list)
    occupation: Optional[OccupationMention] = None
    evidence: list[Evidence] = Field(default_factory=list)


class Education(BaseModel):
    degree: Optional[str] = None
    level: Optional[str] = None          # key of EDUCATION_ORDER
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    evidence: list[Evidence] = Field(default_factory=list)


class Credential(BaseModel):
    name: str
    issuer: Optional[str] = None
    evidence: list[Evidence] = Field(default_factory=list)


class Project(BaseModel):
    name: str
    description: Optional[str] = None
    skills: list[SkillMention] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class Candidate(BaseModel):
    candidate_id: str
    source_document: str
    raw_text: str
    occupations: list[OccupationMention] = Field(default_factory=list)
    skills: list[SkillMention] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    credentials: list[Credential] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    total_experience_years: Optional[float] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    extractor: Optional[str] = None
    user_corrected: bool = False

    def skill_ids(self) -> set[str]:
        return {s.key for s in self.skills}

    def all_evidence(self) -> list[Evidence]:
        out: list[Evidence] = []
        for group in (self.occupations, self.skills, self.experiences, self.education, self.credentials, self.projects):
            for item in group:
                out.extend(item.evidence)
        return out


class Requirement(BaseModel):
    requirement_id: str = ""
    kind: RequirementKind
    importance: Importance = "required"
    text: str
    skill: Optional[SkillMention] = None
    years: Optional[float] = None
    education_level: Optional[str] = None
    evidence: list[Evidence] = Field(default_factory=list)


class Job(BaseModel):
    job_id: str                          # "<source>:<source_job_id>"
    source_job_id: str
    source: str
    title: Optional[str] = None
    company: Optional[str] = None
    company_id: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    location_country: Optional[str] = None
    work_mode: Optional[WorkMode] = None
    employment_type: Optional[EmploymentType] = None
    seniority: Optional[Seniority] = None
    salary_min: Optional[float] = None   # annualised when the pay period is known
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = None
    salary_period: Optional[str] = None  # original pay period
    industry: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)          # source-provided tags only
    requirements: list[Requirement] = Field(default_factory=list)
    occupation: Optional[OccupationMention] = None
    posting_date: Optional[str] = None   # ISO-8601 date
    source_url: Optional[str] = None
    attribution: Optional[str] = None
    is_historical: bool = True
    raw_source_reference: Optional[str] = None
    content_hash: Optional[str] = None
    raw: dict = Field(default_factory=dict)  # preserved raw values for normalised fields

    def index_text(self, max_chars: int = 6000) -> str:
        parts = [self.title or "", self.company or "", " ".join(self.skills)]
        req = " ".join(r.skill.canonical_name or r.skill.mention for r in self.requirements if r.skill)
        parts.append(req)
        parts.append((self.description or "")[:max_chars])
        return "\n".join(p for p in parts if p)

    def requirement_skill_ids(self, importance: Optional[str] = None) -> set[str]:
        return {
            r.skill.key
            for r in self.requirements
            if r.skill and (importance is None or r.importance == importance)
        }
