"""API request/response models."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from common.schemas import Candidate


class Page(BaseModel):
    items: list[Any]
    total: int
    offset: int
    limit: int


class FiltersIn(BaseModel):
    location: Optional[str] = Field(None, max_length=100)
    work_modes: list[Literal["remote", "hybrid", "onsite"]] = Field(default_factory=list)
    seniorities: list[Literal["internship", "entry", "associate", "mid_senior", "director", "executive"]] = Field(default_factory=list)
    employment_types: list[str] = Field(default_factory=list, max_length=10)
    sources: list[str] = Field(default_factory=list, max_length=10)
    min_salary: Optional[float] = Field(None, ge=0, le=10_000_000)
    title_contains: Optional[str] = Field(None, max_length=100)
    required_skill_ids: list[str] = Field(default_factory=list, max_length=20)
    posted_after: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    max_required_years: Optional[float] = Field(None, ge=0, le=50)


class JobSearchIn(BaseModel):
    query: str = Field("", max_length=500)
    filters: FiltersIn = Field(default_factory=FiltersIn)
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class RecommendIn(BaseModel):
    candidate_id: str = Field(..., max_length=80)
    filters: FiltersIn = Field(default_factory=FiltersIn)
    method: Literal["bm25", "tfidf", "jaccard", "dense", "rrf", "ltr", "final"] = "final"
    k: int = Field(10, ge=1, le=50)
    explain_with_llm: bool = False
    use_cross_encoder: bool = True


class JobCandidatesIn(BaseModel):
    job_id: str = Field(..., max_length=120)
    k: int = Field(10, ge=1, le=50)
    source: Optional[str] = None


class ProfileIn(BaseModel):
    profile: Candidate


class ChatIn(BaseModel):
    candidate_id: Optional[str] = Field(None, max_length=80)
    message: str = Field(..., min_length=1, max_length=2000)
    job_id: Optional[str] = Field(None, max_length=120)


class OptimizeIn(BaseModel):
    candidate_id: str = Field(..., max_length=80)
    rewrite_bullet: Optional[str] = Field(None, max_length=1500)


class OutreachIn(BaseModel):
    candidate_id: str = Field(..., max_length=80)
    job_id: str = Field(..., max_length=120)


class FollowupIn(BaseModel):
    candidate_id: str = Field(..., max_length=80)
    job_id: Optional[str] = Field(None, max_length=120)
    when: str = Field("in 7 days", max_length=100)
    note: str = Field("", max_length=500)


class FollowupStatusIn(BaseModel):
    status: Literal["approved", "done", "cancelled"]


class ExplainIn(BaseModel):
    candidate_id: str
    job_id: str
    method: Literal["auto", "template", "llm"] = "auto"


class IngestionIn(BaseModel):
    full: bool = False
    fetch_live: bool = False
    adzuna_query: str = Field("", max_length=100)
    rebuild_indexes: bool = True
    rebuild_graph: bool = True


class ImportSampleIn(BaseModel):
    source: Literal["kaggle_resumes", "resume_corpus"] = "kaggle_resumes"
    limit: int = Field(50, ge=1, le=500)
