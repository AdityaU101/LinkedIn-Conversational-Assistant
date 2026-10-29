"""Job store backed by Data/processed/jobs_indexed.parquet.

Jobs are kept as a DataFrame and materialised into ``Job`` objects lazily
(LRU), so full mode does not need every Job object in memory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from common.config import get_settings
from common.schemas import EDUCATION_ORDER, Job
from ingestion.pipeline import JSON_COLS, jobs_to_frame, row_to_job


def indexed_path() -> Path:
    return get_settings().path("processed") / "jobs_indexed.parquet"


@dataclass
class JobFilters:
    location: Optional[str] = None
    work_modes: list[str] = field(default_factory=list)
    seniorities: list[str] = field(default_factory=list)
    employment_types: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    min_salary: Optional[float] = None
    title_contains: Optional[str] = None
    required_skill_ids: list[str] = field(default_factory=list)
    posted_after: Optional[str] = None
    max_required_years: Optional[float] = None

    def is_empty(self) -> bool:
        return not any([self.location, self.work_modes, self.seniorities, self.employment_types, self.sources,
                        self.min_salary, self.title_contains, self.required_skill_ids, self.posted_after,
                        self.max_required_years is not None])


class JobStore:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.ids: list[str] = self.df["job_id"].tolist()
        self.pos: dict[str, int] = {j: i for i, j in enumerate(self.ids)}
        self.skill_sets: list[frozenset[str]] = []
        self.required_sets: list[frozenset[str]] = []
        self.preferred_sets: list[frozenset[str]] = []
        self.required_years: list[Optional[float]] = []
        self.occ_codes: list[Optional[str]] = []
        self.edu_levels: list[Optional[int]] = []
        for occ in self.df["occupation"]:
            o = json.loads(occ) if isinstance(occ, str) else occ
            self.occ_codes.append(o.get("code") if isinstance(o, dict) else None)
        for reqs in self.df["requirements"]:
            rs = json.loads(reqs) if isinstance(reqs, str) else (reqs or [])
            req = {r["skill"]["canonical_id"] for r in rs if r.get("skill") and r["skill"].get("canonical_id")
                   and r["importance"] == "required"}
            pref = {r["skill"]["canonical_id"] for r in rs if r.get("skill") and r["skill"].get("canonical_id")
                    and r["importance"] == "preferred"} - req
            years = [r["years"] for r in rs if r["kind"] == "experience" and r.get("years") is not None]
            self.required_sets.append(frozenset(req))
            self.preferred_sets.append(frozenset(pref))
            self.skill_sets.append(frozenset(req | pref))
            self.required_years.append(min(years) if years else None)
            levels = [EDUCATION_ORDER[r["education_level"]] for r in rs if r["kind"] == "education"
                      and r["importance"] == "required" and r.get("education_level") in EDUCATION_ORDER]
            self.edu_levels.append(min(levels) if levels else None)
        self.index_texts: Optional[list[str]] = None

    def __len__(self) -> int:
        return len(self.ids)

    @lru_cache(maxsize=4096)
    def job(self, pos: int) -> Job:
        return row_to_job(self.df.iloc[pos].to_dict())

    def by_id(self, job_id: str) -> Optional[Job]:
        p = self.pos.get(job_id)
        return self.job(p) if p is not None else None

    def texts(self) -> list[str]:
        if self.index_texts is None:
            self.index_texts = self.df["index_text"].fillna("").tolist()
        return self.index_texts

    def mask(self, f: Optional[JobFilters]) -> np.ndarray:
        df = self.df
        m = np.ones(len(df), dtype=bool)
        if f is None or f.is_empty():
            return m
        if f.location:
            loc = f.location.lower().strip()
            hay = (df["location"].fillna("") + " " + df["location_state"].fillna("") + " "
                   + df["location_country"].fillna("")).str.lower()
            m &= hay.str.contains(loc, regex=False).to_numpy()
        if f.work_modes:
            m &= df["work_mode"].isin(f.work_modes).to_numpy()
        if f.seniorities:
            m &= df["seniority"].isin(f.seniorities).to_numpy()
        if f.employment_types:
            m &= df["employment_type"].isin(f.employment_types).to_numpy()
        if f.sources:
            m &= df["source"].isin(f.sources).to_numpy()
        if f.min_salary is not None:
            sal = df["salary_max"].fillna(df["salary_min"])
            m &= (sal >= f.min_salary).fillna(False).to_numpy()
        if f.title_contains:
            m &= df["title"].fillna("").str.lower().str.contains(f.title_contains.lower(), regex=False).to_numpy()
        if f.posted_after:
            m &= (df["posting_date"].fillna("") >= f.posted_after).to_numpy()
        if f.required_skill_ids:
            need = set(f.required_skill_ids)
            m &= np.array([need <= s for s in self.skill_sets])
        if f.max_required_years is not None:
            m &= np.array([(y is None or y <= f.max_required_years) for y in self.required_years])
        return m

    def source_counts(self) -> dict[str, int]:
        return self.df["source"].value_counts().to_dict()


def save_store(jobs: list[Job], path: Optional[Path] = None) -> Path:
    path = path or indexed_path()
    df = jobs_to_frame(jobs)
    df["index_text"] = [j.index_text() for j in jobs]
    df.to_parquet(path, index=False)
    return path


@lru_cache(maxsize=1)
def get_store() -> JobStore:
    p = indexed_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run: python scripts/build_indexes.py")
    return JobStore(pd.read_parquet(p))


__all__ = ["JobStore", "JobFilters", "get_store", "save_store", "JSON_COLS"]
