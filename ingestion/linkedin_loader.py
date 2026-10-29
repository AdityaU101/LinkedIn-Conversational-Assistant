"""Loader for the Kaggle LinkedIn Job Postings 2023–2024 dataset.

Reads ``postings.csv`` (~124k rows, ~500 MB) in chunks and joins side tables
(companies, industries, job-function categories) by their real IDs.
"""
from __future__ import annotations

import zlib
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from common.config import get_settings
from common.schemas import Job
from common.text import content_hash
from ingestion.normalize import (
    annualize_salary, clean_str, epoch_ms_to_date, infer_seniority_from_title, infer_work_mode,
    is_missing, normalize_employment_type, normalize_seniority, normalize_title, parse_location,
)

SOURCE = "linkedin_kaggle"
TOTAL_ROWS = 123_849  # from Data/manifests/dataset_manifest.json


def base_dir() -> Path:
    return get_settings().path("raw") / "linkedin_jobs"


def _stable_fraction(key: str) -> float:
    return (zlib.crc32(key.encode()) & 0xFFFFFFFF) / 0xFFFFFFFF


class LinkedInLoader:
    def __init__(self, base: Optional[Path] = None, chunk_size: Optional[int] = None):
        self.base = base or base_dir()
        self.chunk_size = chunk_size or get_settings().get("ingestion.chunk_size", 20000)
        self._side_loaded = False

    def _load_side_tables(self) -> None:
        if self._side_loaded:
            return
        b = self.base
        industries = pd.read_csv(b / "mappings" / "industries.csv").set_index("industry_id")["industry_name"].to_dict()
        functions = pd.read_csv(b / "mappings" / "skills.csv").set_index("skill_abr")["skill_name"].to_dict()
        self.job_industries: dict[int, list[str]] = defaultdict(list)
        for jid, iid in pd.read_csv(b / "jobs" / "job_industries.csv").itertuples(index=False):
            name = industries.get(iid)
            if isinstance(name, str):
                self.job_industries[jid].append(name)
        self.job_functions: dict[int, list[str]] = defaultdict(list)
        for jid, abr in pd.read_csv(b / "jobs" / "job_skills.csv").itertuples(index=False):
            if abr in functions:
                self.job_functions[jid].append(functions[abr])
        comp = pd.read_csv(b / "companies" / "companies.csv", usecols=["company_id", "name", "url", "company_size"])
        self.companies = {int(r.company_id): {"name": r.name, "url": r.url, "size": r.company_size}
                          for r in comp.itertuples(index=False)}
        self._side_loaded = True

    def row_to_job(self, r: dict) -> Optional[Job]:
        title = clean_str(r.get("title"))
        if not title:
            return None
        jid = int(r["job_id"])
        loc = parse_location(r.get("location"))
        period = clean_str(r.get("pay_period"))
        smin = annualize_salary(r.get("min_salary"), period)
        smax = annualize_salary(r.get("max_salary"), period)
        if smin is None and smax is None and not is_missing(r.get("med_salary")):
            smin = smax = annualize_salary(r.get("med_salary"), period)
        company_id = None if is_missing(r.get("company_id")) else str(int(r["company_id"]))
        company = clean_str(r.get("company_name"))
        if company is None and company_id and int(company_id) in self.companies:
            company = clean_str(self.companies[int(company_id)]["name"])
        desc = clean_str(r.get("description")) and str(r.get("description")).strip()
        seniority = normalize_seniority(r.get("formatted_experience_level"))
        return Job(
            job_id=f"{SOURCE}:{jid}",
            source_job_id=str(jid),
            source=SOURCE,
            title=normalize_title(title),
            company=company,
            company_id=company_id,
            description=desc or None,
            location=loc["location"],
            location_city=loc["city"], location_state=loc["state"], location_country=loc["country"],
            work_mode=infer_work_mode(title, loc["location"], remote_flag=r.get("remote_allowed")),
            employment_type=normalize_employment_type(r.get("formatted_work_type")),
            seniority=seniority,
            salary_min=smin, salary_max=smax,
            salary_currency=clean_str(r.get("currency")) if (smin or smax) else None,
            salary_period=period,
            industry=self.job_industries.get(jid, []),
            skills=[],
            posting_date=epoch_ms_to_date(r.get("listed_time")) or epoch_ms_to_date(r.get("original_listed_time")),
            source_url=clean_str(r.get("job_posting_url")),
            attribution="LinkedIn (Kaggle dataset arshkon/linkedin-job-postings)",
            is_historical=True,
            raw_source_reference=f"Data/raw/linkedin_jobs/postings.csv#job_id={jid}",
            content_hash=content_hash(desc or title),
            raw={
                "title": title,
                "location": clean_str(r.get("location")),
                "formatted_experience_level": clean_str(r.get("formatted_experience_level")),
                "formatted_work_type": clean_str(r.get("formatted_work_type")),
                "remote_allowed": None if is_missing(r.get("remote_allowed")) else float(r["remote_allowed"]),
                "min_salary": None if is_missing(r.get("min_salary")) else float(r["min_salary"]),
                "max_salary": None if is_missing(r.get("max_salary")) else float(r["max_salary"]),
                "med_salary": None if is_missing(r.get("med_salary")) else float(r["med_salary"]),
                "linkedin_job_functions": self.job_functions.get(jid, []),
                "skills_desc": clean_str(r.get("skills_desc")),
                "seniority_inferred_from_title": None if seniority else infer_seniority_from_title(title),
            },
        )

    def iter_jobs(self, limit: Optional[int] = None) -> Iterator[Job]:
        """Yield canonical jobs. With ``limit``, a deterministic hash-based sample
        spread across the whole file is returned (not just the first rows)."""
        self._load_side_tables()
        frac = 1.0 if not limit else min(1.0, 1.15 * limit / TOTAL_ROWS)
        n = 0
        for chunk in pd.read_csv(self.base / "postings.csv", chunksize=self.chunk_size, low_memory=False):
            for r in chunk.to_dict("records"):
                if frac < 1.0 and _stable_fraction(str(r["job_id"])) > frac:
                    continue
                job = self.row_to_job(r)
                if job is None:
                    continue
                yield job
                n += 1
                if limit and n >= limit:
                    return
