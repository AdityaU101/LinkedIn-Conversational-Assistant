"""Job parsing orchestrator with a content-hash cache.

Bulk indexing uses the classical parser (fast, deterministic). The LLM parser
can be applied per job or for experiments (Experiment A).
"""
from __future__ import annotations

import json
from typing import Iterable, Optional

from common.config import get_settings
from common.schemas import Job, OccupationMention, Requirement
from common.text import sha1
from parsing.classical_parser import ClassicalJobParser, VERSION as CLASSICAL_VERSION

_parser: Optional[ClassicalJobParser] = None


def classical_job_parser() -> ClassicalJobParser:
    global _parser
    if _parser is None:
        _parser = ClassicalJobParser()
    return _parser


def _cache_path(job: Job, extractor: str):
    d = get_settings().path("cache") / "job_parses" / extractor
    d.mkdir(parents=True, exist_ok=True)
    key = sha1(f"{extractor}|{CLASSICAL_VERSION}|{job.job_id}|{job.title}|{job.content_hash}|{','.join(job.skills)}")
    return d / f"{key[:2]}" / f"{key}.json"


def parse_job(job: Job, use_llm: bool = False, use_cache: bool = True) -> Job:
    if use_llm:
        from parsing.llm_parser import LLMParser

        p = LLMParser()
        if p.available():
            parsed, _ = p.parse_job(job)
            return parsed
    path = _cache_path(job, "classical")
    if use_cache and path.exists():
        data = json.loads(path.read_text())
        return job.model_copy(update={
            "requirements": [Requirement.model_validate(r) for r in data["requirements"]],
            "occupation": OccupationMention.model_validate(data["occupation"]) if data.get("occupation") else None})
    parsed = classical_job_parser().parse(job)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"requirements": [r.model_dump() for r in parsed.requirements],
                                    "occupation": parsed.occupation.model_dump() if parsed.occupation else None}))
    return parsed


def parse_jobs(jobs: Iterable[Job], use_cache: bool = True, progress=None) -> list[Job]:
    out = []
    for i, j in enumerate(jobs):
        out.append(parse_job(j, use_cache=use_cache))
        if progress and i % 500 == 0:
            progress(f"parsed {i} jobs")
    return out
