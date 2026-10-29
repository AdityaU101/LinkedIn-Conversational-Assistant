"""Loader for the DataStax LinkedIn job listings (kept separate from Kaggle postings)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from common.config import get_settings
from common.schemas import Job
from common.text import content_hash
from ingestion.normalize import (
    clean_str, infer_seniority_from_title, infer_work_mode, is_masked, normalize_employment_type, normalize_title,
    parse_date, parse_location,
)

SOURCE = "datastax_linkedin"


def default_path() -> Path:
    return get_settings().path("raw") / "datastax_linkedin" / "postings.csv"


class DataStaxLoader:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or default_path()

    def iter_jobs(self, limit: Optional[int] = None) -> Iterator[Job]:
        df = pd.read_csv(self.path, low_memory=False)
        n = 0
        for r in df.to_dict("records"):
            title = clean_str(r.get("title"))
            if not title:
                continue
            # ~6% of rows have title/company/location redacted with asterisks in the source; a redacted value is
            # unknown, never a value. Masked titles are kept (row order and dedup clusters stay stable) but flagged
            # so the pipeline does not index them.
            title_masked = is_masked(title)
            company = None if is_masked(r.get("company")) else clean_str(r.get("company"))
            raw_loc = None if is_masked(r.get("location")) else r.get("location")
            loc = parse_location(raw_loc)
            desc = clean_str(r.get("description")) and str(r["description"]).strip()
            sid = str(r["id"])
            yield Job(
                job_id=f"{SOURCE}:{sid}",
                source_job_id=sid,
                source=SOURCE,
                title=normalize_title(title),
                company=company,
                description=desc or None,
                location=loc["location"], location_city=loc["city"], location_state=loc["state"],
                location_country=loc["country"],
                work_mode=infer_work_mode(clean_str(r.get("work_type")), None if title_masked else title, loc["location"]),
                employment_type=normalize_employment_type(r.get("employment_type")),
                seniority=None,
                posting_date=parse_date(r.get("date_posted")),
                source_url=clean_str(r.get("link")),
                attribution=f"{clean_str(r.get('source')) or 'LinkedIn'} via DataStax dataset",
                is_historical=True,
                raw_source_reference=f"Data/raw/datastax_linkedin/postings.csv#id={sid}",
                content_hash=content_hash(desc or title),
                raw={"title": title, "location": clean_str(r.get("location")), "source": clean_str(r.get("source")),
                     "work_type": clean_str(r.get("work_type")), "employment_type": clean_str(r.get("employment_type")),
                     "seniority_inferred_from_title": None if title_masked else infer_seniority_from_title(title),
                     "masked_fields": [f for f in ("title", "company", "location") if is_masked(r.get(f))]},
            )
            n += 1
            if limit and n >= limit:
                return
