"""Loader for linkedin_job_posts_insights.xlsx (metadata only — no descriptions)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from common.config import get_settings
from common.schemas import Job
from common.text import content_hash, normalize_key
from ingestion.normalize import (
    clean_str, infer_work_mode, normalize_employment_type, normalize_seniority, normalize_title, parse_date,
    parse_location,
)

SOURCE = "linkedin_insights"


def default_path() -> Path:
    return get_settings().path("raw") / "linkedin_job_insights" / "linkedin_job_posts_insights.xlsx"


class InsightsLoader:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or default_path()

    def load_frame(self) -> pd.DataFrame:
        xls = pd.ExcelFile(self.path)
        frames = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            df["__sheet"] = sheet
            df["__row"] = range(2, len(df) + 2)  # spreadsheet row number (header is row 1)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def iter_jobs(self, limit: Optional[int] = None, drop_exact_duplicates: bool = True) -> Iterator[Job]:
        df = self.load_frame()
        data_cols = [c for c in df.columns if not c.startswith("__")]
        if drop_exact_duplicates:
            df = df.drop_duplicates(subset=data_cols)
        n = 0
        for r in df.to_dict("records"):
            title = clean_str(r.get("job_title"))
            if not title:
                continue
            loc = parse_location(r.get("location"))
            sid = f"{r['__sheet']}-{r['__row']}"
            ident = "|".join(normalize_key(str(r.get(c))) for c in data_cols)
            yield Job(
                job_id=f"{SOURCE}:{sid}",
                source_job_id=sid,
                source=SOURCE,
                title=normalize_title(title),
                company=clean_str(r.get("company_name")),
                description=None,
                location=loc["location"], location_city=loc["city"], location_state=loc["state"],
                location_country=loc["country"],
                work_mode=infer_work_mode(title, loc["location"]),
                employment_type=normalize_employment_type(r.get("employment_type")),
                seniority=normalize_seniority(r.get("seniority_level")),
                industry=[clean_str(r["industry"])] if clean_str(r.get("industry")) else [],
                posting_date=parse_date(r.get("date")),
                attribution="LinkedIn Job Posts Insights dataset",
                is_historical=True,
                raw_source_reference=f"Data/raw/linkedin_job_insights/linkedin_job_posts_insights.xlsx#{sid}",
                content_hash=content_hash(ident),
                raw={"title": title, "hiring_status": clean_str(r.get("hiring_status")),
                     "job_function": clean_str(r.get("job_function")),
                     "seniority_level": clean_str(r.get("seniority_level")),
                     "employment_type": clean_str(r.get("employment_type"))},
            )
            n += 1
            if limit and n >= limit:
                return
