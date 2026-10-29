"""RemoteOK public feed: fetch (optional) + load.

RemoteOK's API terms ask consumers to link back to the job URL on Remote OK
and mention Remote OK as the source. Every job keeps ``source_url`` and an
``attribution`` string, and the frontend renders both.
"""
from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import httpx

from common.config import get_settings
from common.schemas import Job
from common.text import content_hash, fix_mojibake, strip_html
from ingestion.normalize import clean_str, infer_seniority_from_title, normalize_title, parse_date, parse_location

SOURCE = "remoteok"
API_URL = "https://remoteok.com/api"
ATTRIBUTION = "Remote OK (remoteok.com)"


def raw_dir() -> Path:
    return get_settings().path("raw") / "remoteok"


def fetch(client: Optional[httpx.Client] = None, out_dir: Optional[Path] = None) -> Path:
    """Download the current feed and save it verbatim as a new timestamped file."""
    out_dir = out_dir or raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(timeout=30, headers={"User-Agent": "LinkedInAssistant-research/1.0"})
    try:
        resp = client.get(API_URL)
        resp.raise_for_status()
        data = resp.json()
    finally:
        if own:
            client.close()
    if not isinstance(data, list):
        raise ValueError("Unexpected RemoteOK response shape")
    path = out_dir / f"remoteok_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _tags(v) -> list[str]:
    if isinstance(v, list):
        return [str(t) for t in v]
    if isinstance(v, str) and v.startswith("["):
        try:  # some snapshots store the list as its Python repr
            parsed = ast.literal_eval(v)
            return [str(t) for t in parsed] if isinstance(parsed, (list, tuple)) else []
        except (ValueError, SyntaxError):
            return []
    return []


def record_to_job(rec: dict, file_name: str) -> Optional[Job]:
    if "id" not in rec or not rec.get("position"):
        return None
    title = fix_mojibake(str(rec["position"]))
    desc = strip_html(fix_mojibake(rec.get("description") or ""))
    loc = parse_location(fix_mojibake(rec.get("location") or "")) if rec.get("location") else parse_location(None)
    smin = float(rec["salary_min"]) if str(rec.get("salary_min", "0")) not in {"0", "", "None"} else None
    smax = float(rec["salary_max"]) if str(rec.get("salary_max", "0")) not in {"0", "", "None"} else None
    sid = str(rec["id"])
    return Job(
        job_id=f"{SOURCE}:{sid}",
        source_job_id=sid,
        source=SOURCE,
        title=normalize_title(title),
        company=clean_str(fix_mojibake(rec.get("company") or "")),
        description=desc or None,
        location=loc["location"], location_city=loc["city"], location_state=loc["state"],
        location_country=loc["country"],
        work_mode="remote",  # RemoteOK only lists remote positions
        skills=[fix_mojibake(t) for t in _tags(rec.get("tags"))],
        salary_min=smin, salary_max=smax, salary_currency="USD" if (smin or smax) else None,
        salary_period="YEARLY" if (smin or smax) else None,
        posting_date=parse_date(rec.get("date")),
        source_url=clean_str(rec.get("url")) or clean_str(rec.get("apply_url")),
        attribution=ATTRIBUTION,
        is_historical=False,
        raw_source_reference=f"Data/raw/remoteok/{file_name}#id={sid}",
        content_hash=content_hash(desc or title),
        raw={"title": title, "location": rec.get("location"), "tags": rec.get("tags"),
             "apply_url": rec.get("apply_url"), "slug": rec.get("slug"), "epoch": rec.get("epoch"),
             "seniority_inferred_from_title": infer_seniority_from_title(title)},
    )


class RemoteOKLoader:
    def __init__(self, directory: Optional[Path] = None):
        self.directory = directory or raw_dir()

    def iter_jobs(self, limit: Optional[int] = None) -> Iterator[Job]:
        """Load all saved snapshots; when an ID appears twice the newest snapshot wins."""
        latest: dict[str, Job] = {}
        for path in sorted(self.directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for rec in data if isinstance(data, list) else []:
                if isinstance(rec, dict):
                    job = record_to_job(rec, path.name)
                    if job:
                        latest[job.source_job_id] = job
        for i, job in enumerate(latest.values()):
            if limit and i >= limit:
                return
            yield job
