"""Adzuna Jobs API connector.

Credentials come only from ADZUNA_APP_ID / ADZUNA_APP_KEY. They are never
logged, and saved raw responses have them stripped from the request URL.
Without credentials the connector reports ``available() == False`` and the
rest of the application keeps working.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import httpx

from common.config import env, get_settings
from common.schemas import Job
from common.text import content_hash
from ingestion.normalize import (
    clean_str, infer_seniority_from_title, infer_work_mode, normalize_employment_type, normalize_title,
    parse_date, parse_location,
)

log = logging.getLogger(__name__)
SOURCE = "adzuna"
BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"


def raw_dir() -> Path:
    return get_settings().path("raw") / "adzuna"


def available() -> bool:
    return bool(env("ADZUNA_APP_ID") and env("ADZUNA_APP_KEY"))


def fetch(what: str = "", where: str = "", country: str = "us", pages: int = 1, results_per_page: int = 50,
          client: Optional[httpx.Client] = None, out_dir: Optional[Path] = None) -> list[Path]:
    if not available():
        raise RuntimeError("Adzuna credentials not configured (ADZUNA_APP_ID / ADZUNA_APP_KEY)")
    out_dir = out_dir or raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(timeout=30)
    saved: list[Path] = []
    try:
        for page in range(1, pages + 1):
            params = {"app_id": env("ADZUNA_APP_ID"), "app_key": env("ADZUNA_APP_KEY"),
                      "results_per_page": results_per_page, "content-type": "application/json"}
            if what:
                params["what"] = what
            if where:
                params["where"] = where
            resp = client.get(BASE_URL.format(country=country, page=page), params=params)
            if resp.status_code != 200:
                # never echo the URL: it contains the key
                raise RuntimeError(f"Adzuna request failed with HTTP {resp.status_code}")
            payload = resp.json()
            payload["_request"] = {"country": country, "page": page, "what": what, "where": where,
                                   "fetched_at": datetime.now(timezone.utc).isoformat()}
            path = out_dir / f"adzuna_{country}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{page}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            saved.append(path)
            if len(payload.get("results", [])) < results_per_page:
                break
    finally:
        if own:
            client.close()
    return saved


def _location(loc: dict, country: str) -> dict:
    """Prefer Adzuna's structured area list [country, state, (county/city...), locality]."""
    from ingestion.normalize import US_STATES, _STATE_NAMES

    area = [a for a in (loc.get("area") or []) if a]
    out = parse_location(loc.get("display_name"))
    if area:
        out["country"] = "US" if area[0] in ("US", "USA", "United States") else area[0]
        if len(area) >= 2:
            st = area[1]
            code = "DC" if st.startswith("Washington, D.C") else _STATE_NAMES.get(st.lower())
            out["state"] = code if out["country"] == "US" and code else (None if out["country"] == "US" else st)
        out["city"] = area[-1] if len(area) >= 3 else None
        if out["state"] and out["state"] not in US_STATES and out["country"] == "US":
            out["state"] = None
    return out


CURRENCY = {"us": "USD", "gb": "GBP", "ca": "CAD", "au": "AUD", "de": "EUR", "fr": "EUR", "in": "INR", "nl": "EUR"}


def record_to_job(r: dict, file_name: str, country: str = "us") -> Optional[Job]:
    title = clean_str(r.get("title"))
    if not title or "id" not in r:
        return None
    sid = str(r["id"])
    loc = _location(r.get("location") or {}, country)
    desc = clean_str(r.get("description"))
    predicted = str(r.get("salary_is_predicted", "0")) == "1"
    # Adzuna-predicted salaries are model estimates, not posted values: keep them raw only.
    smin = None if predicted else r.get("salary_min")
    smax = None if predicted else r.get("salary_max")
    contract_time = clean_str(r.get("contract_time"))
    contract_type = clean_str(r.get("contract_type"))
    return Job(
        job_id=f"{SOURCE}:{sid}",
        source_job_id=sid,
        source=SOURCE,
        title=normalize_title(title),
        company=clean_str((r.get("company") or {}).get("display_name")),
        description=desc,
        location=loc["location"], location_city=loc["city"], location_state=loc["state"],
        location_country=loc["country"],
        work_mode=infer_work_mode(title, loc["location"]),
        employment_type=normalize_employment_type("contract" if contract_type == "contract" else contract_time),
        salary_min=float(smin) if smin else None, salary_max=float(smax) if smax else None,
        salary_currency=CURRENCY.get(country) if (smin or smax) else None, salary_period="YEARLY" if (smin or smax) else None,
        industry=[clean_str((r.get("category") or {}).get("label"))] if (r.get("category") or {}).get("label") else [],
        posting_date=parse_date(r.get("created")),
        source_url=clean_str(r.get("redirect_url")),
        attribution="Adzuna (adzuna.com)",
        is_historical=False,
        raw_source_reference=f"Data/raw/adzuna/{file_name}#id={sid}",
        content_hash=content_hash(desc or title),
        raw={"title": title, "salary_min": r.get("salary_min"), "salary_max": r.get("salary_max"),
             "salary_is_predicted": r.get("salary_is_predicted"), "contract_time": contract_time,
             "contract_type": contract_type, "seniority_inferred_from_title": infer_seniority_from_title(title),
             "note": "Adzuna descriptions are truncated snippets"},
    )


class AdzunaLoader:
    def __init__(self, directory: Optional[Path] = None):
        self.directory = directory or raw_dir()

    def iter_jobs(self, limit: Optional[int] = None) -> Iterator[Job]:
        seen: dict[str, Job] = {}
        for path in sorted(self.directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            country = (payload.get("_request") or {}).get("country", "us")
            for r in payload.get("results", []):
                job = record_to_job(r, path.name, country)
                if job:
                    seen[job.source_job_id] = job
        for i, job in enumerate(seen.values()):
            if limit and i >= limit:
                return
            yield job
