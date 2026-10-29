"""Ingestion pipeline: sources -> canonical Job -> dedup -> Parquet + splits.

Outputs:
  Data/processed/jobs.parquet           all canonical jobs (+ dedup columns)
  Data/processed/dedup_report.json
  Data/splits/job_splits.csv            cluster-level train/dev/test split
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from common.config import get_settings
from common.schemas import Job
from ingestion import adzuna_loader
from ingestion.adzuna_loader import AdzunaLoader
from ingestion.datastax_loader import DataStaxLoader
from ingestion.deduplicate import assign_splits, check_no_leakage, deduplicate
from ingestion.insights_loader import InsightsLoader
from ingestion.normalize import is_masked
from ingestion.linkedin_loader import LinkedInLoader
from ingestion.remoteok_loader import RemoteOKLoader

log = logging.getLogger(__name__)

JSON_COLS = ["industry", "skills", "requirements", "occupation", "raw"]


def jobs_to_frame(jobs: Iterable[Job]) -> pd.DataFrame:
    rows = []
    for j in jobs:
        d = j.model_dump()
        for c in JSON_COLS:
            d[c] = json.dumps(d[c], ensure_ascii=False, default=str)
        rows.append(d)
    return pd.DataFrame(rows)


def row_to_job(row: dict) -> Job:
    d = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in row.items()}
    for c in JSON_COLS:
        if isinstance(d.get(c), str):
            d[c] = json.loads(d[c])
    allowed = Job.model_fields.keys()
    return Job(**{k: v for k, v in d.items() if k in allowed and v is not None})


def source_loaders() -> dict[str, Callable[[Optional[int]], Iterable[Job]]]:
    return {
        "linkedin_kaggle": lambda lim: LinkedInLoader().iter_jobs(lim),
        "datastax_linkedin": lambda lim: DataStaxLoader().iter_jobs(lim),
        "linkedin_insights": lambda lim: InsightsLoader().iter_jobs(lim),
        "remoteok": lambda lim: RemoteOKLoader().iter_jobs(lim),
        "adzuna": lambda lim: AdzunaLoader().iter_jobs(lim),
    }


def source_limits(max_jobs: Optional[int]) -> dict[str, Optional[int]]:
    """Per-source caps in development mode; None everywhere in full mode."""
    if max_jobs is None:
        return {s: None for s in source_loaders()}
    quota = get_settings().get("ingestion.source_quota", {})
    limits = {s: max(1, int(max_jobs * q)) for s, q in quota.items()}
    limits.setdefault("adzuna", max_jobs)
    limits.setdefault("linkedin_insights", max_jobs)
    return limits


def run(max_jobs: Optional[int] = None, sources: Optional[list[str]] = None, fetch_live: bool = False,
        adzuna_query: str = "", use_minhash: bool = True, out_dir: Optional[Path] = None,
        progress: Callable[[str], None] = print) -> dict:
    settings = get_settings()
    out_dir = out_dir or settings.path("processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = settings.path("splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    status: dict = {"sources": {}, "live_fetch": {}}

    if fetch_live:
        from ingestion import remoteok_loader

        try:
            status["live_fetch"]["remoteok"] = str(remoteok_loader.fetch())
        except Exception as e:  # network failures must not break ingestion
            status["live_fetch"]["remoteok"] = f"failed: {type(e).__name__}"
        if adzuna_loader.available():
            try:
                status["live_fetch"]["adzuna"] = [str(p) for p in adzuna_loader.fetch(what=adzuna_query, pages=2)]
            except Exception as e:
                status["live_fetch"]["adzuna"] = f"failed: {e}"
        else:
            status["live_fetch"]["adzuna"] = "skipped: credentials not configured"

    limits = source_limits(max_jobs)
    frames = []
    for name, loader in source_loaders().items():
        if sources and name not in sources:
            continue
        ts = time.time()
        try:
            frame = jobs_to_frame(loader(limits.get(name)))
        except FileNotFoundError as e:
            status["sources"][name] = {"status": "missing", "error": str(e)}
            continue
        status["sources"][name] = {"status": "ok", "records": len(frame), "seconds": round(time.time() - ts, 1),
                                   "limit": limits.get(name)}
        progress(f"loaded {name}: {len(frame)} jobs")
        if len(frame):
            frames.append(frame)
    if not frames:
        raise RuntimeError("No jobs loaded from any source")
    df = pd.concat(frames, ignore_index=True)

    mh = settings.get("ingestion.minhash", {})
    df, report = deduplicate(df, use_minhash=use_minhash, threshold=mh.get("threshold", 0.85),
                             num_perm=mh.get("num_perm", 64), shingle_size=mh.get("shingle_size", 5))
    df["split"] = assign_splits(df["dedup_cluster"])
    leak = check_no_leakage(df)
    index_sources = set(settings.get("ingestion.index_sources", []))
    # a posting whose title the source redacted cannot be shown, explained or matched on title — kept, not indexed
    title_masked = df["title"].map(is_masked)
    df["indexable"] = (df["is_representative"] & df["source"].isin(index_sources) & df["description"].notna()
                       & ~title_masked)

    df.to_parquet(out_dir / "jobs.parquet", index=False)
    df[["job_id", "source", "dedup_cluster", "split", "is_representative"]].to_csv(splits_dir / "job_splits.csv", index=False)
    report.update({"leakage_check": leak or "ok", "indexable_jobs": int(df["indexable"].sum()),
                   "split_counts": df["split"].value_counts().to_dict(), "mode": settings.mode,
                   "title_masked_in_source": int(title_masked.sum()), "max_jobs": max_jobs, "seconds": round(time.time() - t0, 1), **status})
    (out_dir / "dedup_report.json").write_text(json.dumps(report, indent=2, default=str))
    progress(f"wrote {len(df)} jobs ({report['indexable_jobs']} indexable) in {report['seconds']}s")
    return report


def load_jobs(path: Optional[Path] = None, indexable_only: bool = True) -> list[Job]:
    path = path or get_settings().path("processed") / "jobs.parquet"
    df = pd.read_parquet(path)
    if indexable_only and "indexable" in df:
        df = df[df["indexable"]]
    return [row_to_job(r) for r in df.to_dict("records")]
