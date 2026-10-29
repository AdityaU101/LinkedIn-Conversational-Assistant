"""Run job ingestion.

  python scripts/ingest_data.py                    # development sample (configs/settings.yaml)
  python scripts/ingest_data.py --full             # all available data
  python scripts/ingest_data.py --fetch-live       # also pull RemoteOK (+ Adzuna if credentials set)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import get_settings  # noqa: E402
from ingestion.pipeline import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="ingest all rows (ignores max_jobs)")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--fetch-live", action="store_true")
    ap.add_argument("--adzuna-query", default="")
    ap.add_argument("--no-minhash", action="store_true")
    a = ap.parse_args()
    max_jobs = None if a.full else (a.max_jobs or get_settings().max_jobs)
    report = run(max_jobs=max_jobs, sources=a.sources, fetch_live=a.fetch_live, adzuna_query=a.adzuna_query,
                 use_minhash=not a.no_minhash)
    print(json.dumps({k: report[k] for k in ("records", "clusters", "duplicate_records", "merges_by_signal",
                                              "cross_source_links", "indexable_jobs", "leakage_check")}, indent=2))


if __name__ == "__main__":
    main()
