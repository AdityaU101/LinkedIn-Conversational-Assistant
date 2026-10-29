"""Build the knowledge graph from indexed jobs (+ stored candidates).

  python scripts/build_graph.py                 # Neo4j if configured & reachable, else in-memory
  python scripts/build_graph.py --backend memory
  python scripts/build_graph.py --limit 500     # first N jobs only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402,F401  (OpenMP bootstrap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "neo4j", "memory"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="discard the existing in-memory graph first")
    a = ap.parse_args()
    from common.config import get_settings

    s = get_settings()
    if a.backend:
        s.raw.setdefault("graph", {})["backend"] = a.backend
    from knowledge_graph.neo4j_client import get_graph_store, memory_path

    if a.fresh and memory_path().exists():
        memory_path().rename(memory_path().with_suffix(".pkl.bak"))
    store = get_graph_store()
    from knowledge_graph.graph_builder import build_graph
    from knowledge_graph.queries import consistency_report
    from matching.store import get_store

    js = get_store()
    n = len(js) if a.limit is None else min(a.limit, len(js))
    jobs = [js.job(i) for i in range(n)]
    candidates = []
    try:
        from backend.repositories.candidates import list_candidates

        candidates = list_candidates()
    except Exception:
        pass
    t0 = time.time()
    stats = build_graph(store, jobs, candidates)
    if a.limit is None:  # postings that left the index (e.g. re-ingestion) would otherwise linger as stale nodes
        from knowledge_graph.entity_linking import job_node_id

        current = {job_node_id(j.job_id) for j in jobs}
        stale = [nd["id"] for nd in store.all_nodes("JobPosting") if nd["id"] not in current]
        stats["stale_postings_removed"] = len(stale)
        for nid in stale:
            store.delete_subject(nid)
        if stale and hasattr(store, "save"):
            store.save()
    report = {"backend": store.backend, "jobs": n, "candidates": len(candidates), "written": stats,
              "seconds": round(time.time() - t0, 1), "counts": store.counts(), "consistency": consistency_report(store)}
    out = s.path("indexes") / "graph_build_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("backend", "jobs", "seconds", "counts", "consistency")}, indent=2, default=str))


if __name__ == "__main__":
    main()
