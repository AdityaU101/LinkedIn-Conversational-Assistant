"""Create annotation task files under Data/annotations/ (labels are never pre-filled).

  python scripts/create_annotation_tasks.py              # all task types
  python scripts/create_annotation_tasks.py --only relevance graph
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402,F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--profiles", type=int, default=40)
    a = ap.parse_args()
    from annotation import tasks
    from scripts.evaluate import eval_candidates

    want = set(a.only or ["extraction", "normalization", "relevance", "graph", "explanations", "user_study"])
    out = {}
    if "extraction" in want:
        out["extraction"] = tasks.extraction_tasks()
    if "normalization" in want:
        out["normalization"] = tasks.normalization_tasks()
    cands = eval_candidates(a.profiles) if want & {"relevance", "explanations"} else []
    if "relevance" in want:
        out["relevance"] = tasks.relevance_pool(cands)
    if "graph" in want:
        out["graph"] = tasks.graph_edge_tasks()
    if "explanations" in want:
        from matching.pipeline import get_pipeline
        from backend.repositories import candidates as repo

        pairs = []
        for cid in cands[:20]:
            c = repo.get_candidate(cid)
            pairs += [(cid, r.job_id) for r in get_pipeline().recommend(c, method="ltr", k=3, with_gap=False).results]
        out["explanations"] = tasks.explanation_tasks(pairs[:60])
    if "user_study" in want:
        out["user_study"] = tasks.user_study_template()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
