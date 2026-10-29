"""Run research experiments and write Data/results/*.json.

  python scripts/evaluate.py                    # all experiments
  python scripts/evaluate.py --only B C classifiers
  python scripts/evaluate.py --import-candidates 40   # ensure an evaluation candidate pool exists first
Experiments without the required labels/providers record not_evaluated / blocked — never invented numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402,F401


def eval_candidates(n: int) -> list[str]:
    """Stratified sample of Kaggle resumes (one per category, round-robin), parsed and stored."""
    from backend.repositories import candidates as repo
    from ingestion.resume_loader import iter_resumes
    from parsing.resume_parser import parse_resume_text

    by_cat: dict[str, list] = {}
    for r in iter_resumes():
        by_cat.setdefault(r.category, []).append(r)
    picked, i = [], 0
    while len(picked) < n and any(by_cat.values()):
        for c in sorted(by_cat):
            if len(by_cat[c]) > i and len(picked) < n:
                picked.append(by_cat[c][i])
        i += 1
    ids = []
    for r in picked:
        if repo.get_candidate(r.resume_id) is None:
            repo.save_candidate(parse_resume_text(r.text, r.resume_id, r.resume_id, use_llm=False).candidate,
                                source="kaggle_resumes")
        ids.append(r.resume_id)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: A B C D E F G classifiers")
    ap.add_argument("--import-candidates", type=int, default=40)
    a = ap.parse_args()
    want = set(a.only or ["A", "B", "C", "D", "E", "F", "G", "classifiers"])
    out = {}
    cands = eval_candidates(a.import_candidates) if want & {"D", "F"} else []
    if "classifiers" in want:
        from evaluation import classifiers
        out["classifiers"] = classifiers.run()["status"]
    if "A" in want:
        from evaluation import extraction
        out["A"] = extraction.run()["status"]
    if "B" in want:
        from evaluation import normalization
        out["B"] = normalization.run()["status"]
    if "C" in want:
        from evaluation import graph
        out["C"] = graph.run()["status"]
    if "D" in want:
        from evaluation import ranking_eval
        out["D"] = ranking_eval.run(cands)["status"]
    if "E" in want:
        from training.qlora import status_result
        out["E"] = status_result()["status"]
    if "F" in want:
        from evaluation import explanations
        out["F"] = explanations.run(cands[:20])["status"]
    if "G" in want:
        from evaluation import benchmarking
        out["G"] = benchmarking.run()["status"]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
