"""Train LambdaMART on human relevance judgments.

  python scripts/train_ltr.py                 # all features
  python scripts/train_ltr.py --no-graph      # ablation: without graph features (RQ4)
Refuses to train with fewer than matching.lambdamart.MIN_JUDGMENTS judgments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402,F401


def build_training_set():
    import numpy as np

    from backend.repositories import candidates as repo
    from matching import lambdamart
    from matching.features import build_context, compute_features
    from matching.pipeline import get_pipeline
    from matching.rrf import rrf

    j = lambdamart.load_judgments()
    p = get_pipeline()
    X, y, groups, skipped = [], [], [], 0
    for cid, grp in j.groupby("candidate_id"):
        cand = repo.get_candidate(cid)
        if cand is None:
            skipped += len(grp)
            continue
        ctx = build_context(cand, p.tax, p.encoder)
        comp = p.component_scores(ctx)
        mask = np.ones(len(p.store), dtype=bool)
        lists = {n: p._top(s, mask, p.retrieve_k, positive_only=(n != "dense")) for n, s in comp.items()}
        _, fused, _ = rrf(lists, p.rrf_k)
        pos = [p.store.pos[jid] for jid in grp["job_id"] if jid in p.store.pos]
        labels = [int(l) for jid, l in zip(grp["job_id"], grp["label"]) if jid in p.store.pos]
        skipped += len(grp) - len(pos)
        if not pos:
            continue
        sub = {n: np.array([float(s[q]) for q in pos]) for n, s in comp.items()}
        sub["rrf"] = np.array([fused.get(q, 0.0) for q in pos])
        X.append(compute_features(ctx, p.store, pos, sub, p.tax))
        y += labels
        groups += [cid] * len(pos)
    return (np.vstack(X) if X else np.zeros((0, 0))), np.array(y), groups, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-graph", action="store_true")
    a = ap.parse_args()
    from matching import lambdamart

    st = lambdamart.status()
    if st["state"] == "awaiting_annotations":
        print(json.dumps({"trained": False, "reason": "awaiting annotations", **st}, indent=2))
        return
    X, y, groups, skipped = build_training_set()
    res = lambdamart.train(X, y, groups, drop_graph_features=a.no_graph, save=not a.no_graph)
    res["skipped_pairs"] = skipped
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
