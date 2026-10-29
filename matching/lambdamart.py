"""LambdaMART (LightGBM ``lambdarank``) learning-to-rank.

Trains ONLY on human relevance judgments in
``Data/annotations/relevance/judgments.csv`` (0 irrelevant, 1 partial, 2 high).
No labels are synthesised. Until enough judgments exist the status is
``awaiting_annotations`` and the pipeline uses the documented fallback ranker
(matching/features.py: FALLBACK_WEIGHTS).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from common.config import get_settings
from matching.features import FEATURE_NAMES, GRAPH_FEATURES

MIN_JUDGMENTS = 200
MIN_CANDIDATES = 10


def model_dir() -> Path:
    d = get_settings().path("indexes") / "lambdamart"
    d.mkdir(parents=True, exist_ok=True)
    return d


def judgments_path() -> Path:
    return get_settings().path("annotations") / "relevance" / "judgments.csv"


def load_judgments(path: Optional[Path] = None) -> pd.DataFrame:
    """Aggregate multiple annotators per pair by the median label (rounded down)."""
    path = path or judgments_path()
    cols = ["candidate_id", "job_id", "label", "annotator_id"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    df = df.dropna(subset=["candidate_id", "job_id", "label"])
    df = df[df["label"].isin([0, 1, 2])]
    agg = df.groupby(["candidate_id", "job_id"])["label"].median().apply(np.floor).astype(int).reset_index()
    return agg


def status() -> dict:
    j = load_judgments()
    st_path = model_dir() / "status.json"
    trained = json.loads(st_path.read_text()) if st_path.exists() else None
    ready = len(j) >= MIN_JUDGMENTS and j["candidate_id"].nunique() >= MIN_CANDIDATES if len(j) else False
    return {
        "state": "trained" if trained and (model_dir() / "model.txt").exists() else
                 ("ready_to_train" if ready else "awaiting_annotations"),
        "judgments": int(len(j)), "candidates": int(j["candidate_id"].nunique()) if len(j) else 0,
        "required": {"judgments": MIN_JUDGMENTS, "candidates": MIN_CANDIDATES},
        "fallback": "weighted feature sum (matching/features.py FALLBACK_WEIGHTS)",
        "last_training": trained,
    }


def train(X: np.ndarray, y: np.ndarray, groups: list[str], feature_names: list[str] = FEATURE_NAMES,
          drop_graph_features: bool = False, n_splits: int = 5, seed: int = 13, save: bool = True) -> dict:
    """Group-K-fold CV by candidate, then fit on all data. Returns metrics."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    from evaluation.ranking import ndcg_at_k

    names = [f for f in feature_names if not (drop_graph_features and f in GRAPH_FEATURES)]
    cols = [feature_names.index(f) for f in names]
    X = X[:, cols]
    groups_arr = np.asarray(groups)
    params = dict(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                  subsample=0.9, colsample_bytree=0.9, random_state=seed, verbose=-1)
    fold_ndcg = []
    k = min(n_splits, len(set(groups)))
    if k >= 2:
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups_arr):
            order = np.argsort(groups_arr[tr], kind="stable")
            tr = tr[order]
            _, sizes = np.unique(groups_arr[tr], return_counts=True)
            # np.unique sorts; keep the group sizes in the same sorted order as rows
            m = lgb.LGBMRanker(**params)
            m.fit(X[tr], y[tr], group=sizes)
            pred = m.predict(X[te])
            per_q = []
            for g in np.unique(groups_arr[te]):
                idx = np.where(groups_arr[te] == g)[0]
                ranking = idx[np.argsort(-pred[idx])]
                per_q.append(ndcg_at_k([int(y[te][i]) for i in ranking], 10))
            fold_ndcg.append(float(np.mean(per_q)))
    order = np.argsort(groups_arr, kind="stable")
    _, sizes = np.unique(groups_arr[order], return_counts=True)
    model = lgb.LGBMRanker(**params)
    model.fit(X[order], y[order], group=sizes)
    result = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pairs": int(len(y)),
              "candidates": int(len(set(groups))), "features": names, "cv_folds": k,
              "cv_ndcg@10_mean": round(float(np.mean(fold_ndcg)), 4) if fold_ndcg else None,
              "cv_ndcg@10_folds": [round(v, 4) for v in fold_ndcg],
              "feature_importance": dict(zip(names, [int(v) for v in model.feature_importances_])),
              "drop_graph_features": drop_graph_features}
    if save:
        model.booster_.save_model(str(model_dir() / "model.txt"))
        (model_dir() / "status.json").write_text(json.dumps(result, indent=2))
        get_model.cache_clear()
    return result


@lru_cache(maxsize=1)
def get_model():
    p = model_dir() / "model.txt"
    st = model_dir() / "status.json"
    if not p.exists() or not st.exists():
        return None
    import lightgbm as lgb

    meta = json.loads(st.read_text())
    return lgb.Booster(model_file=str(p)), meta["features"]


def predict(F: np.ndarray) -> Optional[np.ndarray]:
    m = get_model()
    if m is None:
        return None
    booster, names = m
    cols = [FEATURE_NAMES.index(n) for n in names]
    return booster.predict(F[:, cols])
