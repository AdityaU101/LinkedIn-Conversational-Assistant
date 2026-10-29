"""Logistic-regression occupation-family baseline (SOC major group).

Trained on O*NET itself (occupation titles, alternate titles, descriptions and
task statements → SOC major group), so labels are real reference data, not
weak labels. Evaluated on held-out *alternate titles* of every occupation.
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from common.config import get_settings
from ingestion.onet_loader import ONetLoader


def model_path() -> Path:
    return get_settings().path("indexes") / "occupation_family_lr.pkl"


def _training_frames(seed: int = 13) -> tuple[pd.DataFrame, pd.DataFrame]:
    onet = ONetLoader()
    occ = onet.occupations()
    tasks = onet.task_text()
    titles = onet.occupation_titles()
    alt = titles[titles["title_source"] == "alternate"].copy()
    rng = np.random.default_rng(seed)
    alt["u"] = rng.random(len(alt))
    test = alt[alt["u"] < 0.2]
    train_alt = alt[alt["u"] >= 0.2]
    docs = occ.merge(tasks, on="code", how="left")
    train = pd.concat([
        pd.DataFrame({"text": docs["title"] + ". " + docs["description"].fillna("") + " " + docs["tasks"].fillna(""),
                      "code": docs["code"]}),
        pd.DataFrame({"text": train_alt["title"], "code": train_alt["code"]}),
        pd.DataFrame({"text": titles[titles["title_source"].isin(["occupation", "short"])]["title"],
                      "code": titles[titles["title_source"].isin(["occupation", "short"])]["code"]}),
    ], ignore_index=True)
    train["label"] = train["code"].str[:2]
    test = pd.DataFrame({"text": test["title"], "code": test["code"], "label": test["code"].str[:2]})
    # a title that is identical to a training text would leak
    test = test[~test["text"].str.lower().isin(set(train["text"].str.lower()))]
    return train, test


def train_and_evaluate(save: bool = True) -> dict:
    train, test = _training_frames()
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=200_000),
        LogisticRegression(max_iter=2000, C=8.0),
    )
    clf.fit(train["text"], train["label"])
    pred = clf.predict(test["text"])
    result = {"task": "SOC major group from job title", "model": "TF-IDF(1-2gram) + LogisticRegression",
              "train_texts": int(len(train)), "test_titles": int(len(test)),
              "accuracy": round(float(accuracy_score(test["label"], pred)), 4),
              "macro_f1": round(float(f1_score(test["label"], pred, average="macro")), 4),
              "classes": int(train["label"].nunique()),
              "test_set": "20% of O*NET alternate titles held out (seed 13), exact-duplicate texts removed"}
    if save:
        model_path().parent.mkdir(parents=True, exist_ok=True)
        with open(model_path(), "wb") as fh:
            pickle.dump(clf, fh)
        model_path().with_suffix(".json").write_text(json.dumps(result, indent=2))
    get_family_model.cache_clear()
    return result


@lru_cache(maxsize=1)
def get_family_model():
    p = model_path()
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def family_distribution(texts: list[str]) -> Optional[tuple[np.ndarray, list[str]]]:
    m = get_family_model()
    if m is None:
        return None
    return m.predict_proba(texts), list(m.classes_)
