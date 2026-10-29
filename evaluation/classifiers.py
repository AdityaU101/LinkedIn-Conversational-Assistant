"""Logistic-regression baselines evaluated on REAL labels.

1. SOC major group from job title (O*NET held-out alternate titles).
2. Kaggle resume category from resume text (24 categories, stratified 5-fold CV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from evaluation.registry import write_result


def resume_category_cv(seed: int = 13) -> dict:
    from ingestion.resume_loader import base_dir

    df = pd.read_csv(base_dir() / "Resume.csv", usecols=["Resume_str", "Category"])
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=100_000,
                                        stop_words="english"),
                        LogisticRegression(max_iter=3000, C=10.0))
    pred = cross_val_predict(clf, df["Resume_str"], df["Category"],
                             cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    per = pd.Series(pred == df["Category"].to_numpy()).groupby(df["Category"]).mean().round(3).to_dict()
    return {"task": "Kaggle resume category (24 classes)", "model": "TF-IDF(1-2gram) + LogisticRegression",
            "documents": int(len(df)), "cv": "stratified 5-fold, seed 13",
            "accuracy": round(float(accuracy_score(df["Category"], pred)), 4),
            "macro_f1": round(float(f1_score(df["Category"], pred, average="macro")), 4),
            "per_class_accuracy": per}


def run() -> dict:
    from matching.occupation_family import train_and_evaluate

    occ = train_and_evaluate(save=True)
    res = resume_category_cv()
    return write_result("classifiers", "evaluated", {"occupation_family": occ, "resume_category": res})
