"""TF-IDF cosine similarity retriever (word uni+bigrams, sublinear tf)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRetriever:
    def __init__(self, max_features: int = 200_000):
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1,
                                          max_df=0.9, sublinear_tf=True, max_features=max_features, dtype=np.float32)
        self.M = None

    def fit(self, docs: list[str]) -> "TfidfRetriever":
        self.M = self.vectorizer.fit_transform(docs).tocsr()
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return np.asarray((self.M @ q.T).toarray()).ravel()

    def vectors(self, rows) -> "np.ndarray":
        return self.M[rows]

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"vectorizer": self.vectorizer, "M": self.M}, fh)

    @classmethod
    def load(cls, path: Path) -> "TfidfRetriever":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        obj = cls()
        obj.vectorizer, obj.M = d["vectorizer"], d["M"]
        return obj
