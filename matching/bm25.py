"""Okapi BM25 over a sparse term matrix (vectorised; scales to the full corpus).

score(q, d) = Σ_{t∈q} idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
idf(t) = ln(1 + (N − df + 0.5)/(df + 0.5))
The per-document weights are precomputed, so a query is one sparse column sum.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9+#.\-]*[a-zA-Z0-9+#]\b|\b[a-zA-Z0-9]\b"


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN, stop_words="english",
                                          dtype=np.float32)
        self.W: Optional[sparse.csc_matrix] = None

    def fit(self, docs: list[str]) -> "BM25":
        tf = self.vectorizer.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(dl.mean()) if n else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avgdl)
        data = idf[tf.col] * tf.data * (self.k1 + 1) / denom
        self.W = sparse.csc_matrix((data, (tf.row, tf.col)), shape=tf.shape, dtype=np.float32)
        self.idf = idf
        return self

    def query_terms(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return q.indices

    def scores(self, query: str) -> np.ndarray:
        cols = np.unique(self.query_terms(query))
        if cols.size == 0:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        return np.asarray(self.W[:, cols].sum(axis=1)).ravel()

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"k1": self.k1, "b": self.b, "vectorizer": self.vectorizer, "W": self.W, "idf": self.idf}, fh)

    @classmethod
    def load(cls, path: Path) -> "BM25":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        obj = cls(d["k1"], d["b"])
        obj.vectorizer, obj.W, obj.idf = d["vectorizer"], d["W"], d["idf"]
        return obj
