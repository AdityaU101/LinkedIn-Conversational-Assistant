"""Nearest-neighbour lookup over sentence embeddings of canonical names."""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.embeddings import Encoder, get_encoder


class EmbeddingIndex:
    """Cosine top-k over a fixed list of labels (vectors are L2-normalised)."""

    def __init__(self, labels: list[str], ids: list[str], cache_name: str, encoder: Optional[Encoder] = None):
        self.labels = labels
        self.ids = ids
        self.encoder = encoder or get_encoder()
        self.vectors = self.encoder.encode_cached(labels, cache_name) if self.encoder else None

    @property
    def available(self) -> bool:
        return self.vectors is not None

    def query(self, texts: list[str], k: int = 5) -> list[list[tuple[str, str, float]]]:
        if not self.available or not texts:
            return [[] for _ in texts]
        q = self.encoder.encode(texts)
        sims = q @ self.vectors.T
        out = []
        for row in sims:
            top = np.argpartition(-row, min(k, len(row) - 1))[:k]
            top = top[np.argsort(-row[top])]
            out.append([(self.ids[i], self.labels[i], float(row[i])) for i in top])
        return out
