"""Dense retrieval: Sentence-Transformers embeddings + persistent FAISS index.

``IndexIDMap2(IndexFlatIP)`` over L2-normalised vectors = exact cosine search.
Supports rebuild, incremental ``add`` and removal by ID.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from common.embeddings import Encoder, get_encoder


class DenseIndex:
    def __init__(self, dim: int, model_name: str):
        import faiss

        self.faiss = faiss
        self.dim = dim
        self.model_name = model_name
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    @classmethod
    def build(cls, texts: list[str], ids: list[int], encoder: Optional[Encoder] = None, batch: int = 256,
              progress=None) -> "DenseIndex":
        enc = encoder or get_encoder()
        if enc is None:
            raise RuntimeError("Embedding model unavailable")
        obj = cls(enc.dim, enc.model_name)
        for s in range(0, len(texts), batch):
            vecs = enc.encode(texts[s:s + batch])
            obj.index.add_with_ids(vecs, np.asarray(ids[s:s + batch], dtype=np.int64))
            if progress and (s // batch) % 10 == 0:
                progress(f"embedded {min(s + batch, len(texts))}/{len(texts)}")
        return obj

    def add(self, vecs: np.ndarray, ids: list[int]) -> None:
        self.remove(ids)
        self.index.add_with_ids(vecs.astype(np.float32), np.asarray(ids, dtype=np.int64))

    def remove(self, ids: list[int]) -> None:
        self.index.remove_ids(np.asarray(ids, dtype=np.int64))

    def search(self, qvec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.index.ntotal)
        if k == 0:
            return np.array([]), np.array([], dtype=np.int64)
        D, I = self.index.search(qvec.reshape(1, -1).astype(np.float32), k)
        return D[0], I[0]

    def reconstruct(self, ids: list[int]) -> np.ndarray:
        return np.vstack([self.index.reconstruct(int(i)) for i in ids]) if ids else np.zeros((0, self.dim), np.float32)

    def all_scores(self, qvec: np.ndarray) -> np.ndarray:
        """Scores for every stored vector in id order (ids are 0..n-1 positions)."""
        n = self.index.ntotal
        D, I = self.search(qvec, n)
        out = np.full(int(I.max()) + 1 if n else 0, -1.0, dtype=np.float32)
        out[I] = D
        return out

    def save(self, path: Path) -> None:
        self.faiss.write_index(self.index, str(path))
        path.with_suffix(".json").write_text(json.dumps({"dim": self.dim, "model": self.model_name,
                                                         "ntotal": int(self.index.ntotal)}))

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        import faiss

        meta = json.loads(path.with_suffix(".json").read_text())
        obj = cls(meta["dim"], meta["model"])
        obj.index = faiss.read_index(str(path))
        return obj
