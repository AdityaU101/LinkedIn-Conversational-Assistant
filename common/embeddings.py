"""Lazy, shared Sentence-Transformers encoder with on-disk caching.

The model is loaded on first use. If sentence-transformers or the model is
unavailable, :func:`get_encoder` returns ``None`` and callers fall back to
lexical methods — the application keeps working.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from common.config import get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()
_encoders: dict[str, "Encoder"] = {}
_failed: set[str] = set()


class Encoder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        device = os.environ.get("EMBEDDING_DEVICE")
        if not device:
            try:
                import torch

                device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                device = "cpu"
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        getdim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(getdim())

    def encode(self, texts: list[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        bs = batch_size or get_settings().mode_cfg.get("embedding_batch_size", 64)
        vecs = self.model.encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False,
                                 convert_to_numpy=True)
        return vecs.astype(np.float32)

    def encode_cached(self, texts: list[str], cache_name: str) -> np.ndarray:
        """Encode with a cache keyed by model + content hash of the input list."""
        h = hashlib.sha1((self.model_name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
        cache_dir = get_settings().path("cache") / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{cache_name}_{h}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.encode(texts)
        np.save(path, vecs)
        return vecs


def get_encoder(model_name: Optional[str] = None) -> Optional[Encoder]:
    if os.environ.get("DISABLE_EMBEDDINGS") == "1":
        return None
    name = model_name or get_settings().get("matching.embedding_model")
    if name in _failed:
        return None
    with _lock:
        if name not in _encoders:
            try:
                _encoders[name] = Encoder(name)
            except Exception as e:  # model missing / offline / import error
                log.warning("Embedding model %s unavailable (%s); using lexical fallbacks", name, type(e).__name__)
                _failed.add(name)
                return None
        return _encoders[name]
