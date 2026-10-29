"""Cross-encoder reranking of the top candidates (default ms-marco-MiniLM-L-6-v2)."""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

from common.config import get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()
_model = None
_failed = False


def get_cross_encoder():
    global _model, _failed
    if _failed or os.environ.get("DISABLE_CROSS_ENCODER") == "1" or not get_settings().get("matching.use_cross_encoder", True):
        return None
    with _lock:
        if _model is None:
            try:
                from sentence_transformers import CrossEncoder

                _model = CrossEncoder(get_settings().get("matching.cross_encoder_model"), max_length=512)
            except Exception as e:
                log.warning("Cross-encoder unavailable (%s); reranking stage skipped", type(e).__name__)
                _failed = True
                return None
    return _model


def score_pairs(query: str, docs: list[str]) -> Optional[np.ndarray]:
    model = get_cross_encoder()
    if model is None or not docs:
        return None
    pairs = [(query[:2000], d[:2000]) for d in docs]
    return np.asarray(model.predict(pairs, batch_size=16, show_progress_bar=False), dtype=np.float64)
