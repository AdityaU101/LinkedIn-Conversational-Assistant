"""Shared utilities.

macOS note: LightGBM (Homebrew libomp), FAISS, scikit-learn and PyTorch each
bundle an OpenMP runtime. With several runtimes active their thread pools
segfault ("OMP: Error #179" / SIGSEGV in worker threads). Verified fix:
import LightGBM first AND run OpenMP single-threaded. Every entry point
imports ``common`` before anything else, so this runs first. Override with
OMP_NUM_THREADS if your platform has a single OpenMP runtime.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:  # pragma: no cover - import side effect only
    import lightgbm  # noqa: F401
except Exception:
    pass
