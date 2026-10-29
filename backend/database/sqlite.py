"""SQLite application metadata store (parameterised queries only)."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from common.config import get_settings

_lock = threading.RLock()

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS resumes (
    resume_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    original_filename TEXT,
    stored_path TEXT,
    file_type TEXT,
    sha256 TEXT,
    size_bytes INTEGER,
    status TEXT NOT NULL,
    warnings TEXT,
    injection_flags TEXT,
    extractors TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,              -- upload | kaggle_resumes | resume_corpus
    profile_json TEXT NOT NULL,
    original_profile_json TEXT,
    user_corrected INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS match_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    method TEXT NOT NULL,
    filters_json TEXT,
    ranker TEXT,
    results_json TEXT NOT NULL,
    stages_json TEXT,
    timings_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_match_runs_cand ON match_runs(candidate_id, run_id);
CREATE TABLE IF NOT EXISTS followups (
    followup_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    job_id TEXT,
    due_date TEXT NOT NULL,
    note TEXT,
    status TEXT NOT NULL DEFAULT 'pending_approval',  -- pending_approval | approved | done | cancelled
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT,
    kind TEXT NOT NULL,
    summary TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def db_path() -> Path:
    return get_settings().path("sqlite")


_initialised: set[str] = set()


@contextmanager
def connect(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = sqlite3.connect(p, timeout=15, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if str(p) not in _initialised:
            conn.executescript(SCHEMA)
            _initialised.add(str(p))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def log_activity(kind: str, summary: str, candidate_id: Optional[str] = None) -> None:
    with connect() as c:
        c.execute("INSERT INTO activity (candidate_id, kind, summary) VALUES (?,?,?)", (candidate_id, kind, summary))
