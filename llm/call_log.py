"""Persistent log of every LLM call (SQLite table ``llm_calls``)."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from common.config import get_settings

_lock = threading.Lock()
SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    task TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    latency_ms REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    success INTEGER NOT NULL,
    error TEXT,
    retries INTEGER DEFAULT 0,
    experiment TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_task ON llm_calls(task);
"""


def db_path() -> Path:
    return get_settings().path("sqlite")


def _conn(path: Optional[Path] = None) -> sqlite3.Connection:
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p, timeout=10)
    c.executescript(SCHEMA)
    return c


def log_call(task: str, provider: str, model: str, latency_ms: float, input_tokens: Optional[int],
             output_tokens: Optional[int], cost_usd: Optional[float], success: bool, error: Optional[str],
             retries: int, experiment: Optional[str] = None) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO llm_calls (timestamp, task, provider, model, latency_ms, input_tokens, output_tokens, cost_usd,"
            " success, error, retries, experiment) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="milliseconds"), task, provider, model, latency_ms,
             input_tokens, output_tokens, cost_usd, int(success), (error or "")[:500] or None, retries, experiment))
        return int(cur.lastrowid)


def summary(experiment: Optional[str] = None) -> list[dict]:
    q = ("SELECT provider, model, task, COUNT(*) n, AVG(latency_ms) avg_latency_ms, SUM(input_tokens) input_tokens,"
         " SUM(output_tokens) output_tokens, SUM(cost_usd) cost_usd, SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END)"
         " cost_unknown_calls, AVG(success) success_rate, SUM(retries) retries FROM llm_calls")
    args: tuple = ()
    if experiment:
        q += " WHERE experiment = ?"
        args = (experiment,)
    q += " GROUP BY provider, model, task ORDER BY provider, model, task"
    with _conn() as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(q, args)]


def recent(limit: int = 50) -> list[dict]:
    with _conn() as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute("SELECT * FROM llm_calls ORDER BY id DESC LIMIT ?", (limit,))]
