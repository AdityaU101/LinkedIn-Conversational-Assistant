"""Candidate / resume / match-run / follow-up repositories."""
from __future__ import annotations

import json
from typing import Optional

from backend.database.sqlite import connect
from common.schemas import Candidate


def save_candidate(c: Candidate, source: str = "upload", keep_original: bool = True,
                   replace_original: bool = False) -> None:
    """``replace_original`` marks ``c`` as the new machine parse (a re-parse), not a user correction."""
    data = c.model_dump_json()
    with connect() as db:
        cur = db.execute("SELECT original_profile_json FROM candidates WHERE candidate_id = ?", (c.candidate_id,)).fetchone()
        if cur and replace_original:
            db.execute("UPDATE candidates SET profile_json = ?, original_profile_json = ?, user_corrected = ?, "
                       "updated_at = datetime('now') WHERE candidate_id = ?",
                       (data, data, int(c.user_corrected), c.candidate_id))
        elif cur:
            db.execute("UPDATE candidates SET profile_json = ?, user_corrected = ?, updated_at = datetime('now') "
                       "WHERE candidate_id = ?", (data, int(c.user_corrected), c.candidate_id))
        else:
            db.execute("INSERT INTO candidates (candidate_id, source, profile_json, original_profile_json, user_corrected)"
                       " VALUES (?,?,?,?,?)", (c.candidate_id, source, data, data if keep_original else None,
                                               int(c.user_corrected)))


def get_candidate(candidate_id: str) -> Optional[Candidate]:
    with connect() as db:
        row = db.execute("SELECT profile_json FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
    return Candidate.model_validate_json(row["profile_json"]) if row else None


def get_original(candidate_id: str) -> Optional[Candidate]:
    with connect() as db:
        row = db.execute("SELECT original_profile_json FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
    return Candidate.model_validate_json(row["original_profile_json"]) if row and row["original_profile_json"] else None


def list_candidates(source: Optional[str] = None, limit: int = 10_000) -> list[Candidate]:
    q = "SELECT profile_json FROM candidates"
    args: tuple = ()
    if source:
        q += " WHERE source = ?"
        args = (source,)
    q += " ORDER BY created_at DESC LIMIT ?"
    with connect() as db:
        rows = db.execute(q, args + (limit,)).fetchall()
    return [Candidate.model_validate_json(r["profile_json"]) for r in rows]


def candidate_summaries(source: Optional[str] = None, limit: int = 100, offset: int = 0) -> tuple[list[dict], int]:
    where, args = ("WHERE c.source = ?", (source,)) if source else ("", ())
    with connect() as db:
        total = db.execute(f"SELECT COUNT(*) n FROM candidates c {where}", args).fetchone()["n"]
        rows = db.execute(
            f"SELECT c.candidate_id, c.source, c.user_corrected, c.created_at, c.updated_at, r.original_filename,"
            f" r.status FROM candidates c LEFT JOIN resumes r ON r.candidate_id = c.candidate_id {where}"
            " ORDER BY c.created_at DESC LIMIT ? OFFSET ?", args + (limit, offset)).fetchall()
    return [dict(r) for r in rows], int(total)


def delete_candidate(candidate_id: str) -> dict:
    with connect() as db:
        res = db.execute("SELECT stored_path FROM resumes WHERE candidate_id = ?", (candidate_id,)).fetchall()
        paths = [r["stored_path"] for r in res if r["stored_path"]]
        counts = {}
        for table in ("resumes", "candidates", "match_runs", "followups", "chat_messages", "activity"):
            counts[table] = db.execute(f"DELETE FROM {table} WHERE candidate_id = ?", (candidate_id,)).rowcount
    return {"rows_deleted": counts, "files": paths}


def save_resume(resume_id: str, candidate_id: str, filename: str, stored_path: str, file_type: str, sha256: str,
                size: int, status: str, warnings: list, flags: list, extractors: list) -> None:
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO resumes (resume_id, candidate_id, original_filename, stored_path, file_type,"
                   " sha256, size_bytes, status, warnings, injection_flags, extractors) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (resume_id, candidate_id, filename, stored_path, file_type, sha256, size, status,
                    json.dumps(warnings), json.dumps(flags), json.dumps(extractors)))


def get_resume(resume_id: str) -> Optional[dict]:
    with connect() as db:
        row = db.execute("SELECT * FROM resumes WHERE resume_id = ?", (resume_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    for k in ("warnings", "injection_flags", "extractors"):
        d[k] = json.loads(d[k] or "[]")
    return d


def find_resume_by_hash(sha256: str) -> Optional[dict]:
    with connect() as db:
        row = db.execute("SELECT resume_id FROM resumes WHERE sha256 = ?", (sha256,)).fetchone()
    return get_resume(row["resume_id"]) if row else None


def save_match_run(candidate_id: str, method: str, filters: dict, ranker: str, results: list, stages: dict,
                   timings: dict) -> int:
    with connect() as db:
        cur = db.execute("INSERT INTO match_runs (candidate_id, method, filters_json, ranker, results_json, stages_json,"
                         " timings_json) VALUES (?,?,?,?,?,?,?)",
                         (candidate_id, method, json.dumps(filters), ranker, json.dumps(results, default=str),
                          json.dumps(stages), json.dumps(timings)))
        return int(cur.lastrowid)


def latest_match_run(candidate_id: str, method: Optional[str] = None) -> Optional[dict]:
    q = "SELECT * FROM match_runs WHERE candidate_id = ?"
    args: tuple = (candidate_id,)
    if method:
        q += " AND method = ?"
        args += (method,)
    with connect() as db:
        row = db.execute(q + " ORDER BY run_id DESC LIMIT 1", args).fetchone()
    if not row:
        return None
    d = dict(row)
    for k in ("filters_json", "results_json", "stages_json", "timings_json"):
        d[k.replace("_json", "")] = json.loads(d.pop(k) or "null")
    return d


def add_followup(candidate_id: str, job_id: Optional[str], due_date: str, note: str) -> int:
    with connect() as db:
        cur = db.execute("INSERT INTO followups (candidate_id, job_id, due_date, note) VALUES (?,?,?,?)",
                         (candidate_id, job_id, due_date, note))
        return int(cur.lastrowid)


def list_followups(candidate_id: Optional[str] = None) -> list[dict]:
    q, args = ("SELECT * FROM followups WHERE candidate_id = ? ORDER BY due_date", (candidate_id,)) if candidate_id \
        else ("SELECT * FROM followups ORDER BY due_date", ())
    with connect() as db:
        return [dict(r) for r in db.execute(q, args).fetchall()]


def set_followup_status(followup_id: int, status: str) -> bool:
    with connect() as db:
        return db.execute("UPDATE followups SET status = ?, updated_at = datetime('now') WHERE followup_id = ?",
                          (status, followup_id)).rowcount > 0


def add_chat(candidate_id: Optional[str], role: str, content: str, meta: Optional[dict] = None) -> None:
    with connect() as db:
        db.execute("INSERT INTO chat_messages (candidate_id, role, content, meta_json) VALUES (?,?,?,?)",
                   (candidate_id, role, content, json.dumps(meta or {}, default=str)))


def chat_history(candidate_id: Optional[str], limit: int = 50) -> list[dict]:
    with connect() as db:
        rows = db.execute("SELECT role, content, meta_json, created_at FROM chat_messages WHERE candidate_id IS ? "
                          "ORDER BY id DESC LIMIT ?", (candidate_id, limit)).fetchall()
    return [dict(r, meta=json.loads(r["meta_json"] or "{}")) for r in reversed(rows)]


def recent_activity(limit: int = 15) -> list[dict]:
    with connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]
