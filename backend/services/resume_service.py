"""Resume upload, parsing, correction and deletion."""
from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from backend.database.sqlite import log_activity
from backend.repositories import candidates as repo
from common.config import get_settings
from common.schemas import Candidate
from common.text import sha256_bytes
from parsing.resume_parser import parse_resume_file
from parsing.text_extraction import UnsupportedFile, detect_type

log = logging.getLogger(__name__)
ALLOWED = {"pdf", "docx", "txt"}


class UploadError(ValueError):
    pass


def sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "resume").replace("\x00", "")
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .") or "resume"
    return base[:120]


def upload_dir() -> Path:
    d = get_settings().path("uploads")
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def handle_upload(data: bytes, filename: str, use_llm="auto") -> dict:
    max_mb = float(get_settings().get("parsing.max_upload_mb", 5))
    if not data:
        raise UploadError("Empty file")
    if len(data) > max_mb * 1024 * 1024:
        raise UploadError(f"File exceeds the {max_mb:.0f} MB limit")
    safe = sanitize_filename(filename)
    ext = safe.rsplit(".", 1)[-1].lower() if "." in safe else ""
    if ext not in ALLOWED:
        raise UploadError("Only PDF, DOCX and TXT files are accepted")
    try:
        kind = detect_type(data, safe)
    except UnsupportedFile as e:
        raise UploadError(str(e))
    if kind != ext:
        raise UploadError(f"File content is {kind.upper()} but the extension is .{ext}")
    digest = sha256_bytes(data)
    existing = repo.find_resume_by_hash(digest)
    if existing and repo.get_candidate(existing["candidate_id"]):
        ensure_in_graph(existing["candidate_id"])  # the first upload's graph sync may have failed (graph down)
        return {"resume_id": existing["resume_id"], "candidate_id": existing["candidate_id"], "duplicate": True}
    rid = "cand-" + uuid.uuid4().hex[:12]
    stored = upload_dir() / f"{rid}.{ext}"           # server-generated name; never the client's
    stored.write_bytes(data)
    os.chmod(stored, 0o600)
    try:
        res = parse_resume_file(data, safe, rid, use_llm=use_llm)
    except Exception as e:
        stored.unlink(missing_ok=True)
        raise UploadError(f"Could not parse resume: {e}")
    cand = res.candidate
    repo.save_candidate(cand, source="upload")
    repo.save_resume(rid, rid, safe, str(stored), res.file_type, digest, len(data), "parsed", res.warnings,
                     res.injection_flags, res.extractors)
    graph_status = sync_graph(cand)
    log_activity("resume_upload", f"Parsed {safe}: {len(cand.skills)} skills, {len(cand.experiences)} roles", rid)
    return {"resume_id": rid, "candidate_id": rid, "duplicate": False, "warnings": res.warnings,
            "injection_flags": res.injection_flags, "extractors": res.extractors, "graph": graph_status,
            "llm_stats": res.llm_stats}


class ReparseConflict(Exception):
    pass


def reparse_resume(candidate_id: str, use_llm="auto", force: bool = False) -> dict:
    """Parse the stored upload again with the current parser (e.g. after a parser fix).

    A profile the user corrected is not overwritten unless ``force`` — the corrections would be lost.
    """
    cur = repo.get_candidate(candidate_id)
    meta = repo.get_resume(candidate_id)
    if cur is None or meta is None:
        raise KeyError(candidate_id)
    path = Path(meta.get("stored_path") or "")
    if not path.is_file():
        raise UploadError("The original file is no longer stored, so it cannot be re-parsed")
    if cur.user_corrected and not force:
        raise ReparseConflict("This profile has saved corrections; re-parsing would discard them")
    data = path.read_bytes()
    res = parse_resume_file(data, meta["original_filename"], candidate_id, use_llm=use_llm)
    cand = res.candidate
    repo.save_candidate(cand, source="upload", replace_original=True)
    repo.save_resume(candidate_id, candidate_id, meta["original_filename"], str(path), res.file_type, meta["sha256"],
                     len(data), "parsed", res.warnings, res.injection_flags, res.extractors)
    graph_status = sync_graph(cand)
    log_activity("resume_reparse", f"Re-parsed {meta['original_filename']} ({cand.extractor}): "
                                   f"{len(cand.skills)} skills, {len(cand.experiences)} roles", candidate_id)
    return {"resume_id": candidate_id, "candidate_id": candidate_id, "extractor": cand.extractor,
            "before": {"skills": len(cur.skills), "experiences": len(cur.experiences), "education": len(cur.education),
                       "credentials": len(cur.credentials), "projects": len(cur.projects)},
            "after": {"skills": len(cand.skills), "experiences": len(cand.experiences), "education": len(cand.education),
                      "credentials": len(cand.credentials), "projects": len(cand.projects)},
            "warnings": res.warnings, "graph": graph_status}


def sync_graph(c: Candidate) -> dict:
    try:
        from knowledge_graph.graph_builder import upsert_candidate
        from knowledge_graph.neo4j_client import get_graph_store

        store = get_graph_store()
        return {"backend": store.backend, **upsert_candidate(store, c)}
    except Exception as e:  # graph is an enhancement; the profile is still saved
        log.warning("Graph sync failed: %s", type(e).__name__)
        return {"error": f"{type(e).__name__}: {e}"}


def ensure_in_graph(candidate_id: str) -> bool:
    """Write a stored profile to the graph if it is missing there. True when the candidate is in the graph."""
    try:
        from knowledge_graph.entity_linking import candidate_node_id
        from knowledge_graph.neo4j_client import get_graph_store

        if get_graph_store().node(candidate_node_id(candidate_id)) is not None:
            return True
    except Exception as e:
        log.warning("Graph lookup failed: %s", type(e).__name__)
        return False
    cand = repo.get_candidate(candidate_id)
    return cand is not None and "error" not in sync_graph(cand)


def update_profile(candidate_id: str, profile: Candidate) -> Candidate:
    cur = repo.get_candidate(candidate_id)
    if cur is None:
        raise KeyError(candidate_id)
    from normalization.skill_normalizer import get_skill_normalizer

    sk = get_skill_normalizer()
    # skills the user added (method "user") or that are unresolved are normalised again;
    # extracted skills keep their canonical id and evidence
    for s in profile.skills:
        if s.method == "user" or not s.canonical_id:
            n = sk.normalize(s.mention)
            s.canonical_id, s.canonical_name, s.taxonomy, s.confidence = (n.canonical_id, n.canonical_name,
                                                                        n.taxonomy, n.confidence)
            s.method = "user" if s.method == "user" else n.method
    profile.candidate_id = candidate_id
    profile.raw_text = cur.raw_text
    profile.source_document = cur.source_document
    profile.user_corrected = True
    repo.save_candidate(profile)
    sync_graph(profile)
    log_activity("profile_corrected", f"Profile saved with {len(profile.skills)} skills", candidate_id)
    return profile


def delete_resume(candidate_id: str) -> dict:
    res = repo.delete_candidate(candidate_id)
    removed_files = 0
    for p in res["files"]:
        path = Path(p)
        if path.exists() and upload_dir() in path.resolve().parents:
            path.unlink()
            removed_files += 1
    graph_removed = 0
    try:
        from knowledge_graph.entity_linking import candidate_node_id
        from knowledge_graph.neo4j_client import get_graph_store

        store = get_graph_store()
        graph_removed = store.delete_subject(candidate_node_id(candidate_id))
        if hasattr(store, "save"):
            store.save()
    except Exception as e:
        log.warning("Graph delete failed: %s", type(e).__name__)
    return {"candidate_id": candidate_id, **res["rows_deleted"], "files_removed": removed_files,
            "graph_nodes_removed": graph_removed}
