"""Health, dashboard, models, evaluation results, annotations and admin endpoints."""
from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.repositories import candidates as repo
from backend.schemas.api import ImportSampleIn, IngestionIn
from common.config import get_settings
from annotation.status import annotation_status  # module level: see annotations_status

router = APIRouter(tags=["system"])
_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()


@router.get("/health")
def health():
    out = {"status": "ok", "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "mode": get_settings().mode}
    try:
        from matching.store import get_store

        out["jobs_indexed"] = len(get_store())
    except FileNotFoundError:
        out["jobs_indexed"] = 0
        out["status"] = "degraded"
        out["detail"] = "Job index missing: run scripts/ingest_data.py and scripts/build_indexes.py"
    try:
        from knowledge_graph.neo4j_client import get_graph_store

        out["graph_backend"] = get_graph_store().backend
    except Exception as e:
        out["graph_backend"] = f"unavailable: {type(e).__name__}"
    return out


@router.get("/api/dashboard/summary")
def dashboard(candidate_id: str | None = None):
    from matching.store import get_store

    store = get_store()
    out = {"jobs_indexed": len(store), "job_sources": store.source_counts(),
           "work_modes": store.df["work_mode"].fillna("unknown").value_counts().to_dict(),
           "candidates": repo.candidate_summaries(limit=1)[1], "recent_activity": repo.recent_activity(12)}
    if candidate_id:
        c = repo.get_candidate(candidate_id)
        if c is None:
            raise HTTPException(404, "Candidate not found")
        run = repo.latest_match_run(candidate_id, "final") or repo.latest_match_run(candidate_id)
        out["candidate"] = {
            "candidate_id": c.candidate_id, "source_document": c.source_document, "skills": len(c.skills),
            "experience_years": c.total_experience_years, "user_corrected": c.user_corrected,
            "occupations": [{"code": o.code, "title": o.canonical_title, "method": o.method,
                             "confidence": o.confidence} for o in c.occupations][:5],
            "top_skills": [s.canonical_name or s.mention for s in sorted(c.skills, key=lambda s: -len(s.evidence))][:12],
            "resume": {k: v for k, v in (repo.get_resume(candidate_id) or {}).items() if k != "stored_path"}}
        if run:
            res = run["results"][:5]
            covs = [r["coverage_required"] for r in run["results"] if r.get("coverage_required") is not None]
            out["top_recommendations"] = [{"rank": r["rank"], "title": r["job"]["title"], "company": r["job"]["company"],
                                           "source": r["job"]["source"], "score": r["score"],
                                           "coverage_required": r.get("coverage_required"),
                                           "job_id": r["job"]["job_id"]} for r in res]
            out["mean_required_coverage"] = round(sum(covs) / len(covs), 4) if covs else None
            out["last_run"] = {"run_id": run["run_id"], "method": run["method"], "ranker": run["ranker"],
                               "created_at": run["created_at"]}
    return out


@router.get("/api/models/status")
def models_status():
    from llm.router import get_router
    from matching import lambdamart
    from matching.pipeline import get_pipeline

    return {"llm": get_router().status(), "matching": get_pipeline().status(), "lambdamart": lambdamart.status(),
            "lightcast": {"configured": bool(__import__("common.config", fromlist=["env"]).env("LIGHTCAST_CLIENT_ID")),
                          "used": False, "note": "Optional; O*NET + local aliases + TF-IDF + embeddings are used."}}


@router.get("/api/models/metrics")
def models_metrics():
    from llm import call_log

    return {"summary": call_log.summary(), "recent": call_log.recent(30)}


@router.get("/api/evaluation/results")
def evaluation_results():
    d = get_settings().path("results")
    out = {}
    if d.exists():
        for p in sorted(d.glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text())
            except json.JSONDecodeError:
                out[p.stem] = {"error": "unreadable"}
    from evaluation.registry import EXPERIMENTS

    return {"experiments": EXPERIMENTS, "results": out}


@router.get("/api/annotations/status")
def annotations_status():
    return annotation_status()


def _run_job(name: str, fn) -> str:
    with _job_lock:
        if any(j["status"] == "running" for j in _jobs.values()):
            raise HTTPException(409, "Another admin job is running")
        jid = f"{name}-{int(time.time())}"
        _jobs[jid] = {"job": jid, "status": "running", "started": datetime.now(timezone.utc).isoformat(), "log": []}

    def target():
        try:
            _jobs[jid]["result"] = fn(lambda m: _jobs[jid]["log"].append(str(m)))
            _jobs[jid]["status"] = "done"
        except Exception as e:
            _jobs[jid]["status"] = "failed"
            _jobs[jid]["error"] = f"{type(e).__name__}: {e}"
            _jobs[jid]["log"].append(traceback.format_exc()[-2000:])
        _jobs[jid]["finished"] = datetime.now(timezone.utc).isoformat()

    threading.Thread(target=target, daemon=True).start()
    return jid


@router.post("/api/admin/ingestion/run")
def run_ingestion(body: IngestionIn):
    def work(log):
        import subprocess
        import sys

        from common.config import PROJECT_ROOT

        steps = [["scripts/ingest_data.py"] + (["--full"] if body.full else []) + (["--fetch-live"] if body.fetch_live else [])
                 + (["--adzuna-query", body.adzuna_query] if body.adzuna_query else [])]
        if body.rebuild_indexes:
            steps.append(["scripts/build_indexes.py"])
        if body.rebuild_graph:
            steps.append(["scripts/build_graph.py"])
        for st in steps:
            log(f"$ python {' '.join(st)}")
            p = subprocess.run([sys.executable, *st], cwd=PROJECT_ROOT, capture_output=True, text=True)
            log(p.stdout[-3000:])
            if p.returncode != 0:
                raise RuntimeError(f"{st[0]} failed: {p.stderr[-1500:]}")
        from knowledge_graph.neo4j_client import get_graph_store
        from matching.pipeline import reset_pipeline

        reset_pipeline()
        get_graph_store.cache_clear()
        return {"steps": len(steps)}

    return {"job": _run_job("ingestion", work)}


@router.post("/api/admin/candidates/import-sample")
def import_sample(body: ImportSampleIn):
    def work(log):
        from backend.services.resume_service import sync_graph
        from parsing.resume_parser import parse_resume_text

        n = 0
        if body.source == "kaggle_resumes":
            from ingestion.resume_loader import iter_resumes

            recs = ((r.resume_id, r.text) for r in iter_resumes(limit=body.limit))
        else:
            from ingestion.resume_corpus_loader import iter_corpus

            recs = ((r.resume_id, r.text) for r in iter_corpus(limit=body.limit))
        for rid, text in recs:
            if repo.get_candidate(rid):
                continue
            res = parse_resume_text(text, rid, rid, use_llm=False)
            repo.save_candidate(res.candidate, source=body.source)
            sync_graph(res.candidate)
            n += 1
            if n % 10 == 0:
                log(f"imported {n}")
        return {"imported": n}

    return {"job": _run_job("import", work)}


@router.get("/api/admin/jobs/{job}")
def admin_job(job: str):
    if job not in _jobs:
        raise HTTPException(404, "Unknown admin job")
    return _jobs[job]


@router.get("/api/admin/jobs")
def admin_jobs():
    return {"items": list(_jobs.values())}


_ann_lock = threading.Lock()


@router.get("/api/annotations/relevance/{candidate_id}/pool")
def relevance_pool(candidate_id: str, per_method: int = 5, pool_size: int = 20):
    """Depth-k pool over all ranking methods, shuffled deterministically; method names hidden."""
    import random

    from matching.pipeline import METHODS, get_pipeline

    cand = repo.get_candidate(candidate_id)
    if cand is None:
        raise HTTPException(404, "Candidate not found")
    p = get_pipeline()
    pooled: list[str] = []
    for m in METHODS:
        if m == "dense" and p.dense is None:
            continue
        for r in p.recommend(cand, method=m, k=per_method, with_gap=False, use_cross_encoder=(m == "final")).results:
            if r.job_id not in pooled:
                pooled.append(r.job_id)
    random.Random(f"{candidate_id}-pool").shuffle(pooled)
    from backend.services.matching_service import job_card

    jobs = []
    for jid in pooled[:pool_size]:
        j = p.store.by_id(jid)
        jobs.append(job_card(j) | {"description": (j.description or "")[:3000]})
    from matching.lambdamart import judgments_path
    import pandas as pd

    done = {}
    if judgments_path().exists():
        df = pd.read_csv(judgments_path())
        df = df[df["candidate_id"] == candidate_id]
        done = {r.job_id: {"label": int(r.label), "annotator_id": r.annotator_id} for r in df.itertuples()}
    return {"candidate_id": candidate_id, "jobs": jobs, "judged": done}


@router.post("/api/annotations/relevance")
def add_judgment(body: dict):
    import csv

    from matching.lambdamart import judgments_path

    try:
        label = int(body["label"])
        cid, jid, ann = str(body["candidate_id"])[:80], str(body["job_id"])[:120], str(body["annotator_id"])[:40]
    except (KeyError, TypeError, ValueError):
        raise HTTPException(422, "candidate_id, job_id, label (0/1/2) and annotator_id are required")
    if label not in (0, 1, 2) or not ann.strip():
        raise HTTPException(422, "label must be 0, 1 or 2 and annotator_id must be non-empty")
    p = judgments_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _ann_lock:
        new = not p.exists()
        with open(p, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["candidate_id", "job_id", "label", "annotator_id", "timestamp", "notes"])
            w.writerow([cid, jid, label, ann, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        str(body.get("notes", ""))[:500]])
    return {"saved": True}


@router.post("/api/admin/evaluation/run")
def run_evaluation(body: dict):
    allowed = {"A", "B", "C", "D", "E", "F", "G", "classifiers"}
    only = [x for x in (body.get("only") or []) if x in allowed] or sorted(allowed)

    def work(log):
        import subprocess
        import sys

        from common.config import PROJECT_ROOT

        cmd = [sys.executable, "scripts/evaluate.py", "--only", *only]
        log("$ python " + " ".join(cmd[1:]))
        p = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        log(p.stdout[-4000:])
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])
        return {"experiments": only}

    return {"job": _run_job("evaluation", work)}
