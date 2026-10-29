"""Thin HTTP client for the FastAPI backend. All business logic lives in the backend."""
from __future__ import annotations

import os
from typing import Any, Optional

import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")


class APIError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class APIClient:
    def __init__(self, base_url: str = BACKEND_URL, timeout: float = 180.0, transport: Optional[httpx.BaseTransport] = None):
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def _req(self, method: str, path: str, **kw) -> Any:
        try:
            r = self._client.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise APIError(0, f"Backend unreachable at {self.base_url} ({type(e).__name__}). Is FastAPI running?")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise APIError(r.status_code, detail)
        return r.json()

    # system
    def health(self): return self._req("GET", "/health")
    def dashboard(self, candidate_id=None): return self._req("GET", "/api/dashboard/summary", params={"candidate_id": candidate_id} if candidate_id else None)
    def models_status(self): return self._req("GET", "/api/models/status")
    def models_metrics(self): return self._req("GET", "/api/models/metrics")
    def evaluation_results(self): return self._req("GET", "/api/evaluation/results")
    def annotations_status(self): return self._req("GET", "/api/annotations/status")
    # resumes
    def upload_resume(self, filename: str, data: bytes, use_llm: str = "auto"):
        return self._req("POST", "/api/resumes/upload", files={"file": (filename, data)}, params={"use_llm": use_llm})
    def list_resumes(self, limit=100, source=None):
        return self._req("GET", "/api/resumes", params={"limit": limit, **({"source": source} if source else {})})
    def resume(self, rid): return self._req("GET", f"/api/resumes/{rid}")
    def analysis(self, rid): return self._req("GET", f"/api/resumes/{rid}/analysis")
    def save_profile(self, rid, profile: dict): return self._req("PUT", f"/api/resumes/{rid}/profile", json={"profile": profile})
    def delete_resume(self, rid): return self._req("DELETE", f"/api/resumes/{rid}")
    def reparse_resume(self, rid, use_llm="auto", force=False):
        return self._req("POST", f"/api/resumes/{rid}/reparse", params={"use_llm": use_llm, "force": str(force).lower()})
    # jobs / matching
    def jobs(self, **params): return self._req("GET", "/api/jobs", params={k: v for k, v in params.items() if v not in (None, "")})
    def job(self, job_id): return self._req("GET", f"/api/jobs/{job_id}")
    def search_jobs(self, query: str, filters: dict, limit=20, offset=0):
        return self._req("POST", "/api/jobs/search", json={"query": query, "filters": filters, "limit": limit, "offset": offset})
    def recommend(self, candidate_id, filters=None, method="final", k=10, explain_with_llm=False, use_cross_encoder=True):
        return self._req("POST", "/api/matching/recommend", json={"candidate_id": candidate_id, "filters": filters or {},
                                                                  "method": method, "k": k, "explain_with_llm": explain_with_llm,
                                                                  "use_cross_encoder": use_cross_encoder})
    def results(self, candidate_id, method=None):
        return self._req("GET", f"/api/matching/{candidate_id}/results", params={"method": method} if method else None)
    def job_candidates(self, job_id, k=10, source=None):
        return self._req("POST", "/api/matching/job-candidates", json={"job_id": job_id, "k": k, "source": source})
    def gaps(self, candidate_id, job_id=None):
        return self._req("GET", f"/api/skills/{candidate_id}/gaps", params={"job_id": job_id} if job_id else None)
    def explanation(self, candidate_id, job_id, method="auto"):
        return self._req("POST", "/api/explanations", json={"candidate_id": candidate_id, "job_id": job_id, "method": method})
    # graph
    def graph(self, candidate_id, job_id=None, labels=None, max_nodes=150):
        p = {"max_nodes": max_nodes}
        if job_id:
            p["job_id"] = job_id
        if labels:
            p["labels"] = ",".join(labels)
        return self._req("GET", f"/api/graph/{candidate_id}", params=p)
    def graph_status(self): return self._req("GET", "/api/graph/status")
    def evidence(self, evidence_id): return self._req("GET", f"/api/graph/evidence/{evidence_id}")
    # assistant
    def chat(self, candidate_id, message, job_id=None):
        return self._req("POST", "/api/chat", json={"candidate_id": candidate_id, "message": message, "job_id": job_id})
    def chat_history(self, candidate_id): return self._req("GET", f"/api/chat/{candidate_id}/history")
    def optimize(self, candidate_id, rewrite_bullet=None):
        return self._req("POST", "/api/resume/optimize", json={"candidate_id": candidate_id, "rewrite_bullet": rewrite_bullet})
    def outreach(self, candidate_id, job_id): return self._req("POST", "/api/outreach/draft", json={"candidate_id": candidate_id, "job_id": job_id})
    def followup(self, candidate_id, when, job_id=None, note=""):
        return self._req("POST", "/api/followups", json={"candidate_id": candidate_id, "when": when, "job_id": job_id, "note": note})
    def followups(self, candidate_id=None): return self._req("GET", "/api/followups", params={"candidate_id": candidate_id} if candidate_id else None)
    def set_followup(self, fid, status): return self._req("PATCH", f"/api/followups/{fid}", json={"status": status})
    # admin / annotation
    def run_ingestion(self, **body): return self._req("POST", "/api/admin/ingestion/run", json=body)
    def import_sample(self, source="kaggle_resumes", limit=50): return self._req("POST", "/api/admin/candidates/import-sample", json={"source": source, "limit": limit})
    def admin_job(self, job): return self._req("GET", f"/api/admin/jobs/{job}")
    def relevance_pool(self, candidate_id): return self._req("GET", f"/api/annotations/relevance/{candidate_id}/pool")
    def add_judgment(self, candidate_id, job_id, label, annotator_id, notes=""):
        return self._req("POST", "/api/annotations/relevance", json={"candidate_id": candidate_id, "job_id": job_id,
                                                                      "label": label, "annotator_id": annotator_id, "notes": notes})
