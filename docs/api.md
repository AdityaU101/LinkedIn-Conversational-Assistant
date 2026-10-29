# API

Interactive docs: http://localhost:8000/docs (OpenAPI at `/openapi.json`).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, mode, indexed jobs, graph backend |
| POST | `/api/resumes/upload` | multipart `file` (PDF/DOCX/TXT ≤ 5 MB); `?use_llm=auto|true|false` |
| GET | `/api/resumes` | list profiles (`limit`, `offset`, `source`) |
| GET | `/api/resumes/{id}` | metadata + extracted text (stored path never exposed) |
| GET | `/api/resumes/{id}/analysis` | structured candidate with evidence + counts |
| PUT | `/api/resumes/{id}/profile` | save a corrected profile (`{"profile": Candidate}`) |
| DELETE | `/api/resumes/{id}` | delete file, rows, runs, chat, graph nodes |
| GET | `/api/jobs` | paginated jobs (`source`, `work_mode`, `location`, `title`) |
| GET | `/api/jobs/{job_id}` | full canonical job (+ raw values) |
| POST | `/api/jobs/search` | BM25 search + filters |
| POST | `/api/matching/recommend` | `{candidate_id, method, k, filters, explain_with_llm, use_cross_encoder}` |
| GET | `/api/matching/{candidate_id}/results` | latest stored run (`?method=`) |
| POST | `/api/matching/job-candidates` | rank stored profiles for a job |
| GET | `/api/skills/{candidate_id}/gaps` | `?job_id=` (defaults to top recommendation) |
| POST | `/api/explanations` | `{candidate_id, job_id, method: auto|template|llm}` |
| GET | `/api/graph/{candidate_id}` | subgraph (`job_id`, `labels`, `max_nodes`), skills, paths |
| GET | `/api/graph/evidence/{evidence_id}` | one Evidence node |
| GET | `/api/graph/status`, `/api/graph/consistency` | counts; ontology/provenance report |
| POST | `/api/chat` | `{candidate_id, message, job_id?}` → grounded answer, citations, critic report |
| GET | `/api/chat/{candidate_id}/history` | |
| POST | `/api/resume/optimize` | suggestions (+ optional verified bullet rewrite) |
| POST | `/api/outreach/draft` | draft only; `sent` is always false |
| POST / GET / PATCH | `/api/followups[/{id}]` | reminders: `pending_approval → approved → done` |
| GET | `/api/evaluation/results` | experiment definitions + `Data/results/*.json` |
| GET | `/api/models/status`, `/api/models/metrics` | routing/availability; LLM call log summary |
| GET | `/api/dashboard/summary` | dashboard aggregates (`?candidate_id=`) |
| GET | `/api/annotations/status` | progress vs targets |
| GET / POST | `/api/annotations/relevance/{candidate_id}/pool`, `/api/annotations/relevance` | blind pooled judging |
| POST | `/api/admin/ingestion/run` | background: ingest → indexes → graph |
| POST | `/api/admin/candidates/import-sample` | import Kaggle/resume_corpus samples as candidate pool |
| POST | `/api/admin/evaluation/run` | background `scripts/evaluate.py` |
| GET | `/api/admin/jobs[/{job}]` | background job status/logs |

Errors: 404 unknown ids, 409 conflicting admin job / missing prerequisite, 413 file too large, 422 validation
(submitted values are not echoed), 503 indexes not built. CORS origins from `FRONTEND_ORIGIN`.
