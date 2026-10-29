# LinkedIn Assistant — an explainable, open-weight LLM framework for job matching

Upload a resume and get ranked jobs with matched and missing skills. Every recommendation explains itself
with citations to the exact resume and job-posting text, and you can inspect it in a knowledge graph. The
same codebase is the research infrastructure for the four research questions: open-weight vs proprietary
models (RQ1), QLoRA vs zero-shot extraction (RQ2), ontology vs free-form graphs (RQ3), and graph features
plus grounded explanations (RQ4).

Status of every component: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). Design docs:
[architecture](docs/architecture.md) · [data](docs/data.md) · [ontology](docs/ontology.md) ·
[matching](docs/matching.md) · [LLMs & agents](docs/llm.md) · [evaluation](docs/evaluation.md) · [API](docs/api.md).

## Quick start (development mode, ~10 minutes)

```bash
# 1. Python 3.13 environment
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
brew install libomp                      # macOS only (LightGBM)

# 2. Configuration (all keys optional)
cp .env.example .env                     # set NEO4J_PASSWORD at least if you use Neo4j

# 3. Neo4j (optional — an in-memory graph is used if it is not reachable)
docker compose up -d neo4j               # browser UI: http://localhost:7474

# 4. Data → indexes → graph
python scripts/inspect_data.py           # manifests (Data/manifests/)
python scripts/ingest_data.py            # 5,000-job development sample; --full for everything
python scripts/build_indexes.py          # taxonomy, job parsing, BM25, TF-IDF, FAISS
python scripts/build_graph.py            # Neo4j if configured, else in-memory

# 5. Run
uvicorn backend.main:app --port 8000     # API + docs at http://localhost:8000/docs
streamlit run frontend/app.py            # UI at http://localhost:8501
```

Then open the UI, upload a resume on **Resume Analyzer**, and open **Job Recommendations**.

## Datasets

The raw datasets stay where you downloaded them; `Data/raw/` holds read-only copies. Details and computed
statistics: [docs/data.md](docs/data.md) and `Data/manifests/`.

| Source | Use |
|---|---|
| LinkedIn Job Postings 2023–24 (Kaggle) | main job corpus (123,849 postings) |
| DataStax LinkedIn listings | job corpus (1,048) |
| LinkedIn Job Posts Insights (XLSX) | analytics + dedup (no descriptions → not indexed) |
| Kaggle Resume Dataset (CSV + 2,484 PDFs, 24 categories) | candidate pool, evaluation, classifier baseline |
| resume_corpus (Jiechieu & Tsopze) | candidate pool, occupation labels |
| O*NET 31.0 | primary skill/occupation taxonomy, transferable skills, related occupations |
| OSMT import files | SOC hierarchy names |
| RemoteOK API | live remote jobs (attribution and source links preserved) |
| Adzuna API | live jobs when `ADZUNA_APP_ID`/`ADZUNA_APP_KEY` are set |

Development vs full mode: `configs/settings.yaml` (`modes.development.max_jobs: 5000`). Use
`APP_MODE=full` or `scripts/ingest_data.py --full` for all data (expect ~10× longer index builds).

## LLM providers

| Provider | Setup | Used for |
|---|---|---|
| Ollama (local) | `ollama serve && ollama pull qwen2.5:3b-instruct` | resume extraction, explanations, optimizer (resume text stays local) |
| Groq | `GROQ_API_KEY` | explanations, research tasks, benchmarks (never resume text) |
| OpenRouter | `OPENROUTER_API_KEY` | benchmarks |
| Proprietary baseline | `PROPRIETARY_BASE_URL` + `PROPRIETARY_API_KEY` (OpenAI-compatible) | RQ1 baseline |
| Lightcast | `LIGHTCAST_CLIENT_ID/SECRET` | optional; not required (O*NET + aliases + TF-IDF + embeddings are used) |

Model IDs live in `configs/models.yaml`; verify with `python scripts/check_providers.py --ping`.
Without any LLM the system still works end to end: classical extraction, template explanations, and a
rule-based assistant.

## Live job feeds

```bash
python scripts/ingest_data.py --fetch-live --adzuna-query "data scientist"
python scripts/build_indexes.py && python scripts/build_graph.py
```
RemoteOK jobs show "Remote OK" attribution and link back to the original posting, as its API terms require.

## Research workflow

```bash
python scripts/create_annotation_tasks.py   # extraction / normalisation / relevance / graph / explanation tasks
#   …annotate (or judge relevance in the app's Annotation page)…
python scripts/train_ltr.py                 # LambdaMART once ≥ 200 judgments exist
python scripts/evaluate.py                  # writes Data/results/*.json → Research Dashboard
```
Experiments without the labels or providers they need are recorded as **Not evaluated** / **Blocked**;
no numbers are invented. QLoRA (`training/qlora.py`) needs a CUDA GPU; see `requirements-qlora.txt`.

## Tests

```bash
pytest -q            # 87 tests: unit, API, two end-to-end flows, all Streamlit pages (AppTest), live Neo4j (if configured)
```
External APIs are mocked. Application state is written to a temporary directory; the real indexes under
`Data/` are read.

## Docker

```bash
docker compose up --build      # neo4j + backend (:8000) + frontend (:8501); Data/ is mounted into the backend
```
Ollama runs on the host; containers reach it at `host.docker.internal:11434`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OMP: Error #179` or segfault on macOS | `common/__init__.py` must be imported first (it sets `OMP_NUM_THREADS=1` and loads LightGBM first); use the provided scripts/entry points |
| `Library not loaded: libomp.dylib` | `brew install libomp` |
| API returns 503 "Required data not built" | run `ingest_data.py` then `build_indexes.py` |
| Graph backend shows `memory` | Neo4j unreachable or `NEO4J_*` unset — `docker compose up -d neo4j` |
| Explanations are always "template" | no explainer model available; start Ollama or set a hosted key |
| Upload rejected | only real PDF/DOCX/TXT content ≤ 5 MB is accepted (checked by content, not extension) |
| "Python quit unexpectedly" dialogs | crash reports from OpenMP conflicts in ad-hoc scripts that did not import `common` first — dismiss with OK |
