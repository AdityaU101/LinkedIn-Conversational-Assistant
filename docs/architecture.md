# Architecture

```
Streamlit (frontend/)  ──HTTP──▶  FastAPI (backend/)  ──▶  services ──▶  domain packages
   8 pages, no business logic        routers, validation,          ingestion · parsing · normalization
   api_client.py                     uploads, errors               knowledge_graph · matching
                                                                   explanations · agents · llm
                                                                   evaluation · annotation · training
Storage:  SQLite (Data/app.db) — profiles, resumes, match runs, follow-ups, chat, LLM call log
          Neo4j (or in-memory fallback) — ontology graph with Evidence nodes
          FAISS (Data/indexes/faiss_jobs.index) — job embeddings
          Parquet/JSON — canonical jobs, taxonomy, results
```

## Request flow (end-to-end acceptance path)

1. `POST /api/resumes/upload` → content-type sniffing (magic bytes), 5 MB limit, server-generated filename,
   `chmod 600` → `parsing/text_extraction.py` (pypdf → pdfplumber fallback, python-docx, txt).
2. `parsing/resume_parser.py` → `ClassicalResumeParser` (sections + regex + taxonomy gazetteer + spaCy ORG)
   and, when a **local** model is reachable, `LLMParser` (ontology-constrained JSON, grounded evidence).
3. Skills → `normalization/skill_normalizer.py` (O*NET + local supplement); titles →
   `normalization/occupation_normalizer.py` (O*NET-SOC).
4. `knowledge_graph/graph_builder.upsert_candidate` → Neo4j (idempotent `MERGE`, Evidence nodes).
5. `POST /api/matching/recommend` → `matching/pipeline.py`:
   filters → BM25 / TF-IDF / Jaccard / Dense(FAISS) top-200 → RRF → 15 features → LambdaMART
   (or documented fallback) → cross-encoder on top-20 → MMR → top-10, each with a `SkillGap`.
6. `explanations/` builds an evidence registry ([C#] resume spans, [J#] posting spans, [G#] inferred
   relations), writes a template or citation-constrained LLM explanation, and the verifier removes
   unsupported sentences.
7. The frontend renders cards, gaps, evidence and the graph.

## Agents (LangGraph, `agents/graph.py`)

`parser → kg_builder → route → {intent → matching → compose | profile_optimizer | research_outreach |
scheduler} → critic → END`. State is a `TypedDict` (`agents/state.py`). The critic runs on every
answer. Outreach drafts and follow-ups end with `requires_approval=True`; nothing is sent.

## Robustness / fallbacks

| Component | When unavailable | Behaviour |
|---|---|---|
| Neo4j | no credentials / unreachable | in-memory graph store (same API, pickled to `Data/indexes/graph_memory.pkl`) |
| Ollama / hosted LLMs | not running / no key | classical extraction, template explanations, rule-based assistant |
| Embedding model | offline | dense retrieval and embedding normalisation skipped |
| Cross-encoder | offline / disabled | stage skipped (reported in `ranker`) |
| LambdaMART | < 200 human judgments | documented weighted-feature fallback |
| Adzuna / Lightcast | no credentials | skipped; O*NET is the taxonomy |

## macOS OpenMP note

LightGBM (Homebrew libomp), FAISS, scikit-learn and PyTorch each ship an OpenMP runtime; together they
segfault. `common/__init__.py` imports LightGBM first and sets `OMP_NUM_THREADS=1` (verified fix). Every
entry point imports `common` first.

## Security & privacy

* Resume text only goes to providers marked `local: true` (task `pii: true` in `configs/models.yaml`).
* Document text is wrapped as delimited, neutralised data (`common/privacy.wrap_untrusted`); injection
  phrases are flagged. The initial patterns flagged 45 of the 2,484 Kaggle resumes ("act as a liaison");
  after tightening, 0 are flagged — but the patterns were tuned on that corpus, so this is not an
  independent false-positive estimate. Flags are informational: text is always treated as data.
* Only fixed, parameterised Cypher; labels/relationship types are whitelisted from the ontology.
  No endpoint executes user-supplied Cypher. SQLite uses `?` parameters only.
* Logs pass through a PII-redacting filter (emails, phones, URLs, addresses).
* `DELETE /api/resumes/{id}` removes the file, DB rows, match runs, chat history and graph nodes.
