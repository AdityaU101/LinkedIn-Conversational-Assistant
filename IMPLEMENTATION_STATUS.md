# Implementation status

Legend: **TESTED** = implemented and covered by passing automated tests (`pytest -q`: 87 passed, 2026-10-08) ·
**IMPLEMENTED** = works and was run, but has no automated test · **BLOCKED** = needs something not
available here (stated) · **NOT STARTED**.

Environment used: macOS 26 / Apple M3 / 8 GB RAM, Python 3.13.7, Neo4j 5.26 (Docker), no CUDA GPU,
no Ollama, no hosted LLM keys, no Lightcast credentials. Adzuna credentials added 2026-10-08.

## Blockers (what is needed to unblock)

| Item | Needs |
|---|---|
| Live LLM extraction / LLM explanations / LLM SPO / bullet rewriting | a local model: `ollama serve && ollama pull qwen2.5:3b-instruct` (code paths are tested with mocked providers) |
| Experiment G (providers, RQ1) | Ollama running and/or `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `PROPRIETARY_*` |
| Experiment A (extraction P/R/F1) | human span annotations for the 150 documents in `Data/annotations/extraction/tasks.json` |
| Experiment D (nDCG/MRR/P@5) + trained LambdaMART | ≥ 200 relevance judgments (798 pooled pairs over 40 profiles are ready in `Data/annotations/relevance/pool.csv` and in the app's Annotation page) |
| Skill-normalisation gold / edge precision (Exp B) | labels in `normalization/gold.csv` and `graph/edge_validation.csv` |
| Human explanation faithfulness (Exp F) | labels in `explanations/faithfulness.csv` |
| QLoRA training / Experiment E | CUDA GPU + `requirements-qlora.txt` + teacher labels (`training/teacher_labels.py` needs a provider) |
| User study | participants |

## Phases

| # | Phase | Status | Notes / evidence |
|---|---|---|---|
| 0 | Inspect workspace, data, proposal | IMPLEMENTED | No proposal PDF was found in the workspace; the task brief was used as the specification. Prior scaffold backed up to `.prior_scaffold_backup.tar.gz` and replaced (its manifest counts were wrong, e.g. 3.38M vs 123,849 postings). |
| 1 | Data/ organisation + manifests | TESTED | `scripts/inspect_data.py`; originals untouched; `Data/raw` files read-only; `test_manifest_matches_files` |
| 2 | Architecture, deps, FastAPI, Streamlit, Docker | TESTED (app) / IMPLEMENTED (Docker) | Neo4j container runs; backend/frontend Dockerfiles written but images **not built** here |
| 3 | Canonical schemas + ingestion | TESTED | all loaders (CSV chunked, XLSX, ZIP, JSON, O*NET), normalisers, dedup (hash/key/MinHash), leakage-safe splits. Full mode (`--full`) implemented but only development mode was run. Adzuna live fetch run for real: 800 US jobs (8 queries), 394 indexed after dedup |
| 4 | Resume / job parsing | TESTED | classical (sections, regex, gazetteer, spaCy ORG); LLM ontology-constrained extraction with grounding — tested with mocked provider, **live run BLOCKED** (no local model) |
| 5 | Skill / occupation normalisation | TESTED | O*NET 31.0 + local supplement (153 concepts) + 1,166 aliases, TF-IDF, embeddings, optional LLM tie-break (off). **Lightcast: NOT STARTED** (optional; no credentials to build/test against) |
| 6 | BM25 / TF-IDF / Jaccard (+ filters, LR family baseline) | TESTED | BM25 verified against hand computation |
| 7 | Neo4j ontology/evidence graph | TESTED | in-memory backend in unit tests; live Neo4j in `tests/integration/test_neo4j.py` (runs when configured — passed). Dev graph: 133,638 edges, 0 ontology violations, 100% evidence completeness |
| 8 | Sentence-Transformers + FAISS | TESTED | persisted `IndexIDMap2`, incremental add/remove |
| 9 | RRF | TESTED | component ranks preserved |
| 10 | LambdaMART infrastructure | TESTED / **model BLOCKED** | training + CV + `scripts/train_ltr.py` tested; no model trained (no judgments); documented fallback ranker in use |
| 11 | Cross-encoder + MMR | TESTED | real cross-encoder test + MMR unit test; job→candidate ranking via API |
| 12 | Evidence-grounded explanations + verifier | TESTED | template explanations; LLM explanation path implemented, **live BLOCKED** |
| 13 | Model router | TESTED | PII routing, fallback, retries, cost=null when unknown, JSON repair, call logging (mocked providers) |
| 14 | LangGraph agents (7) | TESTED | via API tests of chat/optimize/outreach/follow-ups |
| 15 | Career assistant / profile optimiser | TESTED | 8 intents grounded with citations; rewrites rejected if they add skills/numbers |
| 16 | QLoRA infrastructure | IMPLEMENTED / **training BLOCKED** | `training/qlora.py`, `teacher_labels.py`; status recorded as blocked |
| 17 | Annotation infrastructure | TESTED | task files generated (150 docs, 500 mentions, 798 pairs, 100 edges, 60 explanations), blind pooling, agreement stats, relevance UI |
| 18 | Evaluation / ablation framework | IMPLEMENTED (executed) | all experiments run; results in `Data/results/`; metric functions tested |
| 19 | Frontend pages (7 + annotation) | TESTED | every page renders against the real backend (Streamlit AppTest); live servers verified over HTTP. Click-level flows were walked through in a real browser (Playwright, 2026-10-08) — manually, not as automated tests |
| 20 | Research dashboard | TESTED | shows "Not evaluated" for missing metrics |
| 21 | Integration tests | TESTED | resume→parse→normalise→match→explain; job data→ingest→normalise→graph→index→retrieve |
| 22 | Documentation | IMPLEMENTED | README + docs/*.md |

## Research results recorded (development data)

See `docs/evaluation.md` and `Data/results/*.json`. Evaluated with real labels: occupation-family LR,
resume-category LR, occupation normalisation (O*NET held-out titles), graph structure (C), template
explanation verification (F, automatic). Everything else: Not evaluated / Blocked as listed above.

## Fixes from the browser walkthrough (2026-10-08)

* **PDF text extraction**: pypdf emits one word per line for Google-Docs PDFs; detected and re-extracted with
  pdfplumber (`parsing/text_extraction.py`). Before: an uploaded resume parsed with 0 roles.
* **Resume parser 1.2** (`RESUME_VERSION`; job parser and its cache unchanged at 1.1): `Title | Company | City | Dates`
  role lines, `|`-separated certifications, degree→institution search on the layout's side only (no school shared
  by two degrees), wrapped project lines/demo links, comma-qualified titles (`Software Engineer Intern, AI …`), and
  `<Title> with N+ years` summaries. Re-parse of the 40 stored Kaggle profiles: 12 changed, all institution fixes.
* **Re-parse** of a stored upload: `POST /api/resumes/{id}/reparse` + Resume Analyzer button; refuses (409) to
  overwrite a user-corrected profile unless forced.
* **Knowledge-graph view** no longer expands through shared Skill/Occupation hubs into unrelated postings;
  explanation paths de-duplicated.
* **Career assistant**: requests to run queries/commands, delete data or send to an address get an explicit
  "can't do that" answer instead of being routed by keyword (e.g. to outreach). No such action was ever possible.
* **DataStax redacted fields** (`**** ****`): treated as unknown; 52 redacted-title postings left the index and
  3 readable twins re-entered (4,690 → 4,641). Splits/clusters unchanged (0 of 10,447 rows moved).
  `Data/results/*.json` were computed on the 4,690-job index — re-run `python scripts/evaluate.py` to refresh.
  Unlabelled annotation tasks referencing removed postings: 2 normalisation mentions, 1 graph edge (still judgeable
  from their text).
* **Neo4j**: the container had exited (code 137, not OOM-killed) while `/health` still reported `neo4j`; health
  reports the backend chosen at startup, not live connectivity.

## Second pass (2026-10-09): education parsing, local LLM, silver evaluations

* **Education parser** (resume parser 1.3): entries are confined to their own block; the school/degree layout is
  chosen by which reading pairs more degrees with schools (ties: smaller gaps); field from the degree's own line
  (`M.S. Computer Science`), year = nearest date to the degree (a range counts as its end); adjacent education
  sections rejoined; bare `Master`/`Bachelor` only as standalone words; more degree forms (M.Ed, B.F.A, LLB …);
  institution names need a name around the keyword. Checked on the 40 stored Kaggle profiles (all changes
  inspected) and 300 random Kaggle resumes; synthetic layout tests in `tests/unit/test_parsing_normalization.py`.
* **Projects**: a line ending in a date range is a project title (fixes a title starting 'Distributed …').
* **Lexicon**: curated `Java` (case-sensitive), LightGBM, Streamlit, Next.js, FastAPI, Google Colab, D3.js,
  Matplotlib, OpenCV; capitalised forms of lowercase tool names ('Pandas') in skill lists. Job parser 1.2 → all
  4,641 postings re-parsed, indexes and graph rebuilt.
* **Local LLM installed**: Ollama 0.40.1 + `qwen2.5:3b-instruct`. Real runs exposed and fixed: a crash in the JSON
  repair call (duplicate `provider` argument); a whole extraction rejected because some items lacked an evidence
  quote (now dropped individually and counted as `rejected_invalid`); placeholder / ungrounded secondary values
  ('Not specified' companies) now dropped; LLM facts overlapping classical facts (a project re-read as a job) no
  longer merged in.
* **Privacy fix**: the `explainer` task quotes resume text but was `pii: false` with Groq as fallback; it (and
  `tiebreak`) is now `pii: true` → local providers only.
* **Silver evaluations** (`evaluation/silver.py`), reported separately from human-label metrics and never used to
  train LambdaMART: resume_corpus expert occupation labels → (A) occupation extraction P/R/F1, (D) silver relevance
  (job title matches the resume's expert class) for every ranking method, (C) ranking impact of graph features.
  Crosswalk class → O*NET-SOC = documented base ∪ occupations O*NET lists the class name under. Parser heuristics
  for flattened resumes were developed on a dev sample (seed 7) disjoint from the evaluation sample (seed 23).

## Known limitations

* Fuzzy title→exact O*NET code mapping is hard (hybrid top-1 0.295 on held-out titles); the UI shows the
  method/confidence and users can correct it.
* The gazetteer skill extractor favours precision (case and list-context rules); recall on free-form skill
  phrasing is limited until LLM extraction is available.
* LLM-based components are verified only with mocked providers in this environment.
* The bundled `rank-bm25` package is not used by the app (own vectorised BM25).
* OpenMP runs single-threaded (`OMP_NUM_THREADS=1`) to avoid the macOS multi-runtime crash; heavy CV jobs are slower.

## Reproduce

```bash
python scripts/inspect_data.py && python scripts/ingest_data.py && python scripts/build_indexes.py \
  && python scripts/build_graph.py && python scripts/evaluate.py && pytest -q
```
