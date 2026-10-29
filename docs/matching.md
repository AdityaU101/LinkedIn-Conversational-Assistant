# Matching

`matching/pipeline.py` — every stage is independently callable (`method=` bm25 | tfidf | jaccard | dense |
rrf | ltr | final) and every result keeps per-component scores and ranks.

| Stage | Implementation |
|---|---|
| Filters | location substring, work mode, seniority, employment type, source, min salary (known salaries only), title, required skills, posted after, max required years (`matching/store.py`) |
| BM25 | own vectorised Okapi BM25 (k1 = 1.5, b = 0.75) over a sparse matrix; verified against a hand computation |
| TF-IDF | word 1–2-grams, sublinear tf, cosine |
| Jaccard | normalised skill-ID sets with an inverted index |
| Dense | `all-MiniLM-L6-v2` (MPS/CPU), FAISS `IndexIDMap2(IndexFlatIP)`, persisted; `--add-new` for incremental updates |
| RRF | k = 60 over the four top-200 lists; component ranks preserved |
| Features (15) | bm25, tfidf, jaccard, dense, rrf, required/preferred coverage, missing required, #required, occupation similarity (O*NET code / SOC group / related occupations), experience gap, education compatibility, graph inverse shortest path, direct skill paths, RELATED_TO bridges |
| LambdaMART | LightGBM `lambdarank`, GroupKFold by candidate; trains only on `Data/annotations/relevance/judgments.csv` (≥ 200 judgments, ≥ 10 candidates) |
| Fallback ranker | documented hand-set weights (`FALLBACK_WEIGHTS`): 0.25 RRF + 0.30 smoothed required coverage (matched / (#required + 2), so postings with one or two extracted requirements — e.g. Adzuna snippets — cannot get full credit from a single match) + 0.05 preferred coverage + 0.15 dense + 0.10 occupation similarity + 0.05 education + 0.05 experience fit + 0.05 bridges. Not learned. |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` on top-20; blended 50/50 with the ranker score (min-max) |
| MMR | λ = 0.7 over dense job vectors → top-10 |

Job → candidates: `MatchingPipeline.recommend_candidates` (`POST /api/matching/job-candidates`) ranks stored
profiles (uploads and imported Kaggle/resume_corpus samples) with the same retrievers + RRF + coverage.

## Skill gap (`matching/skill_gap.py`)

Matched / missing required / missing preferred by canonical skill ID; transferable = missing skills with a
RELATED_TO bridge from a candidate skill (shown with method and confidence); occupation-level transfer from
O*NET importance ratings; related occupations from O*NET.

## Occupation-family baseline

`matching/occupation_family.py`: TF-IDF + logistic regression predicting the SOC major group, trained on
O*NET titles/descriptions/tasks and evaluated on 7,986 held-out O*NET alternate titles.

## Thresholds

Occupation fuzzy thresholds (TF-IDF 0.65, embedding 0.70) were chosen on a dev sample of O*NET titles
(seed 7); Experiment B reports on a different sample (seed 13). Fuzzy title→exact-O*NET-code precision is
modest (≈ 0.40–0.50 on dev), which is why the UI shows the method and confidence and lets users correct it.
