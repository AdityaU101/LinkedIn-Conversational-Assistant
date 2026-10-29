"""End-to-end flows.

1. RESUME -> PARSE -> NORMALIZE -> MATCH -> EXPLAIN   (real indexes)
2. JOB DATA -> INGEST -> NORMALIZE -> GRAPH -> INDEX -> RETRIEVE   (temp directory, real DataStax file)
"""
import numpy as np
import pandas as pd

from tests.conftest import needs_indexes


@needs_indexes()
def test_resume_to_explanation(resume_text):
    from explanations.llm_explainer import explain
    from matching.pipeline import MatchingPipeline
    from parsing.resume_parser import parse_resume_text

    res = parse_resume_text(resume_text, "e2e-1", "fixture", use_llm=False)
    cand = res.candidate
    assert res.injection_flags                                   # injection text flagged, parsed as data
    assert all(s.canonical_id for s in cand.skills)             # normalised to taxonomy ids
    assert any(s.taxonomy == "onet" for s in cand.skills)
    out = MatchingPipeline().recommend(cand, k=5, use_cross_encoder=False)
    assert len(out.results) == 5 and out.ranker.startswith("fallback_weighted_features")
    for stage in ("bm25", "tfidf", "jaccard", "dense", "rrf"):
        assert out.stages.get(stage, 0) > 0
    top = out.results[0]
    assert {"bm25", "tfidf", "jaccard", "rrf"} <= set(top.component_scores)
    exp = explain(top.gap, prefer_llm=False)
    assert exp.verification["faithfulness"] in (1.0, None)
    for item in exp.registry.items:
        if item.kind == "candidate" and item.start is not None:
            assert resume_text[item.start:item.end] == item.text[: item.end - item.start]


def test_job_data_to_retrieval(tmp_path, monkeypatch):
    from common.config import get_settings
    from ingestion.pipeline import load_jobs, run
    from knowledge_graph.graph_builder import build_graph
    from knowledge_graph.neo4j_client import MemoryGraphStore
    from knowledge_graph.queries import consistency_report
    from matching.bm25 import BM25
    from matching.store import JobStore, save_store
    from parsing.job_parser import parse_jobs

    monkeypatch.setenv("LA_PATH_SPLITS", str(tmp_path / "splits"))
    monkeypatch.setenv("LA_PATH_CACHE", str(tmp_path / "cache"))
    report = run(max_jobs=None, sources=["datastax_linkedin", "remoteok"], out_dir=tmp_path, progress=lambda m: None)
    assert report["records"] >= 1000 and report["leakage_check"] == "ok"
    jobs = load_jobs(tmp_path / "jobs.parquet")[:150]
    parsed = parse_jobs(jobs, use_cache=False)
    assert sum(1 for j in parsed if j.requirement_skill_ids()) > 100
    store = MemoryGraphStore()
    build_graph(store, parsed, progress=None)
    rep = consistency_report(store)
    assert rep["ontology_violations"] == 0 and rep["evidence_completeness"] == 1.0
    p = save_store(parsed, tmp_path / "idx.parquet")
    js = JobStore(pd.read_parquet(p))
    bm = BM25().fit(js.texts())
    target = next(j for j in parsed if j.title and len(j.title.split()) >= 2)
    best = int(np.argmax(bm.scores(target.title + " " + (target.description or "")[:300])))
    assert js.ids[best] == target.job_id
