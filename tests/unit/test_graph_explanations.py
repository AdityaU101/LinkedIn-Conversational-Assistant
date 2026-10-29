"""Graph construction, provenance, queries; explanation verification."""
import pytest

from common.schemas import Job
from knowledge_graph import queries
from knowledge_graph.graph_builder import candidate_graph, job_graph, taxonomy_graph, upsert_candidate
from knowledge_graph.neo4j_client import MemoryGraphStore, _check_label
from parsing.classical_parser import ClassicalJobParser, ClassicalResumeParser


@pytest.fixture(scope="module")
def graph(resume_text, tmp_path_factory):
    store = MemoryGraphStore(tmp_path_factory.mktemp("g") / "g.pkl")
    cand = ClassicalResumeParser().parse(resume_text, "cand-g", "doc")
    text = "Requirements:\n- Python and SQL\n- Experience with TensorFlow\nNice to have: Kubernetes"
    job = ClassicalJobParser().parse(Job(job_id="t:9", source_job_id="9", source="t", title="Data Scientist",
                                         company="Acme", description=text))
    g = job_graph(job)
    g.merge(taxonomy_graph({r.skill.canonical_id for r in job.requirements if r.skill}, {job.occupation.code}))
    g.write(store)
    upsert_candidate(store, cand)
    return store, cand, job


def test_graph_counts_and_consistency(graph):
    store, cand, job = graph
    c = store.counts()
    assert c["nodes"]["Candidate"] == 1 and c["nodes"]["JobPosting"] == 1 and c["relationships"]["HAS_SKILL"] >= 5
    rep = queries.consistency_report(store)
    assert rep["ontology_violations"] == 0 and rep["evidence_completeness"] == 1.0


def test_idempotent_merge(graph):
    store, cand, job = graph
    before = store.counts()
    upsert_candidate(store, cand)
    job_graph(job).write(store)
    assert store.counts() == before


def test_queries_match_missing_transferable(graph):
    store, cand, job = graph
    m = queries.match(store, "cand-g", "t:9")
    assert "Python" in {r["skill"] for r in m["matched"]} and "Kubernetes" in {r["skill"] for r in m["missing"]}
    assert all(r["candidate_evidence"] and r["evidence"] for r in m["matched"])
    paths = queries.explanation_paths(store, "cand-g", "t:9")
    assert any(p["type"] == "direct_skill" for p in paths)
    sub = queries.subgraph(store, "cand-g", "t:9", max_nodes=60)
    assert sub["nodes"] and sub["edges"]


def test_delete_candidate_removes_owned_nodes(graph):
    store, cand, job = graph
    removed = store.delete_subject("cand:cand-g")
    assert removed > 1 and store.node("cand:cand-g") is None
    assert store.node("job:t:9") is not None   # shared job data untouched
    upsert_candidate(store, cand)


def test_label_validation_blocks_injection():
    with pytest.raises(ValueError):
        _check_label("Skill) DETACH DELETE (n")


# ------------------------------------------------------------------ explanations
def test_template_explanation_verifies_and_critic_rejects_hallucinations(graph):
    from explanations.llm_explainer import _template_numbers
    from explanations.template import template_explanation
    from explanations.verifier import verify
    from matching.skill_gap import compute_gap

    store, cand, job = graph
    gap = compute_gap(cand, job)
    exp = template_explanation(gap)
    v = verify(exp.text, exp.registry, _template_numbers(gap))
    assert v["faithfulness"] == 1.0 and v["citation_validity"] == 1.0 and v["hallucination_rate"] == 0.0
    py = next(i.cite_id for i in exp.registry.items if i.kind == "candidate" and "Python" in i.skill_names)
    bad = (f"You have 9 years of Kubernetes experience [{py}]. You are proficient in Rust. "
           f"Your Python work is relevant [{py}]. See also [C99].")
    v = verify(bad, exp.registry)
    sup = {s["sentence"][:20]: s["supported"] for s in v["sentences"]}
    assert sup["You have 9 years of "] is False and sup["You are proficient i"] is False
    assert sup["Your Python work is "] is True
    assert v["citation_validity"] < 1 and v["hallucination_rate"] > 0
    assert "Kubernetes" not in v["verified_text"]
