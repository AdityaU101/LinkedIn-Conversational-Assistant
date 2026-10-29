"""Optional: runs against a live Neo4j when NEO4J_URI/USERNAME/PASSWORD are set and reachable.

Writes only nodes whose ids/documents carry a unique test prefix and deletes exactly those afterwards;
shared Skill/Occupation nodes are merged (idempotent) but never deleted.
"""
import uuid

import pytest

from common.config import env


@pytest.fixture(scope="module")
def neo():
    if not (env("NEO4J_URI") and env("NEO4J_USERNAME") and env("NEO4J_PASSWORD")):
        pytest.skip("Neo4j not configured")
    from knowledge_graph.neo4j_client import Neo4jGraphStore

    try:
        store = Neo4jGraphStore(env("NEO4J_URI"), env("NEO4J_USERNAME"), env("NEO4J_PASSWORD"))
    except Exception as e:
        pytest.skip(f"Neo4j unreachable: {type(e).__name__}")
    prefix = f"pytest-{uuid.uuid4().hex[:8]}"
    yield store, prefix
    store._run("MATCH (n:Node) WHERE n.id STARTS WITH $c OR n.id STARTS WITH $j OR n.document_id STARTS WITH $p "
               "OR n.id STARTS WITH $r DETACH DELETE n", c=f"cand:{prefix}", j=f"job:{prefix}", p=prefix,
               r=f"req:{prefix}")
    store.close()


def test_neo4j_merge_queries_idempotent(neo, resume_text):
    from common.schemas import Job
    from knowledge_graph import queries
    from knowledge_graph.graph_builder import job_graph, upsert_candidate
    from parsing.classical_parser import ClassicalJobParser, ClassicalResumeParser

    store, prefix = neo
    cand = ClassicalResumeParser().parse(resume_text, f"{prefix}-c", f"{prefix}-doc")
    job = ClassicalJobParser().parse(Job(job_id=f"{prefix}:1", source_job_id="1", source=prefix, title="Data Scientist",
                                         description="Requirements: Python, SQL and Kubernetes."))
    job_graph(job).write(store)
    upsert_candidate(store, cand)
    m = queries.match(store, f"{prefix}-c", f"{prefix}:1")
    assert {"Python", "Structured query language SQL"} <= {r["skill"] for r in m["matched"]}
    assert "Kubernetes" in {r["skill"] for r in m["missing"]}
    assert all(r["candidate_evidence"] for r in m["matched"])
    n1 = len(store.edges(f"cand:{prefix}-c", "HAS_SKILL"))
    upsert_candidate(store, cand)
    assert len(store.edges(f"cand:{prefix}-c", "HAS_SKILL")) == n1
    assert store.delete_subject(f"cand:{prefix}-c") > 1 and store.node(f"cand:{prefix}-c") is None
