from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.services import resume_service
from knowledge_graph import queries
from knowledge_graph.entity_linking import candidate_node_id
from knowledge_graph.neo4j_client import LABELS, get_graph_store

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/status")
def status():
    store = get_graph_store()
    return store.counts()


@router.get("/evidence/{evidence_id}")
def evidence(evidence_id: str):
    ev = queries.evidence(get_graph_store(), evidence_id)
    if not ev:
        raise HTTPException(404, "Evidence not found")
    return ev


@router.get("/consistency")
def consistency():
    return queries.consistency_report(get_graph_store())


@router.get("/{candidate_id}")
def candidate_graph(candidate_id: str, job_id: str | None = None, labels: str | None = Query(None, max_length=300),
                    max_nodes: int = Query(150, ge=10, le=500)):
    store = get_graph_store()
    # a profile saved while the graph was unreachable is written on first view
    if store.node(candidate_node_id(candidate_id)) is None and not resume_service.ensure_in_graph(candidate_id):
        raise HTTPException(404, "Candidate not in graph")
    lab = {l for l in (labels or "").split(",") if l} or None
    if lab and not lab <= LABELS:
        raise HTTPException(422, f"Unknown labels: {sorted(lab - LABELS)}")
    sub = queries.subgraph(store, candidate_id, job_id, lab, max_nodes)
    out = {"backend": store.backend, **sub,
           "skills": queries.candidate_skills(store, candidate_id),
           "related_occupations": queries.related_occupations(store, candidate_id)}
    if job_id:
        out["match"] = queries.match(store, candidate_id, job_id)
        out["paths"] = queries.explanation_paths(store, candidate_id, job_id)
    return out
