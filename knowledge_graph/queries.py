"""Graph queries (backend-agnostic, built on GraphStore primitives).

Only these fixed queries are exposed to the API — there is no endpoint that
executes user-supplied Cypher.
"""
from __future__ import annotations

from typing import Optional

from knowledge_graph.entity_linking import candidate_node_id, job_node_id
from knowledge_graph.neo4j_client import ONTOLOGY, GraphStore


def _evidence(store: GraphStore, ids: list[str]) -> list[dict]:
    out = []
    for i in ids or []:
        n = store.node(i)
        if n:
            out.append({"evidence_id": i, "text": n.get("text"), "document_id": n.get("document_id"),
                        "start": n.get("start"), "end": n.get("end"), "extractor": n.get("extractor"),
                        "confidence": n.get("confidence")})
    return out


def candidate_skills(store: GraphStore, candidate_id: str) -> list[dict]:
    cid = candidate_node_id(candidate_id)
    out = []
    for nb in store.neighbors(cid, "HAS_SKILL"):
        out.append({"skill_id": nb["node"].get("canonical_id"), "name": nb["node"].get("name"),
                    "confidence": nb["edge"]["props"].get("confidence"),
                    "evidence": _evidence(store, nb["edge"]["props"].get("evidence_ids", []))})
    return sorted(out, key=lambda x: x["name"] or "")


def job_requirements(store: GraphStore, job_id: str) -> list[dict]:
    out = []
    for nb in store.neighbors(job_node_id(job_id), "HAS_REQUIREMENT"):
        req = nb["node"]
        skills = [x["node"] for x in store.neighbors(req["id"], "TARGETS")]
        out.append({"requirement_id": req["id"], "kind": req.get("kind"), "importance": req.get("importance"),
                    "text": req.get("text"), "years": req.get("years"),
                    "skill_id": skills[0].get("canonical_id") if skills else None,
                    "skill": skills[0].get("name") if skills else None,
                    "evidence": _evidence(store, nb["edge"]["props"].get("evidence_ids", []))})
    return out


def match(store: GraphStore, candidate_id: str, job_id: str) -> dict:
    cand = {s["skill_id"]: s for s in candidate_skills(store, candidate_id)}
    matched, missing = [], []
    for r in job_requirements(store, job_id):
        if r["kind"] != "skill" or not r["skill_id"]:
            continue
        if r["skill_id"] in cand:
            matched.append({**r, "candidate_evidence": cand[r["skill_id"]]["evidence"]})
        else:
            missing.append(r)
    return {"matched": matched, "missing": missing}


def transferable(store: GraphStore, candidate_id: str, job_id: str) -> list[dict]:
    cand = {s["skill_id"] for s in candidate_skills(store, candidate_id)}
    out = []
    for r in match(store, candidate_id, job_id)["missing"]:
        sid = f"skill:{r['skill_id']}"
        for direction in ("out", "in"):
            for e in store.edges(sid, "RELATED_TO", direction):
                other = e["dst"] if direction == "out" else e["src"]
                n = store.node(other)
                if n and n.get("canonical_id") in cand:
                    out.append({"missing_skill": r["skill"], "candidate_skill": n.get("name"),
                                "inferred": e["props"].get("inferred", True), "method": e["props"].get("method"),
                                "confidence": e["props"].get("confidence")})
    return out


def related_occupations(store: GraphStore, candidate_id: str, limit: int = 10) -> list[dict]:
    out, seen = [], set()
    for nb in store.neighbors(candidate_node_id(candidate_id), "TARGETS"):
        if nb["node"]["label"] != "Occupation":
            continue
        for r in store.neighbors(nb["node"]["id"], "RELATED_TO"):
            if r["node"]["id"] in seen:
                continue
            seen.add(r["node"]["id"])
            out.append({"code": r["node"].get("code"), "title": r["node"].get("title"),
                        "via": nb["node"].get("title"), "tier": r["edge"]["props"].get("tier"),
                        "method": r["edge"]["props"].get("method")})
    return out[:limit]


def explanation_paths(store: GraphStore, candidate_id: str, job_id: str) -> list[dict]:
    """Paths Candidate -> ... -> JobPosting with the evidence on every extracted edge."""
    m = match(store, candidate_id, job_id)
    paths = [{"type": "direct_skill", "skill": r["skill"],
              "path": ["Candidate", "HAS_SKILL", "Skill", "TARGETS⁻¹", "Requirement", "HAS_REQUIREMENT⁻¹", "JobPosting"],
              "candidate_evidence": r["candidate_evidence"], "job_evidence": r["evidence"]} for r in m["matched"]]
    seen = set()
    for t in transferable(store, candidate_id, job_id):
        key = (t["missing_skill"], t["candidate_skill"], t["method"])
        if key in seen:  # the same skill can be targeted by several requirement nodes of one posting
            continue
        seen.add(key)
        paths.append({"type": "related_skill_bridge", "skill": t["missing_skill"], "via": t["candidate_skill"],
                      "path": ["Candidate", "HAS_SKILL", "Skill", "RELATED_TO", "Skill", "TARGETS⁻¹", "Requirement",
                               "HAS_REQUIREMENT⁻¹", "JobPosting"],
                      "inferred": True, "method": t["method"], "confidence": t["confidence"]})
    cand_occ = {nb["node"]["id"] for nb in store.neighbors(candidate_node_id(candidate_id), "TARGETS")}
    for nb in store.neighbors(job_node_id(job_id), "FOR_ROLE"):
        if nb["node"]["id"] in cand_occ:
            paths.append({"type": "same_occupation", "occupation": nb["node"].get("title"),
                          "path": ["Candidate", "TARGETS", "Occupation", "FOR_ROLE⁻¹", "JobPosting"]})
    return paths


_HUBS = {"Skill", "Occupation", "Company", "Location"}


def subgraph(store: GraphStore, candidate_id: str, job_id: Optional[str] = None,
             labels: Optional[set[str]] = None, max_nodes: int = 150) -> dict:
    """Candidate neighbourhood (+ optional job) for visualisation."""
    start = [candidate_node_id(candidate_id)] + ([job_node_id(job_id)] if job_id else [])
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    frontier = list(start)
    depth = {s: 0 for s in start}
    job_reqs: set[str] = set()  # requirements of the selected job — the only Requirement nodes shown
    while frontier and len(nodes) < max_nodes:
        nid = frontier.pop(0)
        n = store.node(nid)
        if not n:
            continue
        if labels and n["label"] not in labels and nid not in start:
            continue
        # Skills and occupations are shared hubs: following them inward would pull in every other posting,
        # its requirements and other candidates, crowding out the two documents being compared.
        if nid not in start and (n["label"] in {"JobPosting", "Candidate"}
                                 or (n["label"] == "Requirement" and nid not in job_reqs)):
            continue
        nodes[nid] = n
        if job_id and nid == job_node_id(job_id):
            job_reqs.update(e["dst"] for e in store.edges(nid, "HAS_REQUIREMENT", "out"))
        if depth[nid] >= 2:
            continue
        for direction in ("out", "in"):
            for e in store.edges(nid, None, direction):
                other = e["dst"] if direction == "out" else e["src"]
                # keep job/candidate neighbourhoods focused
                if e["type"] in {"TYPICAL_FOR"} or (e["type"] == "RELATED_TO" and depth[nid] >= 1):
                    continue
                # inward edges into a shared hub come from other postings / requirements / candidates
                if direction == "in" and n["label"] in _HUBS and other not in depth and other not in job_reqs:
                    continue
                if other not in depth:
                    depth[other] = depth[nid] + 1
                    frontier.append(other)
                edges.append(e)
    ids = set(nodes)
    edges = [e for e in edges if e["src"] in ids and e["dst"] in ids]
    uniq = {(e["src"], e["type"], e["dst"]): e for e in edges}
    return {"nodes": list(nodes.values()), "edges": list(uniq.values())}


def evidence(store: GraphStore, evidence_id: str) -> Optional[dict]:
    r = _evidence(store, [evidence_id])
    return r[0] if r else None


def consistency_report(store: GraphStore, sample_limit: int = 20) -> dict:
    """Ontology conformance + provenance completeness of the stored graph."""
    rels = ONTOLOGY["relations"]
    total = violations = missing_ev = dangling = extracted = 0
    examples = []
    for e in store.all_edges():
        if e["type"] in ("SPO",):
            continue
        total += 1
        spec = rels.get(e["type"])
        s, d = store.node(e["src"]), store.node(e["dst"])
        if not spec or not s or not d or s["label"] not in spec["from"] or d["label"] not in spec["to"]:
            violations += 1
            if len(examples) < sample_limit:
                examples.append({"edge": e["type"], "src": s and s["label"], "dst": d and d["label"]})
        props = e.get("props") or {}
        if spec and spec.get("evidence_required") and not props.get("inferred") and props.get("source") not in ("source_tags", "user_assertion"):
            extracted += 1
            ids = props.get("evidence_ids") or []
            if not ids:
                missing_ev += 1
            elif any(store.node(i) is None for i in ids):
                dangling += 1
    return {"edges": total, "ontology_violations": violations,
            "ontology_conformance": round(1 - violations / total, 4) if total else None,
            "evidence_required_edges": extracted, "edges_missing_evidence": missing_ev,
            "edges_with_dangling_evidence": dangling,
            "evidence_completeness": round(1 - (missing_ev + dangling) / extracted, 4) if extracted else None,
            "violation_examples": examples}
