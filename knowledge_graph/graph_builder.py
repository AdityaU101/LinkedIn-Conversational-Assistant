"""Build ontology-constrained, evidence-grounded graph data.

Every relation derived from an extracted fact carries ``evidence_ids`` and the
subject node has SUPPORTED_BY edges to the Evidence nodes. Reference/inferred
relations carry ``inferred``, ``method`` and ``confidence``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from common.schemas import Candidate, Evidence, Job
from knowledge_graph.entity_linking import (
    candidate_node_id, company_node_id, fact_node_id, job_node_id, location_node_id, occupation_node_id, skill_node_id,
)
from knowledge_graph.evidence import evidence_node
from knowledge_graph.neo4j_client import GraphStore
from normalization.taxonomy import Taxonomy, load_taxonomy


class GraphBatch:
    def __init__(self):
        self.nodes: dict[str, dict[str, dict]] = defaultdict(dict)
        self.edges: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)

    def node(self, label: str, **props) -> str:
        cur = self.nodes[label].get(props["id"])
        if cur:
            cur.update({k: v for k, v in props.items() if v is not None})
        else:
            self.nodes[label][props["id"]] = props
        return props["id"]

    def edge(self, rel: str, src: str, dst: str, **props) -> None:
        key = (src, dst)
        cur = self.edges[rel].get(key)
        if cur:
            if "evidence_ids" in props:
                props["evidence_ids"] = sorted(set(cur["props"].get("evidence_ids", [])) | set(props["evidence_ids"]))
            cur["props"].update(props)
        else:
            self.edges[rel][key] = {"src": src, "dst": dst, "props": props}

    def evidence(self, owner: str, evs: Iterable[Evidence]) -> list[str]:
        ids = []
        for ev in evs:
            self.node("Evidence", **evidence_node(ev))
            self.edge("SUPPORTED_BY", owner, ev.evidence_id)
            ids.append(ev.evidence_id)
        return ids

    def skill(self, canonical_id: str, tax: Taxonomy) -> str:
        info = tax.skill_by_id.get(canonical_id, {})
        return self.node("Skill", id=skill_node_id(canonical_id), canonical_id=canonical_id,
                         name=info.get("name", canonical_id), taxonomy=info.get("taxonomy"),
                         skill_type=info.get("skill_type"), hot_technology=bool(info.get("hot_technology", False)))

    def occupation(self, code: str, tax: Taxonomy) -> str:
        info = tax.occ_by_code.get(code, {})
        return self.node("Occupation", id=occupation_node_id(code), code=code, title=info.get("title", code),
                         major_group=code[:2])

    def merge(self, other: "GraphBatch") -> None:
        for label, d in other.nodes.items():
            for n in d.values():
                self.node(label, **n)
        for rel, d in other.edges.items():
            for e in d.values():
                self.edge(rel, e["src"], e["dst"], **e["props"])

    def write(self, store: GraphStore) -> dict:
        order = ["Evidence", "Skill", "Occupation", "Company", "Location", "Candidate", "JobPosting", "Requirement",
                 "Experience", "Credential", "Education", "Project"]
        for label in order + [l for l in self.nodes if l not in order]:
            if self.nodes.get(label):
                store.merge_nodes(label, list(self.nodes[label].values()))
        for rel, d in self.edges.items():
            store.merge_edges(rel, list(d.values()))
        return {"nodes": {l: len(d) for l, d in self.nodes.items()}, "edges": {r: len(d) for r, d in self.edges.items()}}


def candidate_graph(c: Candidate, tax: Optional[Taxonomy] = None) -> GraphBatch:
    tax = tax or load_taxonomy()
    g = GraphBatch()
    cid = g.node("Candidate", id=candidate_node_id(c.candidate_id), candidate_id=c.candidate_id,
                 source_document=c.source_document, total_experience_years=c.total_experience_years,
                 extractor=c.extractor, user_corrected=c.user_corrected)
    for s in c.skills:
        if not s.canonical_id:
            continue
        sid = g.skill(s.canonical_id, tax)
        ev = g.evidence(cid, s.evidence)
        g.edge("HAS_SKILL", cid, sid, evidence_ids=ev, confidence=s.confidence, method=s.method,
               mention=s.mention, inferred=False,
               source="user_assertion" if (s.method == "user" and not ev) else "resume")
    for o in c.occupations:
        if o.code:
            ev = g.evidence(cid, o.evidence)
            g.edge("TARGETS", cid, g.occupation(o.code, tax), evidence_ids=ev, method=o.method,
                   confidence=o.confidence, inferred=False)
    for i, e in enumerate(c.experiences):
        xid = g.node("Experience", id=fact_node_id("exp", c.candidate_id, i, e.title, e.start_date), title=e.title,
                     start_date=e.start_date, end_date=e.end_date, duration_months=e.duration_months)
        ev = g.evidence(xid, e.evidence)
        g.edge("HELD", cid, xid, evidence_ids=ev, inferred=False)
        comp = company_node_id(e.company)
        if comp:
            g.node("Company", id=comp, name=e.company)
            g.edge("AT", xid, comp, evidence_ids=ev, inferred=False)
        if e.occupation and e.occupation.code:
            g.edge("FOR_ROLE", xid, g.occupation(e.occupation.code, tax), method=e.occupation.method,
                   confidence=e.occupation.confidence, evidence_ids=ev, inferred=False)
        for s in e.skills:
            if s.canonical_id:
                sev = g.evidence(xid, s.evidence)
                g.edge("USED", xid, g.skill(s.canonical_id, tax), evidence_ids=sev, inferred=False)
    for i, ed in enumerate(c.education):
        eid = g.node("Education", id=fact_node_id("edu", c.candidate_id, i, ed.degree), degree=ed.degree,
                     level=ed.level, field=ed.field, institution=ed.institution)
        g.edge("HAS_EDUCATION", cid, eid, evidence_ids=g.evidence(eid, ed.evidence), inferred=False)
    for i, cr in enumerate(c.credentials):
        crid = g.node("Credential", id=fact_node_id("cred", c.candidate_id, i, cr.name), name=cr.name)
        g.edge("HAS_CREDENTIAL", cid, crid, evidence_ids=g.evidence(crid, cr.evidence), inferred=False)
    for i, p in enumerate(c.projects):
        pid = g.node("Project", id=fact_node_id("proj", c.candidate_id, i, p.name), name=p.name)
        g.edge("WORKED_ON", cid, pid, evidence_ids=g.evidence(pid, p.evidence), inferred=False)
        for s in p.skills:
            if s.canonical_id:
                g.edge("USED", pid, g.skill(s.canonical_id, tax), evidence_ids=g.evidence(pid, s.evidence), inferred=False)
    return g


def job_graph(job: Job, tax: Optional[Taxonomy] = None, include_responsibilities: bool = False) -> GraphBatch:
    tax = tax or load_taxonomy()
    g = GraphBatch()
    jid = g.node("JobPosting", id=job_node_id(job.job_id), job_id=job.job_id, title=job.title, source=job.source,
                 source_url=job.source_url, posting_date=job.posting_date, work_mode=job.work_mode,
                 seniority=job.seniority, employment_type=job.employment_type)
    comp = company_node_id(job.company)
    if comp:
        g.node("Company", id=comp, name=job.company)
        g.edge("POSTED_BY", jid, comp, method="source_metadata", inferred=False)
    loc = location_node_id(job.location_city, job.location_state, job.location_country)
    if loc:
        g.node("Location", id=loc, name=job.location, state=job.location_state, country=job.location_country)
        g.edge("LOCATED_IN", jid, loc, method="source_metadata", inferred=False)
    if job.occupation and job.occupation.code:
        g.edge("FOR_ROLE", jid, g.occupation(job.occupation.code, tax), method=job.occupation.method,
               confidence=job.occupation.confidence, inferred=False)
    for r in job.requirements:
        if r.kind == "responsibility" and not include_responsibilities:
            continue
        rid = g.node("Requirement", id=r.requirement_id or fact_node_id("req", job.job_id, r.text), kind=r.kind,
                     importance=r.importance, text=r.text[:300], years=r.years, education_level=r.education_level)
        ev = g.evidence(rid, r.evidence)
        g.edge("HAS_REQUIREMENT", jid, rid, evidence_ids=ev, inferred=False,
               source="source_tags" if not ev else "description")
        if r.skill and r.skill.canonical_id:
            g.edge("TARGETS", rid, g.skill(r.skill.canonical_id, tax), evidence_ids=ev, method=r.skill.method,
                   confidence=r.skill.confidence, inferred=False)
    return g


def taxonomy_graph(skill_ids: set[str], occ_codes: set[str], tax: Optional[Taxonomy] = None,
                   related_occ_limit: int = 10, expand: bool = False) -> GraphBatch:
    """Reference/inferred edges among the skills and occupations in use."""
    tax = tax or load_taxonomy()
    g = GraphBatch()
    for sid in skill_ids:
        g.skill(sid, tax)
        for other, conf in tax.related_skills.get(sid, []):
            if (other in skill_ids and sid < other) or (expand and other not in skill_ids):
                if expand and other not in skill_ids:
                    g.skill(other, tax)
                g.edge("RELATED_TO", skill_node_id(sid), skill_node_id(other), inferred=True,
                       method=tax.relation_method.get((sid, other), "related"), confidence=conf)
    rel = tax.related_occupations
    rel = rel[rel["code"].isin(occ_codes)]
    extra = set()
    for code, grp in rel.groupby("code"):
        for r in grp.sort_values("rank").head(related_occ_limit).itertuples():
            g.occupation(code, tax)
            g.occupation(r.related_code, tax)
            extra.add(r.related_code)
            g.edge("RELATED_TO", occupation_node_id(code), occupation_node_id(r.related_code), inferred=False,
                   method="onet_related_occupations", tier=r.tier, confidence=1.0 / (1 + int(r.rank) / 10))
    for code in occ_codes | extra:
        g.occupation(code, tax)
        for sid, w in tax.skills_by_occ.get(code, {}).items():
            if sid in skill_ids:
                g.edge("TYPICAL_FOR", skill_node_id(sid), occupation_node_id(code), inferred=False,
                       method="onet_reference", confidence=w)
    return g


def build_graph(store: GraphStore, jobs: list[Job], candidates: list[Candidate] = (), progress=print) -> dict:
    tax = load_taxonomy()
    total = GraphBatch()
    skills, occs = set(), set()
    for i, j in enumerate(jobs):
        total.merge(job_graph(j, tax))
        skills |= {r.skill.canonical_id for r in j.requirements if r.skill and r.skill.canonical_id}
        if j.occupation and j.occupation.code:
            occs.add(j.occupation.code)
        if progress and i % 1000 == 0:
            progress(f"graph: {i}/{len(jobs)} jobs")
    for c in candidates:
        total.merge(candidate_graph(c, tax))
        skills |= {s.canonical_id for s in c.skills if s.canonical_id}
        occs |= {o.code for o in c.occupations if o.code}
    total.merge(taxonomy_graph(skills, occs, tax))
    stats = total.write(store)
    if hasattr(store, "save"):
        store.save()
    return stats


def upsert_candidate(store: GraphStore, c: Candidate) -> dict:
    """Idempotent: removes the candidate's previous facts, then merges the new ones."""
    tax = load_taxonomy()
    store.delete_subject(candidate_node_id(c.candidate_id))
    g = candidate_graph(c, tax)
    skills = {s.canonical_id for s in c.skills if s.canonical_id}
    occs = {o.code for o in c.occupations if o.code}
    g.merge(taxonomy_graph(skills, occs, tax, related_occ_limit=5, expand=True))
    stats = g.write(store)
    if hasattr(store, "save"):
        store.save()
    return stats
