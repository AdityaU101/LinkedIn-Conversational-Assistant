"""Experiment C — ontology-constrained graph vs free-form SPO graph (same documents).

Structural metrics (no labels needed):
  duplicate_node_rate      share of entity nodes that collapse onto another node after light surface
                           normalisation (case, punctuation, plural) — i.e. redundant nodes
  entity_fragmentation     mean number of distinct entity nodes per taxonomy concept, over concepts
                           that entity strings resolve to (1.0 = no fragmentation)
  predicate_fragmentation  distinct predicates / edges, plus the distinct-predicate count
  consistency              share of edges whose (subject type, predicate, object type) conforms to the
                           ontology schema, and share of edges grounded in a verbatim source span
Ranking impact requires relevance judgments (see Experiment D).
"""
from __future__ import annotations

import random
import re
import time
from collections import defaultdict

import numpy as np

from evaluation.registry import write_result
from knowledge_graph.neo4j_client import ONTOLOGY


def _surface_key(s: str) -> str:
    s = re.sub(r"[^a-z0-9+# ]", " ", s.lower())
    s = " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in s.split())
    return s.strip()


def structural_metrics(nodes: list[str], edges: list[tuple[str, str, str]], grounded: int,
                       typed_edges: list[tuple[str, str, str]] | None, resolver) -> dict:
    uniq = sorted(set(nodes))
    keys = defaultdict(set)
    for n in uniq:
        keys[_surface_key(n)].add(n)
    dup = sum(len(v) - 1 for v in keys.values())
    concepts = defaultdict(set)
    for n, cid in zip(uniq, resolver(uniq)):
        if cid:
            concepts[cid].add(n)
    preds = [p for _, p, _ in edges]
    rels = ONTOLOGY["relations"]
    conform = 0
    if typed_edges is not None:
        for st, p, ot in typed_edges:
            spec = rels.get(p)
            conform += int(bool(spec) and st in spec["from"] and ot in spec["to"])
    return {"nodes": len(uniq), "edges": len(edges),
            "duplicate_node_rate": round(dup / len(uniq), 4) if uniq else None,
            "entity_fragmentation": round(float(np.mean([len(v) for v in concepts.values()])), 4) if concepts else None,
            "resolvable_concepts": len(concepts),
            "distinct_predicates": len(set(preds)),
            "predicate_fragmentation": round(len(set(preds)) / len(preds), 4) if preds else None,
            "schema_conformance": round(conform / len(typed_edges), 4) if typed_edges else 0.0,
            "grounded_edge_rate": round(grounded / len(edges), 4) if edges else None}


def run(n_jobs: int = 120, n_resumes: int = 60, seed: int = 13, use_llm: bool = True, n_llm_jobs: int = 20,
        n_silver: int = 40) -> dict:
    from ingestion.resume_loader import iter_resumes
    from knowledge_graph.spo import classical_spo, llm_spo
    from matching.store import get_store
    from normalization.skill_normalizer import get_skill_normalizer
    from parsing.classical_parser import ClassicalJobParser, ClassicalResumeParser
    from parsing.llm_parser import LLMParser

    rng = random.Random(seed)
    store = get_store()
    test = [i for i, s in enumerate(store.df["split"]) if s == "test"] if "split" in store.df else list(range(len(store)))
    jobs = [store.job(i) for i in rng.sample(test, min(n_jobs, len(test)))]
    resumes = rng.sample(list(iter_resumes()), n_resumes)
    docs = [(j.job_id, j.description or "", "job", j) for j in jobs] + [(r.resume_id, r.text, "resume", None) for r in resumes]
    sk = get_skill_normalizer()

    def resolver(names):
        return [m.canonical_id for m in sk.normalize_batch(list(names), strategy="classical")]

    # --- ontology-constrained graph (classical extractor) -------------------------------------------
    jp, rp = ClassicalJobParser(), ClassicalResumeParser()
    t0 = time.perf_counter()
    o_nodes, o_edges, o_typed, o_ground = [], [], [], 0
    for did, text, kind, job in docs:
        if kind == "job":
            pj = jp.parse(job)
            owner = f"JobPosting:{did}"
            for r in pj.requirements:
                if r.kind == "responsibility":
                    continue
                rid = f"Requirement:{r.requirement_id}"
                o_edges.append((owner, "HAS_REQUIREMENT", rid))
                o_typed.append(("JobPosting", "HAS_REQUIREMENT", "Requirement"))
                o_ground += int(bool(r.evidence))
                if r.skill and r.skill.canonical_id:
                    o_nodes.append(r.skill.canonical_name)
                    o_edges.append((rid, "TARGETS", r.skill.canonical_name))
                    o_typed.append(("Requirement", "TARGETS", "Skill"))
                    o_ground += int(bool(r.evidence))
        else:
            c = rp.parse(text, did, did)
            for s in c.skills:
                if s.canonical_id:
                    o_nodes.append(s.canonical_name)
                    o_edges.append((did, "HAS_SKILL", s.canonical_name))
                    o_typed.append(("Candidate", "HAS_SKILL", "Skill"))
                    o_ground += int(bool(s.evidence))
            for o in c.occupations:
                o_nodes.append(o.canonical_title)
                o_edges.append((did, "TARGETS", o.canonical_title))
                o_typed.append(("Candidate", "TARGETS", "Occupation"))
                o_ground += int(bool(o.evidence))
    ont_t = (time.perf_counter() - t0) * 1000 / len(docs)
    ontology = structural_metrics(o_nodes, o_edges, o_ground, o_typed, resolver) | {"latency_ms_per_doc": round(ont_t, 1)}

    # --- free-form SPO graph (classical dependency parser) --------------------------------------------
    t0 = time.perf_counter()
    triples = [t for did, text, _, _ in docs for t in classical_spo(did, text)]
    spo_t = (time.perf_counter() - t0) * 1000 / len(docs)
    s_nodes = [t.subject for t in triples] + [t.object for t in triples]
    s_edges = [(t.subject, t.predicate, t.object) for t in triples]
    spo = structural_metrics(s_nodes, s_edges, len(triples), [("?", t.predicate.upper(), "?") for t in triples], resolver)
    spo["latency_ms_per_doc"] = round(spo_t, 1)
    spo["note"] = "Free-form predicates are lemmatised verbs (+preposition); none belong to the ontology's relation set."

    metrics = {"ontology_classical": ontology, "spo_classical": spo}
    llm_note = None
    if use_llm:
        lp = LLMParser(task="spo")
        if lp.available():
            # jobs only (no resume text to the 'spo' task); a subset — a 3B local model takes ~10-30 s per posting
            job_docs = [(d, t) for d, t, k, _ in docs if k == "job"][:n_llm_jobs]
            trip = [t for d, t in job_docs for t in llm_spo(d, t)]
            metrics["spo_llm_jobs"] = structural_metrics([t.subject for t in trip] + [t.object for t in trip],
                                                         [(t.subject, t.predicate, t.object) for t in trip], len(trip),
                                                         [("?", t.predicate.upper(), "?") for t in trip], resolver)
        else:
            llm_note = "LLM ontology/SPO variants not run: no model available for task 'spo'."
    # ranking impact under silver relevance: the final pipeline with vs without the graph features
    from evaluation import silver as silver_mod

    sr = silver_mod.run_ranking(silver_mod.test_sample(n_silver), methods=["final"])
    keep = ("final", "final_no_graph_features")
    silver_res = {"ranking_impact": {k: v for k, v in sr["metrics"].items() if k in keep},
                  "details": sr["details"],
                  "labels": "silver relevance from resume_corpus expert occupation labels (see Experiment D)"}
    return write_result("C_graph", "partially_evaluated", metrics, silver=silver_res,
                        reason="Structural metrics computed on identical documents. Ranking impact with human "
                               "judgments needs Experiment D annotations; reported here under silver relevance. "
                               + (llm_note or ""),
                        details={"documents": len(docs), "jobs": len(jobs), "resumes": len(resumes),
                                 "job_split": "test", "seed": seed,
                                 "llm_spo_jobs": n_llm_jobs if "spo_llm_jobs" in metrics else 0})
