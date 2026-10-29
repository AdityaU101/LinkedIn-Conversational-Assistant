"""Candidate–job features for learning-to-rank and the fallback ranker."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from common.schemas import EDUCATION_ORDER, Candidate
from matching.skill_gap import candidate_skill_evidence
from matching.store import JobStore
from normalization.taxonomy import Taxonomy, load_taxonomy

FEATURE_NAMES = [
    "bm25", "tfidf", "jaccard", "dense", "rrf",
    "required_coverage", "preferred_coverage", "missing_required", "n_required",
    "occupation_similarity", "experience_gap_years", "education_compatibility",
    "graph_inv_shortest_path", "graph_direct_skill_paths", "graph_related_bridges",
]
GRAPH_FEATURES = {"graph_inv_shortest_path", "graph_direct_skill_paths", "graph_related_bridges"}

# Documented fallback ranker (used while LambdaMART awaits human relevance labels).
# Hand-set, not learned; see docs/matching.md.
FALLBACK_WEIGHTS = {
    "rrf": 0.25, "required_coverage": 0.30, "preferred_coverage": 0.05, "dense": 0.15,
    "occupation_similarity": 0.10, "education_compatibility": 0.05, "experience_fit": 0.05,
    "graph_related_bridges_norm": 0.05,
}


@dataclass
class CandidateContext:
    candidate: Candidate
    skill_ids: set[str]
    occ_codes: list[str]
    years: Optional[float]
    edu_level: Optional[int]
    query_text: str
    qvec: Optional[np.ndarray] = None
    related_occ: dict[str, float] = field(default_factory=dict)


def candidate_query_text(c: Candidate) -> str:
    parts = []
    parts += [o.canonical_title or o.mention for o in c.occupations]
    parts += [e.title for e in c.experiences if e.title]
    parts += [s.canonical_name or s.mention for s in c.skills]
    parts += [f"{e.degree or ''} {e.field or ''}".strip() for e in c.education]
    parts += [x.name for x in c.credentials]
    if c.summary:
        parts.append(c.summary[:600])
    text = " \n".join(p for p in parts if p)
    return text or c.raw_text[:3000]


def build_context(c: Candidate, tax: Optional[Taxonomy] = None, encoder=None) -> CandidateContext:
    tax = tax or load_taxonomy()
    skills = set(candidate_skill_evidence(c))
    codes = list(dict.fromkeys([o.code for o in c.occupations if o.code]
                               + [e.occupation.code for e in c.experiences if e.occupation and e.occupation.code]))
    levels = [EDUCATION_ORDER[e.level] for e in c.education if e.level in EDUCATION_ORDER]
    rel: dict[str, float] = {}
    if codes:
        r = tax.related_occupations[tax.related_occupations["code"].isin(codes)]
        for code, tier in zip(r["related_code"], r["tier"]):
            rel[code] = max(rel.get(code, 0), 0.8 if tier.startswith("Primary") else 0.6)
    q = candidate_query_text(c)
    qvec = encoder.encode([q])[0] if encoder is not None else None
    return CandidateContext(c, skills, codes, c.total_experience_years, max(levels) if levels else None, q, qvec, rel)


def occupation_similarity(ctx: CandidateContext, job_code: Optional[str]) -> float:
    if not job_code or not ctx.occ_codes:
        return 0.0
    best = 0.0
    for c in ctx.occ_codes:
        if c == job_code:
            return 1.0
        if c[:7] == job_code[:7]:
            best = max(best, 0.9)       # same detailed SOC (different O*NET specialty)
        elif c[:5] == job_code[:5]:
            best = max(best, 0.5)       # same SOC minor group
        elif c[:2] == job_code[:2]:
            best = max(best, 0.3)       # same SOC major group
    return max(best, ctx.related_occ.get(job_code, 0.0))


def graph_features(ctx: CandidateContext, job_skills: frozenset[str], job_code: Optional[str],
                   tax: Taxonomy) -> tuple[float, float, float]:
    """Path features over the ontology graph:
    Candidate-HAS_SKILL->Skill<-TARGETS-Requirement<-HAS_REQUIREMENT-Job (3 hops),
    Candidate-HAS_SKILL->Skill-RELATED_TO->Skill<-TARGETS-Requirement<-Job (4 hops),
    Candidate-TARGETS->Occupation<-FOR_ROLE-Job (2 hops)."""
    direct = len(ctx.skill_ids & job_skills)
    missing = job_skills - ctx.skill_ids
    bridges = sum(1 for s in missing if any(o in ctx.skill_ids for o, _ in tax.related_skills.get(s, ())))
    path = None
    if job_code and job_code in ctx.occ_codes:
        path = 2
    elif direct:
        path = 3
    elif bridges:
        path = 4
    return (1.0 / path if path else 0.0), float(direct), float(bridges)


def compute_features(ctx: CandidateContext, store: JobStore, positions: list[int], comp: dict[str, np.ndarray],
                     tax: Optional[Taxonomy] = None) -> np.ndarray:
    """``comp`` maps component name -> score array over ``positions`` (same order)."""
    tax = tax or load_taxonomy()
    rows = []
    for i, p in enumerate(positions):
        req, pref = store.required_sets[p], store.preferred_sets[p]
        job_code = store.occ_codes[p]
        req_cov = len(req & ctx.skill_ids) / len(req) if req else 0.0
        pref_cov = len(pref & ctx.skill_ids) / len(pref) if pref else 0.0
        need_years = store.required_years[p]
        if need_years is None:
            gap = 0.0
        elif ctx.years is None:
            gap = need_years / 2.0
        else:
            gap = max(0.0, need_years - ctx.years)
        edu_req = store.edu_levels[p]
        if edu_req is None:
            edu = 1.0
        elif ctx.edu_level is None:
            edu = 0.5
        else:
            edu = 1.0 if ctx.edu_level >= edu_req else 0.0
        g_path, g_direct, g_bridge = graph_features(ctx, req | pref, job_code, tax)
        rows.append([comp.get("bm25", np.zeros(len(positions)))[i], comp.get("tfidf", np.zeros(len(positions)))[i],
                     comp.get("jaccard", np.zeros(len(positions)))[i], comp.get("dense", np.zeros(len(positions)))[i],
                     comp.get("rrf", np.zeros(len(positions)))[i], req_cov, pref_cov, float(len(req - ctx.skill_ids)),
                     float(len(req)), occupation_similarity(ctx, job_code), gap, edu, g_path, g_direct, g_bridge])
    return np.asarray(rows, dtype=np.float64).reshape(len(positions), len(FEATURE_NAMES))


def fallback_score(F: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    """Weighted sum of (min-max normalised) features. ``norm`` holds normalised columns."""
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    gap = F[:, idx["experience_gap_years"]]
    exp_fit = 1.0 / (1.0 + gap)
    bridges = F[:, idx["graph_related_bridges"]]
    bridges_n = bridges / (1.0 + bridges)
    # Smoothed coverage: matched / (n_required + 2). A posting with 1 extracted requirement (short snippets,
    # e.g. Adzuna) can no longer reach full credit from a single generic match.
    n_req = F[:, idx["n_required"]]
    matched = F[:, idx["required_coverage"]] * n_req
    req_cov_smoothed = matched / (n_req + 2.0)
    w = FALLBACK_WEIGHTS
    return (w["rrf"] * norm["rrf"] + w["required_coverage"] * req_cov_smoothed
            + w["preferred_coverage"] * F[:, idx["preferred_coverage"]] + w["dense"] * norm["dense"]
            + w["occupation_similarity"] * F[:, idx["occupation_similarity"]]
            + w["education_compatibility"] * F[:, idx["education_compatibility"]]
            + w["experience_fit"] * exp_fit + w["graph_related_bridges_norm"] * bridges_n)
