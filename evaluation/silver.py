"""Silver (dataset-derived) evaluations from resume_corpus expert occupation labels.

resume_corpus (Jiechieu & Tsopze, 2020) gives every resume one or more occupation classes assigned by
domain experts (10 normalised IT classes). These are real reference labels, but they are *not* the
human annotations the proposal specifies for Experiments A and D, so every number produced here is
reported separately and labelled "silver". Nothing here trains LambdaMART (human judgments only).

* Occupation extraction (part of A): do the occupations a parser extracts from a resume fall into
  the resume's expert classes? Classes are mapped to O*NET-SOC codes by a fixed, documented
  crosswalk (vocabulary level — no per-resume label is created or edited).
* Silver ranking (D) and graph ranking impact (C): a job is relevant to a resume when the job's
  *title* matches one of the resume's expert classes (rules below). The rule reads only the job
  title and the expert label, never a ranker's features — but lexical methods also read titles,
  so silver scores favour them; they rank methods only under this proxy.
"""
from __future__ import annotations

import re
import time
import zlib
from typing import Callable, Optional

# expert class -> O*NET-SOC codes. Base: the occupation(s) each class names in the O*NET 2019 taxonomy.
# ``crosswalk()`` adds every occupation that O*NET itself lists the class name under as a job title
# (job_titles.csv / sample_of_reported_titles.csv) — e.g. O*NET files "Java Developer" under Computer
# Programmers (15-1251.00). Vocabulary mapping only; no resume's label is touched.
CLASS_TITLES: dict[str, list[str]] = {
    "Software_Developer": ["Software Developer"], "Python_Developer": ["Python Developer"],
    "Java_Developer": ["Java Developer"], "Web_Developer": ["Web Developer"],
    "Front_End_Developer": ["Front End Developer", "Front-End Developer"],
    "Network_Administrator": ["Network Administrator"], "Systems_Administrator": ["Systems Administrator", "System Administrator"],
    "Database_Administrator": ["Database Administrator"], "Security_Analyst": ["Security Analyst", "Information Security Analyst"],
    "Project_manager": ["Project Manager", "IT Project Manager"],
}
BASE_CLASS_TO_SOC: dict[str, set[str]] = {
    "Software_Developer": {"15-1252.00"},
    "Python_Developer": {"15-1252.00"},
    "Java_Developer": {"15-1252.00"},
    "Web_Developer": {"15-1254.00"},
    "Front_End_Developer": {"15-1254.00", "15-1255.00"},
    "Network_Administrator": {"15-1244.00", "15-1241.00"},
    "Systems_Administrator": {"15-1244.00"},
    "Database_Administrator": {"15-1242.00", "15-1243.00"},
    "Security_Analyst": {"15-1212.00"},
    "Project_manager": {"15-1299.09", "13-1082.00", "11-3021.00"},
}

_CROSSWALK: Optional[dict[str, set[str]]] = None


def crosswalk() -> dict[str, set[str]]:
    global _CROSSWALK
    if _CROSSWALK is None:
        import pandas as pd

        from common.config import get_settings

        raw = get_settings().path("raw") / "onet"
        frames = [pd.read_csv(raw / f) for f in ("job_titles.csv", "sample_of_reported_titles.csv") if (raw / f).exists()]
        out = {k: set(v) for k, v in BASE_CLASS_TO_SOC.items()}
        for cls, names in CLASS_TITLES.items():
            low = {n.lower() for n in names}
            for df in frames:
                for col in [c for c in df.columns if c in ("Job Title", "Reported Job Title")]:
                    out[cls] |= set(df.loc[df[col].astype(str).str.lower().isin(low), "O*NET-SOC Code"])
        _CROSSWALK = out
    return _CROSSWALK


CLASS_TO_SOC = BASE_CLASS_TO_SOC  # class names (membership checks); codes come from crosswalk()


# expert class -> job-title rule (case-insensitive) used for silver relevance
TITLE_RULES: dict[str, re.Pattern] = {k: re.compile(v, re.I) for k, v in {
    "Software_Developer": r"\b(software (?:developer|engineer|development engineer)|programmer|application developer)\b",
    "Python_Developer": r"\bpython\b.*\b(developer|engineer|programmer)\b|\b(developer|engineer)\b.*\bpython\b",
    "Java_Developer": r"\bjava\b(?!script).*\b(developer|engineer|programmer)\b|\b(developer|engineer)\b.*\bjava\b(?!script)",
    "Web_Developer": r"\b(web (?:developer|engineer|programmer)|full[- ]?stack)\b",
    "Front_End_Developer": r"\b(front[- ]?end|ui (?:developer|engineer)|react (?:developer|engineer))\b",
    "Network_Administrator": r"\bnetwork (?:administrator|engineer|admin|analyst|technician)\b",
    "Systems_Administrator": r"\b(systems? (?:administrator|admin|engineer)|sys ?admin|linux administrator)\b",
    "Database_Administrator": r"\b(database (?:administrator|engineer|admin)|dba)\b",
    "Security_Analyst": r"\b((?:information |cyber ?|it )?security (?:analyst|engineer|specialist)|cyber ?security)\b",
    "Project_manager": r"\b((?:it |technical |software )?(?:project|program) manager|it manager|scrum master)\b",
}.items()}


def sample_corpus(n: int, seed: int = 11) -> list:
    """Deterministic sample of resume_corpus resumes, round-robin over primary expert class."""
    from ingestion.resume_corpus_loader import iter_corpus

    by_cls: dict[str, list] = {}
    for r in iter_corpus():
        labels = [l for l in r.labels if l in CLASS_TO_SOC]
        if not labels or len(r.text.split()) < 80:
            continue
        key = zlib.crc32(f"{seed}:{r.resume_id}".encode())
        by_cls.setdefault(labels[0], []).append((key, r))
    for v in by_cls.values():
        v.sort(key=lambda x: x[0])
    picked, i = [], 0
    while len(picked) < n and any(len(v) > i for v in by_cls.values()):
        for c in sorted(by_cls):
            if len(by_cls[c]) > i and len(picked) < n:
                picked.append(by_cls[c][i][1])
        i += 1
    return picked


DEV_SEED, TEST_SEED = 7, 23  # parser heuristics were inspected on the dev sample only


def test_sample(n: int, dev_n: int = 60) -> list:
    """Evaluation sample, disjoint from the dev sample used while developing parser heuristics."""
    dev = {r.resume_id for r in sample_corpus(dev_n, seed=DEV_SEED)}
    return [r for r in sample_corpus(n + dev_n, seed=TEST_SEED) if r.resume_id not in dev][:n]


def _labels(r) -> list[str]:
    return [l for l in r.labels if l in CLASS_TO_SOC]


def occupation_scores(resumes: list, extract: Callable[[object], list[str]]) -> dict:
    """``extract(resume) -> ordered O*NET codes``. Multi-label scoring against expert classes.

    precision: share of predicted codes that belong to one of the resume's expert classes;
    recall: share of expert classes hit by at least one predicted code; top1: first prediction correct.
    """
    tp_p = n_p = tp_r = n_r = top1 = with_pred = 0
    per_class: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    for r in resumes:
        gold = _labels(r)
        codes = extract(r)
        cw = crosswalk()
        ok = set().union(*(cw[g] for g in gold))
        n_p += len(codes)
        tp_p += sum(1 for c in codes if c in ok)
        for g in gold:
            hit = any(c in cw[g] for c in codes)
            n_r += 1
            tp_r += hit
            per_class.setdefault(g, [0, 0])
            per_class[g][0] += hit
            per_class[g][1] += 1
        if codes:
            with_pred += 1
            top1 += codes[0] in ok
    p = tp_p / n_p if n_p else 0.0
    rec = tp_r / n_r if n_r else 0.0
    return {"resumes": len(resumes), "precision": round(p, 4), "recall": round(rec, 4),
            "f1": round(2 * p * rec / (p + rec), 4) if p + rec else 0.0,
            "top1_accuracy": round(top1 / len(resumes), 4) if resumes else None,
            "resumes_with_any_occupation": round(with_pred / len(resumes), 4) if resumes else None,
            "recall_by_class": {k: round(v[0] / v[1], 3) for k, v in sorted(per_class.items())},
            "latency_ms_per_doc": round((time.perf_counter() - t0) * 1000 / max(1, len(resumes)), 1)}


def silver_qrels(resumes: list, job_titles: dict[str, str]) -> dict[str, dict[str, int]]:
    """Complete labels: every indexed job is judged 1 (title matches an expert class) or 0."""
    out = {}
    for r in resumes:
        pats = [TITLE_RULES[g] for g in _labels(r)]
        out[r.resume_id] = {jid: int(any(p.search(t or "") for p in pats)) for jid, t in job_titles.items()}
    return out


def run_ranking(resumes: list, methods: Optional[list[str]] = None, k: int = 50) -> dict:
    from evaluation.ranking import evaluate_rankings
    from matching.pipeline import METHODS, get_pipeline
    from parsing.classical_parser import ClassicalResumeParser

    p = get_pipeline()
    store = p.store
    titles = dict(zip(store.df["job_id"], store.df["title"]))
    qrels = silver_qrels(resumes, titles)
    parser = ClassicalResumeParser()
    cands = {r.resume_id: parser.parse(r.text, r.resume_id, r.resume_id) for r in resumes}
    runs_spec = [(m, {"method": m, "use_cross_encoder": m == "final"}) for m in (methods or METHODS)
                 if not (m == "dense" and p.dense is None)]
    runs_spec += [("final_no_cross_encoder", {"method": "final", "use_cross_encoder": False}),
                  ("final_no_mmr", {"method": "final", "use_mmr": False}),
                  ("final_no_graph_features", {"method": "final", "drop_graph_features": True})]
    metrics = {}
    for name, kw in runs_spec:
        runs = {rid: [x.job_id for x in p.recommend(c, k=k, with_gap=False, **kw).results] for rid, c in cands.items()}
        metrics[name] = evaluate_rankings(runs, qrels, judged_only=False)
    rel_counts = [sum(v.values()) for v in qrels.values()]
    return {"metrics": metrics,
            "details": {"queries": len(resumes), "jobs_judged_per_query": len(titles),
                        "relevant_jobs_per_query_mean": round(sum(rel_counts) / max(1, len(rel_counts)), 1),
                        "queries_with_no_relevant_job": sum(1 for x in rel_counts if x == 0),
                        "label_rule": "job title matches a TITLE_RULES pattern of the resume's expert class",
                        "gain": "binary (2^1-1)"}}


def run_occupations(n_classical: int = 200, n_llm: int = 30) -> dict:
    """Occupation extraction vs expert labels: classical parser, and the LLM extractor when a local model runs."""
    from parsing.classical_parser import ClassicalResumeParser
    from parsing.llm_parser import LLMParser

    rs = test_sample(max(n_classical, n_llm))
    cp = ClassicalResumeParser()
    out = {"classical": occupation_scores(rs[:n_classical], lambda r: [o.code for o in cp.parse(
        r.text, r.resume_id, r.resume_id).occupations if o.code])}
    lp = LLMParser()
    if lp.available():
        stats_tot = {"chunks": 0, "failed_chunks": 0, "grounded": 0, "items": 0, "rejected_invalid": 0}

        def llm_codes(r):
            cand, st = lp.parse_resume(r.text, r.resume_id, r.resume_id)
            for k in stats_tot:
                stats_tot[k] += getattr(st, k)
            return [o.code for o in cand.occupations if o.code]

        out["llm"] = occupation_scores(rs[:n_llm], llm_codes)
        out["llm"]["model"] = f"{lp._target()[0]}/{lp._target()[1]}"
        out["llm"]["extraction_stats"] = stats_tot
        # classical on exactly the LLM's documents, for a paired comparison
        out["classical_same_docs"] = occupation_scores(rs[:n_llm], lambda r: [o.code for o in cp.parse(
            r.text, r.resume_id, r.resume_id).occupations if o.code])
    out["labels"] = ("resume_corpus expert occupation classes (Jiechieu & Tsopze 2020) mapped to O*NET-SOC by "
                     "evaluation/silver.py crosswalk(); test sample disjoint from the dev sample")
    return out
