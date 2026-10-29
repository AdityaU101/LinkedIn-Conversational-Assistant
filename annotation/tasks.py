"""Generate annotation task files (blind to system output where it matters).

Outputs under Data/annotations/<task>/ — only *tasks* and empty label
columns are written; labels are never pre-filled.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from annotation.status import root
from common.text import collapse_ws

GUIDELINES = {
    "extraction": """# Extraction annotation guidelines
Annotate every explicitly stated fact as a character span with one label:
Occupation, Skill, Experience (title+employer+dates block), Education, Credential, Project.
Do not annotate implied facts. Two annotators per document (target 150 documents).
Write one JSON object per span to annotations.jsonl:
{"doc_id": ..., "annotator_id": ..., "label": "Skill", "start": int, "end": int, "text": "exact span"}
Label Studio users: import tasks.json with label_studio_config.xml.""",
    "normalization": """# Skill-normalization gold
For each mention (shown with context, WITHOUT the system's prediction), choose the canonical skill from
Data/normalized/skills.parquet that means the SAME skill (not a related one). Fill gold_skill_id and
gold_skill_name; use NONE if no taxonomy entry is the same skill. Target 500 mentions.""",
    "relevance": """# Candidate–job relevance judgments
Scale: 0 = irrelevant, 1 = partially relevant, 2 = highly relevant (would apply / strong fit).
Judge from the profile and posting only. Jobs per profile are POOLED from all ranking methods and shuffled;
method names are hidden. Target 40 profiles × 20 jobs = 800 judgments; a subset should be double-judged.""",
    "graph": """# Graph edge validation
For each sampled edge decide valid = 1 if the relation is correct AND supported by the evidence text, else 0.
For inferred edges (RELATED_TO), judge whether the two skills are genuinely related (not identical). Target 100 edges.""",
    "explanations": """# Explanation faithfulness
For each sentence, faithful = 1 if every factual claim is supported by the listed evidence, else 0;
hallucination = 1 if it asserts a candidate skill/experience absent from the resume. Target 60 explanations.""",
}

LABEL_STUDIO_XML = """<View>
  <Labels name="label" toName="text">
    <Label value="Occupation"/><Label value="Skill"/><Label value="Experience"/>
    <Label value="Education"/><Label value="Credential"/><Label value="Project"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>"""


def _write_guidelines(task: str) -> Path:
    d = root() / task
    d.mkdir(parents=True, exist_ok=True)
    (d / "GUIDELINES.md").write_text(GUIDELINES[task])
    return d


def extraction_tasks(n_resumes: int = 100, n_jobs: int = 50, seed: int = 13) -> dict:
    from ingestion.resume_loader import iter_resumes
    from matching.store import get_store

    d = _write_guidelines("extraction")
    rng = random.Random(seed)
    resumes = list(iter_resumes())
    by_cat: dict[str, list] = {}
    for r in resumes:
        by_cat.setdefault(r.category, []).append(r)
    picked = []
    cats = sorted(by_cat)
    while len(picked) < n_resumes and cats:  # stratified over categories
        for c in cats:
            if by_cat[c] and len(picked) < n_resumes:
                picked.append(by_cat[c].pop(rng.randrange(len(by_cat[c]))))
    store = get_store()
    test_pos = [i for i, s in enumerate(store.df.get("split", pd.Series(["test"] * len(store)))) if s == "test"] \
        or list(range(len(store)))
    jobs = [store.job(i) for i in rng.sample(test_pos, min(n_jobs, len(test_pos)))]
    tasks = [{"doc_id": r.resume_id, "doc_type": "resume", "category": r.category, "text": r.text} for r in picked]
    tasks += [{"doc_id": j.job_id, "doc_type": "job", "text": j.description or ""} for j in jobs]
    (d / "tasks.json").write_text(json.dumps([{"data": t} for t in tasks], indent=1))
    (d / "label_studio_config.xml").write_text(LABEL_STUDIO_XML)
    if not (d / "annotations.jsonl").exists():
        (d / "annotations.jsonl").write_text("")
    return {"documents": len(tasks), "path": str(d)}


def normalization_tasks(n: int = 500, seed: int = 13) -> dict:
    """Mentions sampled from parsed jobs and resumes; the system prediction is stored separately (hidden)."""
    from matching.store import get_store

    d = _write_guidelines("normalization")
    rng = random.Random(seed)
    store = get_store()
    rows, hidden = [], []
    for i in rng.sample(range(len(store)), min(len(store), 2000)):
        reqs = json.loads(store.df.at[i, "requirements"])
        desc = store.df.at[i, "description"] or ""
        for r in reqs:
            if r["kind"] == "skill" and r.get("evidence"):
                ev = r["evidence"][0]
                ctx = collapse_ws(desc[max(0, ev["start_offset"] - 80):ev["end_offset"] + 80])
                rows.append({"mention_id": f"m{len(rows):05d}", "mention": ev["source_text"], "context": ctx,
                             "source_doc": store.ids[i]})
                hidden.append({"mention_id": rows[-1]["mention_id"], "system_skill_id": r["skill"]["canonical_id"],
                               "system_method": r["skill"]["method"]})
        if len(rows) >= n * 3:
            break
    sample = rng.sample(range(len(rows)), min(n, len(rows)))
    task = pd.DataFrame([rows[i] for i in sample])
    task["gold_skill_id"] = ""
    task["gold_skill_name"] = ""
    task["annotator_id"] = ""
    task.to_csv(d / "tasks.csv", index=False)
    pd.DataFrame([hidden[i] for i in sample]).to_csv(d / "system_predictions_HIDDEN.csv", index=False)
    if not (d / "gold.csv").exists():
        task.head(0).to_csv(d / "gold.csv", index=False)
    return {"mentions": len(task), "path": str(d),
            "note": "Gazetteer-found mentions are biased toward resolvable surface forms; add free-text mentions "
                    "from LLM extraction when available."}


def relevance_pool(candidate_ids: list[str], per_method: int = 8, pool_size: int = 20, seed: int = 13) -> dict:
    """Pool top results of every method per candidate (depth-k pooling), shuffle, hide methods."""
    from backend.repositories import candidates as repo
    from matching.pipeline import METHODS, get_pipeline

    d = _write_guidelines("relevance")
    rng = random.Random(seed)
    p = get_pipeline()
    rows, provenance = [], []
    for cid in candidate_ids:
        cand = repo.get_candidate(cid)
        if cand is None:
            continue
        pooled: dict[str, set] = {}
        for m in METHODS:
            if m == "dense" and p.dense is None:
                continue
            try:
                res = p.recommend(cand, method=m, k=per_method, with_gap=False, use_cross_encoder=(m == "final"))
            except ValueError:
                continue
            for r in res.results:
                pooled.setdefault(r.job_id, set()).add(m)
        jobs = list(pooled)
        rng.shuffle(jobs)
        for jid in jobs[:pool_size]:
            job = p.store.by_id(jid)
            rows.append({"candidate_id": cid, "job_id": jid, "job_title": job.title, "company": job.company,
                         "label": "", "annotator_id": ""})
            provenance.append({"candidate_id": cid, "job_id": jid, "methods": ";".join(sorted(pooled[jid]))})
    pd.DataFrame(rows).to_csv(d / "pool.csv", index=False)
    pd.DataFrame(provenance).to_csv(d / "pool_provenance_HIDDEN.csv", index=False)
    if not (d / "judgments.csv").exists():
        pd.DataFrame(columns=["candidate_id", "job_id", "label", "annotator_id", "timestamp", "notes"]).to_csv(
            d / "judgments.csv", index=False)
    return {"pairs": len(rows), "candidates": len(set(r["candidate_id"] for r in rows)), "path": str(d)}


def graph_edge_tasks(n: int = 100, seed: int = 13) -> dict:
    from knowledge_graph.neo4j_client import get_graph_store

    d = _write_guidelines("graph")
    rng = random.Random(seed)
    store = get_graph_store()
    edges = [e for e in store.all_edges() if e["type"] not in ("SUPPORTED_BY",)]
    rows = []
    for e in rng.sample(edges, min(n, len(edges))):
        s, t = store.node(e["src"]), store.node(e["dst"])
        ev_ids = (e.get("props") or {}).get("evidence_ids") or []
        ev = store.node(ev_ids[0]) if ev_ids else None
        rows.append({"edge_key": f"{e['src']}|{e['type']}|{e['dst']}", "relation": e["type"],
                     "source": s.get("name") or s.get("title") or s.get("text") or s["id"],
                     "target": t.get("name") or t.get("title") or t.get("text") or t["id"],
                     "inferred": (e.get("props") or {}).get("inferred"), "method": (e.get("props") or {}).get("method"),
                     "evidence_text": ev.get("text") if ev else "", "valid": "", "annotator_id": ""})
    pd.DataFrame(rows).to_csv(d / "edge_validation_tasks.csv", index=False)
    if not (d / "edge_validation.csv").exists():
        pd.DataFrame(rows).head(0).to_csv(d / "edge_validation.csv", index=False)
    return {"edges": len(rows), "path": str(d)}


def explanation_tasks(pairs: list[tuple[str, str]]) -> dict:
    from backend.services.matching_service import explanation

    d = _write_guidelines("explanations")
    rows = []
    for cid, jid in pairs:
        for method in ("template", "llm"):
            try:
                e = explanation(cid, jid, method)
            except Exception:
                continue
            if method == "llm" and e["method"] == "template":
                continue  # no LLM available: skip rather than duplicate
            eid = f"{cid}|{jid}|{e['method']}"
            ev = {i["cite_id"]: i["text"] for i in e["registry"]["items"]}
            for k, s in enumerate(e["sentences"]):
                rows.append({"explanation_id": eid, "sentence_idx": k, "sentence": s["text"],
                             "citations": ";".join(s.get("citations", [])),
                             "cited_evidence": " || ".join(ev.get(c, "") for c in s.get("citations", [])),
                             "faithful": "", "hallucination": "", "annotator_id": ""})
    pd.DataFrame(rows).to_csv(d / "faithfulness_tasks.csv", index=False)
    if not (d / "faithfulness.csv").exists():
        pd.DataFrame(rows).head(0).to_csv(d / "faithfulness.csv", index=False)
    return {"sentences": len(rows), "explanations": len({r["explanation_id"] for r in rows}), "path": str(d)}


def user_study_template() -> dict:
    d = root() / "user_study"
    d.mkdir(parents=True, exist_ok=True)
    (d / "PROTOCOL.md").write_text(
        "# User study (target 15–20 participants)\nEach participant uploads their own resume (or a provided "
        "persona), reviews top-10 recommendations and explanations, and rates on 1–5 Likert scales: relevance, "
        "explanation usefulness, trust, perceived accuracy of extracted skills. Record one row per participant "
        "per item in responses.csv. Obtain informed consent; store no resume text in this folder.\n")
    if not (d / "responses.csv").exists():
        pd.DataFrame(columns=["participant_id", "item", "rating", "comment", "timestamp"]).to_csv(d / "responses.csv", index=False)
    return {"path": str(d)}
