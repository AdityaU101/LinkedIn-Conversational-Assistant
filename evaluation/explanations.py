"""Experiment F — template vs citation-constrained LLM explanations.

Automatic metrics come from the Critic/Verifier (citation validity,
faithfulness = supported/factual sentences, hallucination rate). Human
faithfulness comes from Data/annotations/explanations/faithfulness.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from annotation.status import root
from evaluation.registry import write_result


def _agg(vals: list[dict]) -> dict:
    out = {}
    for k in ("citation_validity", "faithfulness", "hallucination_rate"):
        xs = [v[k] for v in vals if v.get(k) is not None]
        out[k] = round(float(np.mean(xs)), 4) if xs else None
    out["explanations"] = len(vals)
    out["mean_sentences"] = round(float(np.mean([v["n_sentences"] for v in vals])), 2) if vals else None
    return out


def run(candidate_ids: list[str], jobs_per_candidate: int = 3, n_llm: int = 20) -> dict:
    from backend.repositories import candidates as repo
    from explanations.llm_explainer import _template_numbers, llm_explanation
    from explanations.template import template_explanation
    from explanations.verifier import verify
    from llm.router import get_router
    from matching.pipeline import get_pipeline

    p = get_pipeline()
    tmpl, llm = [], []
    llm_ok = get_router().is_available("explainer")
    for cid in candidate_ids:
        c = repo.get_candidate(cid)
        if not c:
            continue
        res = p.recommend(c, method="ltr", k=jobs_per_candidate, with_gap=True)
        for r in res.results:
            e = template_explanation(r.gap)
            v = verify(e.text, e.registry, _template_numbers(r.gap))
            tmpl.append(v)
            if llm_ok and len(llm) < n_llm:  # a 3B local model takes ~10-20 s per explanation
                le = llm_explanation(r.gap, experiment="F_explanations")
                llm.append(le.verification)
    metrics = {"template_automatic": _agg(tmpl)}
    if llm:
        metrics["llm_automatic"] = _agg(llm)
    hp = root() / "explanations" / "faithfulness.csv"
    human = None
    if hp.exists():
        h = pd.read_csv(hp)
        h = h[h["faithful"].notna() & (h["faithful"].astype(str).str.strip() != "")]
        if len(h):
            h["method"] = h["explanation_id"].str.rsplit("|", n=1).str[-1]
            human = h.groupby("method").agg(faithfulness=("faithful", "mean"),
                                            hallucination_rate=("hallucination", "mean"),
                                            sentences=("faithful", "size")).round(4).to_dict("index")
            metrics["human"] = human
    status = "evaluated" if (llm and human) else "partially_evaluated"
    reasons = []
    if not llm:
        reasons.append("LLM explanations not generated: no model available for task 'explainer'")
    if not human:
        reasons.append("no human faithfulness labels yet")
    return write_result("F_explanations", status, metrics, reason="; ".join(reasons) or None,
                        details={"candidates": len(candidate_ids), "jobs_per_candidate": jobs_per_candidate,
                                 "note": "Template explanations are faithful by construction; the automatic score "
                                         "validates the verifier as much as the explainer."})
