"""Experiment G — provider/model comparison (RQ1: quality loss vs cost & latency).

Runs ontology-constrained job extraction (job postings only: no resume text
leaves the machine) with each configured benchmark provider/model on the same
documents. Quality = F1 against human extraction gold when available,
otherwise only grounding rate (descriptive). Cost uses provider-reported cost
or configured pricing; unknown cost is reported as null, never 0.
"""
from __future__ import annotations

import random

from common.config import get_model_config
from evaluation.registry import write_result


def run(n_docs: int = 20, seed: int = 13) -> dict:
    from llm.router import get_router
    from matching.store import get_store
    from parsing.llm_parser import LLMParser

    router = get_router()
    bench = get_model_config().get("benchmark", [])
    available, skipped = [], []
    for b in bench:
        prov = router.providers.get(b["provider"])
        if prov and prov.available(b["model"]):
            available.append(b)
        else:
            reason = "not reachable" if prov and prov.local else "API key not configured"
            skipped.append({**b, "reason": reason})
    if not available:
        return write_result("G_providers", "not_evaluated",
                            reason="No benchmark provider is available (start Ollama and/or set GROQ_API_KEY, "
                                   "OPENROUTER_API_KEY, PROPRIETARY_API_KEY).", details={"skipped": skipped})
    store = get_store()
    rng = random.Random(seed)
    test = [i for i, s in enumerate(store.df["split"]) if s == "test"] if "split" in store.df else list(range(len(store)))
    jobs = [store.job(i) for i in rng.sample(test, min(n_docs, len(test)))]
    metrics = {}
    for b in available:
        lp = LLMParser(task="research", provider=b["provider"], model=b["model"], extractor_name="llm_bench")
        lp.cache_dir = lp.cache_dir.parent / "llm_bench"  # separate cache so latency is measured
        lp.cache_dir.mkdir(parents=True, exist_ok=True)
        tot = {"latency_ms": 0.0, "in": 0, "out": 0, "items": 0, "grounded": 0, "failed": 0, "docs": 0}
        for j in jobs:
            _, st = lp.parse_job(j)
            tot["latency_ms"] += st.latency_ms
            tot["in"] += st.input_tokens
            tot["out"] += st.output_tokens
            tot["items"] += st.items
            tot["grounded"] += st.grounded
            tot["failed"] += st.failed_chunks
            tot["docs"] += 1
        price = (get_model_config().get("pricing") or {}).get(f"{b['provider']}/{b['model']}") or {}
        cost = None
        if price.get("input") is not None and price.get("output") is not None:
            cost = (tot["in"] * price["input"] + tot["out"] * price["output"]) / 1e6
        metrics[f"{b['provider']}/{b['model']}"] = {
            "docs": tot["docs"], "mean_latency_ms_per_doc": round(tot["latency_ms"] / tot["docs"], 1),
            "input_tokens_per_doc": round(tot["in"] / tot["docs"], 1),
            "output_tokens_per_doc": round(tot["out"] / tot["docs"], 1),
            "grounding_rate": round(tot["grounded"] / tot["items"], 4) if tot["items"] else None,
            "failed_chunks": tot["failed"],
            "cost_per_1000_docs_usd": round(cost / tot["docs"] * 1000, 4) if cost is not None else None}
    return write_result("G_providers", "partially_evaluated", metrics,
                        reason="Quality vs human gold requires extraction annotations; grounding rate is descriptive.",
                        details={"skipped": skipped, "documents": len(jobs), "task": "job extraction"})
