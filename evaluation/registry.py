"""Experiment definitions (shown on the research dashboard)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from common.config import get_settings

Status = Literal["evaluated", "partially_evaluated", "not_evaluated", "blocked"]

EXPERIMENTS = {
    "A_extraction": {"title": "Classical vs LLM extraction", "rq": "RQ2", "metrics": ["precision", "recall", "f1"],
                     "requires": "human extraction annotations (Data/annotations/extraction/annotations.jsonl)"},
    "B_normalization": {"title": "Rule/TF-IDF vs embedding normalisation", "rq": "RQ3",
                        "metrics": ["top1_accuracy", "edge_precision"],
                        "requires": "skill gold (Data/annotations/normalization/gold.csv); occupation gold = O*NET held-out titles"},
    "C_graph": {"title": "Ontology graph vs free-form SPO graph", "rq": "RQ3",
                "metrics": ["duplicate_node_rate", "predicate_fragmentation", "entity_fragmentation", "consistency",
                            "ranking_impact"], "requires": "none for structural metrics; relevance labels for ranking impact"},
    "D_ranking": {"title": "BM25 / TF-IDF / Jaccard / Dense / RRF / LambdaMART / Cross-encoder", "rq": "RQ4",
                  "metrics": ["ndcg@10", "mrr", "precision@5"], "requires": "relevance judgments (0/1/2)"},
    "E_qlora": {"title": "QLoRA extractor vs zero-shot LLM", "rq": "RQ2", "metrics": ["f1", "latency"],
                "requires": "CUDA GPU for training + extraction gold"},
    "F_explanations": {"title": "Template vs citation-constrained LLM explanations", "rq": "RQ4",
                       "metrics": ["faithfulness", "hallucination_rate", "citation_validity"],
                       "requires": "automatic verifier (no labels); human faithfulness labels for the human metric"},
    "G_providers": {"title": "Ollama vs Groq vs OpenRouter vs proprietary baseline", "rq": "RQ1",
                    "metrics": ["quality", "latency", "tokens", "cost_per_1000_docs"],
                    "requires": "reachable providers (Ollama running / API keys)"},
    "classifiers": {"title": "Logistic-regression baselines with real labels", "rq": "baseline",
                    "metrics": ["accuracy", "macro_f1"], "requires": "none (O*NET titles, Kaggle resume categories)"},
}


def write_result(exp_id: str, status: Status, metrics: Optional[dict] = None, reason: Optional[str] = None,
                 details: Optional[dict] = None, descriptive: Optional[dict] = None,
                 silver: Optional[dict] = None) -> dict:
    """``metrics`` hold results against the labels the proposal specifies (human annotations, O*NET reference
    titles); ``silver`` holds results against dataset-derived labels (resume_corpus expert occupations) and is
    always reported separately."""
    s = get_settings()
    out = {"experiment": exp_id, **EXPERIMENTS.get(exp_id, {}), "status": status, "reason": reason,
           "metrics": metrics or {}, "silver": silver or {}, "descriptive": descriptive or {}, "details": details or {},
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": s.mode}
    d = s.path("results")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{exp_id}.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def read_result(exp_id: str) -> Optional[dict[str, Any]]:
    p = get_settings().path("results") / f"{exp_id}.json"
    return json.loads(p.read_text()) if p.exists() else None
