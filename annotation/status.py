"""Annotation progress vs. research targets (counts only real files on disk)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common.config import get_settings

TARGETS = {
    "extraction_documents_double_annotated": 150,
    "normalization_mentions": 500,
    "graph_edges": 100,
    "relevance_candidate_profiles": 40,
    "relevance_pooled_jobs_per_profile": 20,
    "relevance_judgments": 800,
    "explanations": 60,
    "user_study_participants": 15,
}


def root() -> Path:
    return get_settings().path("annotations")


def _csv_labelled(path: Path, label_col: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")] if label_col in df else pd.DataFrame()


def annotation_status() -> dict:
    r = root()
    out = {"targets": TARGETS, "progress": {}, "task_files": {}}
    ext = r / "extraction" / "annotations.jsonl"
    docs_by_annot: dict[str, set] = {}
    if ext.exists():
        for line in ext.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                docs_by_annot.setdefault(d["doc_id"], set()).add(d["annotator_id"])
    out["progress"]["extraction_documents_double_annotated"] = sum(1 for v in docs_by_annot.values() if len(v) >= 2)
    out["progress"]["extraction_documents_any"] = len(docs_by_annot)
    out["progress"]["normalization_mentions"] = int(_csv_labelled(r / "normalization" / "gold.csv", "gold_skill_id")["mention_id"].nunique()) \
        if (r / "normalization" / "gold.csv").exists() and len(_csv_labelled(r / "normalization" / "gold.csv", "gold_skill_id")) else 0
    rel = _csv_labelled(r / "relevance" / "judgments.csv", "label")
    out["progress"]["relevance_judgments"] = int(len(rel))
    out["progress"]["relevance_candidate_profiles"] = int(rel["candidate_id"].nunique()) if len(rel) else 0
    ge = _csv_labelled(r / "graph" / "edge_validation.csv", "valid")
    out["progress"]["graph_edges"] = int(ge["edge_key"].nunique()) if len(ge) else 0
    ex = _csv_labelled(r / "explanations" / "faithfulness.csv", "faithful")
    out["progress"]["explanations"] = int(ex["explanation_id"].nunique()) if len(ex) else 0
    us = r / "user_study" / "responses.csv"
    out["progress"]["user_study_participants"] = int(pd.read_csv(us)["participant_id"].nunique()) if us.exists() else 0
    for sub in ("extraction", "normalization", "relevance", "graph", "explanations", "user_study"):
        d = r / sub
        out["task_files"][sub] = sorted(p.name for p in d.glob("*")) if d.exists() else []
    return out
