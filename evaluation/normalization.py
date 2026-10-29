"""Experiment B — normalisation accuracy (classical vs embedding vs hybrid).

* Occupations: REAL reference gold — 20% of O*NET alternate titles held out
  (seed 13); exact lexicon lookups are disabled so fuzzy methods are tested.
* Skills: human gold from Data/annotations/normalization/gold.csv (if any).
* Skills synthetic diagnostic: surface-form perturbations of canonical names;
  reported separately and explicitly marked synthetic (not a substitute for gold).
* Edge precision: from human graph-edge validation (if any).
"""
from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd

from annotation.status import root
from evaluation.registry import write_result


def occupation_eval(n: int = 1000, seed: int = 13) -> dict:
    from normalization.occupation_normalizer import OccupationNormalizer
    from normalization.taxonomy import load_taxonomy

    tax = load_taxonomy()
    titles = tax.occupation_titles
    alt = titles[titles["title_source"] == "alternate"]
    rng = np.random.default_rng(seed)
    test = alt.iloc[rng.choice(len(alt), size=min(n, len(alt)), replace=False)]
    norm = OccupationNormalizer(tax)
    out = {}
    for strategy in ("classical", "embedding", "hybrid"):
        t0 = time.perf_counter()
        preds = norm.normalize_batch(test["title"].tolist(), strategy=strategy, exclude_exact=True)
        lat = (time.perf_counter() - t0) * 1000 / len(test)
        codes = [p.code for p in preds]
        gold = test["code"].tolist()
        resolved = [c is not None for c in codes]
        out[strategy] = {
            "top1_accuracy": round(float(np.mean([c == g for c, g in zip(codes, gold)])), 4),
            "accuracy_on_resolved": round(float(np.mean([c == g for c, g, r in zip(codes, gold, resolved) if r])), 4)
            if any(resolved) else None,
            "detailed_soc_accuracy": round(float(np.mean([bool(c) and c[:7] == g[:7] for c, g in zip(codes, gold)])), 4),
            "major_group_accuracy": round(float(np.mean([bool(c) and c[:2] == g[:2] for c, g in zip(codes, gold)])), 4),
            "coverage": round(float(np.mean(resolved)), 4), "latency_ms_per_item": round(lat, 2)}
    return {"test_items": int(len(test)), "gold": "O*NET job_titles.csv alternate titles (held-out sample, seed 13)",
            "note": "Lexicon exact-match lookups disabled; other titles of the same occupation remain in the index.",
            "results": out}


def skill_gold_eval() -> dict | None:
    p = root() / "normalization" / "gold.csv"
    if not p.exists():
        return None
    g = pd.read_csv(p)
    g = g[g["gold_skill_id"].notna() & (g["gold_skill_id"].astype(str).str.strip() != "")]
    if g.empty:
        return None
    from normalization.skill_normalizer import get_skill_normalizer

    sk = get_skill_normalizer()
    g = g.drop_duplicates("mention_id")
    out = {}
    for strategy in ("classical", "embedding", "hybrid"):
        preds = sk.normalize_batch(g["mention"].astype(str).tolist(), strategy=strategy)
        gold = g["gold_skill_id"].astype(str).tolist()
        pred = [p.canonical_id or "NONE" for p in preds]
        out[strategy] = {"top1_accuracy": round(float(np.mean([a == b for a, b in zip(pred, gold)])), 4),
                         "abstention_rate": round(float(np.mean([x == "NONE" for x in pred])), 4)}
    return {"mentions": int(len(g)), "results": out}


def _perturb(name: str, rng: random.Random) -> str:
    ops = [lambda s: s.lower(), lambda s: s.replace(" ", ""), lambda s: s.replace("-", " "),
           lambda s: s + "s" if not s.endswith("s") else s[:-1],
           lambda s: (lambda i: s[:i] + s[i + 1] + s[i] + s[i + 2:])(rng.randrange(1, len(s) - 2)) if len(s) > 4 else s]
    return rng.choice(ops)(name)


def skill_synthetic_diagnostic(n: int = 400, seed: int = 13) -> dict:
    from normalization.skill_normalizer import get_skill_normalizer
    from normalization.taxonomy import load_taxonomy

    rng = random.Random(seed)
    tax = load_taxonomy()
    pool = tax.skills[(tax.skills["hot_technology"].astype(bool)) | (tax.skills["skill_type"] == "concept")]
    pool = pool.sample(min(n, len(pool)), random_state=seed)
    mentions = [_perturb(x, rng) for x in pool["name"]]
    sk = get_skill_normalizer()
    out = {}
    for strategy in ("classical", "embedding", "hybrid"):
        preds = sk.normalize_batch(mentions, strategy=strategy)
        ok = [p.canonical_id == g for p, g in zip(preds, pool["skill_id"])]
        wrong = [p.canonical_id is not None and p.canonical_id != g for p, g in zip(preds, pool["skill_id"])]
        out[strategy] = {"top1_accuracy": round(float(np.mean(ok)), 4), "wrong_link_rate": round(float(np.mean(wrong)), 4)}
    return {"SYNTHETIC": True, "items": len(mentions),
            "construction": "hot technologies + local concepts; one random surface perturbation each "
                            "(case, spacing, hyphen, plural, adjacent-character swap); seed 13", "results": out}


def edge_precision() -> dict | None:
    p = root() / "graph" / "edge_validation.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["valid"].notna() & (df["valid"].astype(str).str.strip() != "")]
    if df.empty:
        return None
    df["valid"] = df["valid"].astype(int)
    by = df.groupby("relation")["valid"].mean().round(4).to_dict()
    return {"edges": int(len(df)), "edge_precision": round(float(df["valid"].mean()), 4), "by_relation": by}


def run() -> dict:
    occ = occupation_eval()
    gold = skill_gold_eval()
    diag = skill_synthetic_diagnostic()
    ep = edge_precision()
    metrics = {"occupation": occ}
    if gold:
        metrics["skill_gold"] = gold
    if ep:
        metrics["edge_validation"] = ep
    status = "evaluated" if gold and ep else "partially_evaluated"
    missing = [x for x, v in (("skill gold annotations", gold), ("graph edge validation", ep)) if not v]
    return write_result("B_normalization", status, metrics,
                        reason=("Not yet evaluated: " + ", ".join(missing)) if missing else None,
                        descriptive={"skill_synthetic_diagnostic": diag})
