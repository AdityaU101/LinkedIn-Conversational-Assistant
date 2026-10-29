"""Supervised classical sequence-labelling baseline (token-level BIO + logistic regression).

Trains on human span annotations (Data/annotations/extraction/annotations.jsonl)
once they exist; until then ``status()`` reports awaiting_annotations. Features:
lower-cased token, shape, prefix/suffix, gazetteer hit, section name, ±2-token
window. This complements the rule/gazetteer extractor in parsing/classical_parser.py.
"""
from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

from annotation.status import root
from common.config import get_settings

TOKEN = re.compile(r"\S+")


def status() -> dict:
    p = root() / "extraction" / "annotations.jsonl"
    n = sum(1 for l in p.read_text().splitlines() if l.strip()) if p.exists() else 0
    return {"state": "ready" if n >= 500 else "awaiting_annotations", "annotated_spans": n,
            "required_spans": 500}


def _shape(t: str) -> str:
    return re.sub(r"[a-z]+", "a", re.sub(r"[A-Z]+", "A", re.sub(r"\d+", "0", t)))


def token_features(tokens: list[tuple[int, int, str]], i: int, gaz: set[str]) -> dict:
    f = {}
    for k in range(-2, 3):
        j = i + k
        if 0 <= j < len(tokens):
            t = tokens[j][2]
            f[f"w{k}"] = t.lower()
            f[f"s{k}"] = _shape(t)
            f[f"g{k}"] = t.lower().strip(",.;:()") in gaz
    t = tokens[i][2]
    f.update({"pre3": t[:3].lower(), "suf3": t[-3:].lower(), "upper": t.isupper(), "title": t.istitle()})
    return f


def train() -> dict:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    st = status()
    if st["state"] != "ready":
        return st
    from normalization.skill_normalizer import get_skill_normalizer

    gaz = {k for k in get_skill_normalizer().spot_keys if " " not in k}
    tasks = {t["data"]["doc_id"]: t["data"]["text"]
             for t in json.loads((root() / "extraction" / "tasks.json").read_text())}
    spans = defaultdict(list)
    for line in (root() / "extraction" / "annotations.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            spans[d["doc_id"]].append((d["start"], d["end"], d["label"]))
    X, y = [], []
    for doc, text in tasks.items():
        if doc not in spans:
            continue
        toks = [(m.start(), m.end(), m.group()) for m in TOKEN.finditer(text)]
        for i, (s, e, _) in enumerate(toks):
            lab = "O"
            for a, b, l in spans[doc]:
                if a <= s < b:
                    lab = ("B-" if s == a or i == 0 or not (a <= toks[i - 1][0] < b) else "I-") + l
            X.append(token_features(toks, i, gaz))
            y.append(lab)
    vec = DictVectorizer()
    clf = LogisticRegression(max_iter=2000)
    clf.fit(vec.fit_transform(X), y)
    path = get_settings().path("indexes") / "classical_token_tagger.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"vec": vec, "clf": clf}, fh)
    return {"state": "trained", "tokens": len(y), "labels": sorted(set(y)), "path": str(path)}
