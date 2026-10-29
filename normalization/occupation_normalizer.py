"""Job-title -> O*NET-SOC occupation normalisation.

Methods (recorded in ``OccupationMention.method``):
  exact          normalised title found among O*NET occupation/alternate/reported titles
  exact_cleaned  same after removing seniority/level/location noise
  tfidf          word+char TF-IDF cosine over all O*NET titles        (classical)
  embedding      sentence-embedding cosine over canonical/reported titles (modern)
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from common.schemas import OccupationMention
from common.text import normalize_key
from normalization.taxonomy import Taxonomy, load_taxonomy

Strategy = Literal["classical", "embedding", "hybrid"]
_SOURCE_RANK = {"occupation": 0, "occupation_singular": 0, "reported": 1, "short": 2, "alternate": 3}


def singularize(title: str) -> str:
    """'Data Scientists' -> 'Data Scientist'; 'Secretaries' -> 'Secretary' (last word only)."""
    words = title.split()
    if not words:
        return title
    w = words[-1]
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("ches", "shes", "sses", "xes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return " ".join(words[:-1] + [w])
_NOISE = re.compile(
    r"\b(senior|sr|junior|jr|lead|principal|staff|chief(?= of staff)|head|entry level|entry-level|associate(?= \w+ (?:engineer|developer|analyst))|"
    r"i{1,3}|iv|v|level \d|l\d|remote|hybrid|onsite|on-site|full[- ]time|part[- ]time|contract|temporary|temp|"
    r"intern(?:ship)?|trainee|apprentice|night shift|day shift|weekend|prn|per diem)\b", re.I)


def clean_title(title: str) -> str:
    t = re.sub(r"[\(\[].*?[\)\]]", " ", title)          # parentheticals
    t = re.split(r"\s+[-–|@]\s+|,\s+(?=[A-Z][a-z]+,?\s+[A-Z]{2}\b)", t)[0]  # ' - Location', '| Company'
    t = _NOISE.sub(" ", t)
    return normalize_key(t)


class OccupationNormalizer:
    def __init__(self, taxonomy: Optional[Taxonomy] = None, use_embeddings: bool = True,
                 tfidf_threshold: float = 0.65, embedding_threshold: float = 0.70):
        # thresholds chosen on a DEV sample of O*NET alternate titles (seed 7); Experiment B tests on seed 13
        self.tax = taxonomy or load_taxonomy()
        self.tfidf_threshold = tfidf_threshold
        self.embedding_threshold = embedding_threshold
        self._use_embeddings = use_embeddings
        self._emb = None
        titles = self.tax.occupation_titles.copy()
        occ = titles[titles["title_source"] == "occupation"]
        sing = occ.assign(title=occ["title"].map(singularize), title_source="occupation_singular")
        titles = pd.concat([titles, sing], ignore_index=True)
        titles["key"] = titles["title"].map(normalize_key)
        titles["rank"] = titles["title_source"].map(_SOURCE_RANK).fillna(9)
        # key -> candidate codes from the best-ranked source for that key
        self.exact: dict[str, list[str]] = {}
        for key, grp in titles.groupby("key", sort=False):
            best = grp[grp["rank"] == grp["rank"].min()]
            self.exact[key] = list(dict.fromkeys(best["code"]))
        uniq = titles.sort_values("rank").drop_duplicates(subset=["key"])
        self._keys = uniq["key"].tolist()
        self._w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        self._c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True)
        self._m = hstack([self._w.fit_transform(self._keys), self._c.fit_transform(self._keys)]).tocsr()
        self._emb_titles = titles[titles["title_source"].isin(["occupation", "occupation_singular", "reported", "short"])].drop_duplicates(subset=["key"])
        # occupation documents for disambiguating titles shared by several occupations
        occs = self.tax.occupations
        self._occ_codes = occs["code"].tolist()
        self._occ_pos = {c: i for i, c in enumerate(self._occ_codes)}
        self._occ_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._occ_m = self._occ_vec.fit_transform((occs["title"] + ". " + occs["title"] + ". " + occs["description"].fillna("")).tolist())
        self._occ_title_emb = None

    def _disambiguate(self, query: str, codes: list[str]) -> tuple[str, float]:
        """Pick the candidate occupation most similar to the query title."""
        if len(codes) == 1:
            return codes[0], 1.0
        idx = [self._occ_pos[c] for c in codes if c in self._occ_pos]
        if not idx:
            return codes[0], 0.5
        if self._use_embeddings:
            from common.embeddings import get_encoder

            enc = get_encoder()
            if enc is not None:
                if self._occ_title_emb is None:
                    self._occ_title_emb = enc.encode_cached(self.tax.occupations["title"].tolist(), "occupation_canonical_titles")
                q = enc.encode([query])[0]
                sims = self._occ_title_emb[idx] @ q
                j = int(np.argmax(sims))
                return self._occ_codes[idx[j]], float(sims[j])
        sims = (self._occ_m[idx] @ self._occ_vec.transform([query]).T).toarray().ravel()
        j = int(np.argmax(sims))
        return self._occ_codes[idx[j]], float(sims[j])

    @property
    def emb(self):
        if self._emb is None and self._use_embeddings:
            from normalization.embedding_normalizer import EmbeddingIndex

            idx = EmbeddingIndex(self._emb_titles["title"].tolist(), self._emb_titles["code"].tolist(), "occupation_titles")
            self._emb = idx if idx.available else False
        return self._emb or None

    def _mk(self, mention: str, code: str, method: str, conf: float) -> OccupationMention:
        return OccupationMention(mention=mention, code=code, canonical_title=self.tax.occupation_title(code),
                                 method=method, confidence=round(float(conf), 4))

    def normalize(self, title: Optional[str], strategy: Strategy = "hybrid",
                  exclude_exact: bool = False) -> OccupationMention:
        return self.normalize_batch([title or ""], strategy, exclude_exact)[0]

    def normalize_batch(self, titles: list[str], strategy: Strategy = "hybrid",
                        exclude_exact: bool = False) -> list[OccupationMention]:
        """``exclude_exact`` disables lexicon lookups (used to evaluate fuzzy methods on held-out titles)."""
        out: list[Optional[OccupationMention]] = [None] * len(titles)
        pending = []
        for i, t in enumerate(titles):
            if not t or not t.strip():
                out[i] = OccupationMention(mention=t or "", method="unresolved")
                continue
            if not exclude_exact:
                k, kc = normalize_key(t), clean_title(t)
                hit = k if k in self.exact else (kc if kc and kc in self.exact else None)
                if hit:
                    codes = self.exact[hit]
                    code, _ = self._disambiguate(t, codes)
                    method = ("exact" if hit == k else "exact_cleaned") + ("_ambiguous" if len(codes) > 1 else "")
                    conf = (1.0 if hit == k else 0.95) * (0.85 if len(codes) > 1 else 1.0)
                    out[i] = self._mk(t, code, method, conf)
                    continue
            pending.append(i)
        if pending and strategy in ("classical", "hybrid"):
            q = [clean_title(titles[i]) or normalize_key(titles[i]) for i in pending]
            qm = hstack([self._w.transform(q), self._c.transform(q)]).tocsr()
            sims = (qm @ self._m.T).toarray() / 2.0
            still = []
            for i, row in zip(pending, sims):
                j = int(np.argmax(row))
                if exclude_exact and self._keys[j] == normalize_key(titles[i]):
                    row[j] = -1
                    j = int(np.argmax(row))
                if row[j] >= self.tfidf_threshold:
                    code, _ = self._disambiguate(titles[i], self.exact[self._keys[j]])
                    out[i] = self._mk(titles[i], code, "tfidf", row[j])
                else:
                    still.append(i)
            pending = still
        if pending and strategy in ("embedding", "hybrid") and self.emb:
            res = self.emb.query([titles[i] for i in pending], k=3)
            for i, cands in zip(pending, res):
                cands = [c for c in cands if not (exclude_exact and normalize_key(c[1]) == normalize_key(titles[i]))]
                if cands and cands[0][2] >= self.embedding_threshold:
                    out[i] = self._mk(titles[i], cands[0][0], "embedding", cands[0][2])
        return [o or OccupationMention(mention=t or "", method="unresolved") for o, t in zip(out, titles)]

    def major_group(self, code: Optional[str]) -> Optional[str]:
        return code[:2] if code else None


@lru_cache(maxsize=1)
def get_occupation_normalizer() -> OccupationNormalizer:
    return OccupationNormalizer()
