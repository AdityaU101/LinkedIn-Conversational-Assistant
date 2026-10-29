"""Skill normalisation (mention -> canonical taxonomy entry) and gazetteer extraction.

Methods, in order (each recorded in ``SkillMention.method``):
  exact       normalised mention == canonical name
  alias       curated or derived alias (aliases.py / taxonomy build)
  tfidf       character 3–5-gram TF-IDF cosine ≥ threshold          (classical)
  embedding   sentence-embedding cosine ≥ threshold with top-1 margin (modern)
  llm         optional LLM tie-break among embedding candidates      (modern, off by default)
Mentions that clear no threshold stay *unresolved* (canonical_id=None) rather
than being forced onto a related-but-different skill.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common.config import get_settings
from common.schemas import Evidence, SkillMention
from common.text import normalize_key
from normalization.aliases import CASE_SENSITIVE, LIST_CONTEXT_ONLY
from normalization.taxonomy import Taxonomy, load_taxonomy

Strategy = Literal["classical", "embedding", "hybrid"]
VERSION = "1.0"

_TOKEN = re.compile(r"[A-Za-z0-9+#]+(?:[.\-'][A-Za-z0-9+#]+)*")
_LIST_PUNCT = set(",;/|•·()[]:\n\t-–—*.")
_SPECIAL = [  # symbol-heavy names the tokenizer cannot see
    (re.compile(r"(?<![A-Za-z0-9])\.NET\b"), "Microsoft .NET Framework"),
    (re.compile(r"(?<![A-Za-z0-9])ASP\.NET\b"), "Microsoft ASP.NET"),
]
MAX_NGRAM = 6
_CS_KEYS = {normalize_key(x): x for x in CASE_SENSITIVE}


@lru_cache(maxsize=1)
def _english_words() -> frozenset[str]:
    """System word list (macOS/Linux) used to recognise ordinary English words; empty if unavailable,
    in which case single-word derived aliases are never spotted."""
    from pathlib import Path

    p = Path("/usr/share/dict/words")
    if not p.exists():
        return frozenset()
    return frozenset(w.strip().lower() for w in p.read_text(errors="ignore").splitlines())


def _is_english_word(w: str) -> bool:
    words = _english_words()
    return True if not words else w.lower() in words  # no word list: be conservative


@dataclass
class _Entry:
    skill_id: str
    surface: str
    method: str            # exact | alias:<kind>
    case_sensitive: bool
    list_context: bool
    lower_ok_in_list: bool = False   # all-lowercase form accepted in list context


class SkillNormalizer:
    def __init__(self, taxonomy: Optional[Taxonomy] = None, use_embeddings: bool = True):
        self.tax = taxonomy or load_taxonomy()
        cfg = get_settings().section("normalization")
        self.tfidf_threshold = cfg.get("tfidf_threshold", 0.86)
        self.emb_threshold = cfg.get("embedding_threshold", 0.82)
        self.emb_margin = cfg.get("embedding_margin", 0.03)
        self.llm_tiebreak = cfg.get("llm_tiebreak", False)
        self._use_embeddings = use_embeddings
        self._emb_index = None
        self._build_lexicon()

    # ------------------------------------------------------------------ lexicon
    def _build_lexicon(self) -> None:
        self.lexicon: dict[str, list[_Entry]] = {}
        skills = self.tax.skills
        for sid, name, stype, hot in skills[["skill_id", "name", "skill_type", "hot_technology"]].itertuples(index=False):
            key = normalize_key(name)
            if not key:
                continue
            # single-word O*NET element names ('Writing', 'Science') are too generic to spot in text
            # knowledge areas ('Education and Training') read like section headers, and single-word
            # O*NET element names ('Writing', 'Science') are too generic to spot in free text
            spot = stype != "knowledge" and not (stype in {"essential_skill", "transferable_skill"} and len(key.split()) == 1)
            single_tech = stype == "technology" and len(key.split()) == 1
            # 'Ada' must not match 'ADA' (Americans with Disabilities Act); 'act' must not match 'Act!'
            plain_word = single_tech and name[:1].isupper() and name[1:].islower() and name.isalpha() and not bool(hot)
            self._add(key, _Entry(sid, name, "exact", name in CASE_SENSITIVE or single_tech,
                                  key in LIST_CONTEXT_ONLY or plain_word,
                                  lower_ok_in_list=single_tech and len(key) >= 3), spot)
        for alias, sid, method in self.tax.aliases[["alias", "skill_id", "method"]].itertuples(index=False):
            key = normalize_key(alias)
            if not key or key in self.lexicon:
                continue
            derived = method != "curated_alias"
            single_generic = derived and len(alias.split()) == 1 and alias[:1].isupper() and alias[1:].islower()
            acronym = derived and alias.isupper()
            self._add(key, _Entry(sid, alias, f"alias:{method}",
                                  alias in CASE_SENSITIVE or single_generic or acronym,
                                  key in LIST_CONTEXT_ONLY or single_generic or (acronym and len(alias) <= 3)),
                      # 'Inventory' (Oracle Inventory) is an English word: normalisable, never spotted.
                      # 'Visio' / 'Bomgar' are not: spotted in list context only.
                      spot=not (single_generic and _is_english_word(alias)))
        # TF-IDF index over every surface form
        self._surfaces = list(self._all_surfaces.keys())
        self._surface_ids = [self._all_surfaces[s] for s in self._surfaces]
        self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        self._tfidf_matrix = self._tfidf.fit_transform(self._surfaces)

    def _add(self, key: str, entry: _Entry, spot: bool) -> None:
        if not hasattr(self, "_all_surfaces"):
            self._all_surfaces: dict[str, str] = {}
            self.spot_keys: set[str] = set()
        self.lexicon.setdefault(key, []).append(entry)
        self._all_surfaces.setdefault(key, entry.skill_id)
        if spot:
            self.spot_keys.add(key)

    @property
    def emb_index(self):
        if self._emb_index is None and self._use_embeddings:
            from normalization.embedding_normalizer import EmbeddingIndex

            names = self.tax.skills["name"].tolist()
            idx = EmbeddingIndex(names, self.tax.skills["skill_id"].tolist(), "skill_names_all")
            self._emb_index = idx if idx.available else False
        return self._emb_index or None

    # ---------------------------------------------------------------- normalise
    def _resolved(self, mention: str, sid: str, method: str, conf: float) -> SkillMention:
        info = self.tax.skill_by_id.get(sid, {})
        return SkillMention(mention=mention, canonical_id=sid, canonical_name=info.get("name", sid),
                            taxonomy=info.get("taxonomy"), method=method, confidence=round(float(conf), 4))

    def candidates_tfidf(self, mentions: list[str], k: int = 5) -> list[list[tuple[str, str, float]]]:
        q = self._tfidf.transform([normalize_key(m) for m in mentions])
        sims = (q @ self._tfidf_matrix.T).toarray()
        out = []
        for row in sims:
            top = np.argsort(-row)[:k]
            out.append([(self._surface_ids[i], self._surfaces[i], float(row[i])) for i in top])
        return out

    def normalize(self, mention: str, strategy: Strategy = "hybrid") -> SkillMention:
        return self.normalize_batch([mention], strategy)[0]

    def normalize_batch(self, mentions: list[str], strategy: Strategy = "hybrid") -> list[SkillMention]:
        results: list[Optional[SkillMention]] = [None] * len(mentions)
        pending = []
        for i, m in enumerate(mentions):
            key = normalize_key(m)
            entries = self.lexicon.get(key)
            if entries:
                e = entries[0]
                results[i] = self._resolved(m, e.skill_id, "exact" if e.method == "exact" else "alias",
                                            1.0 if e.method == "exact" else 0.97)
            else:
                pending.append(i)
        if pending and strategy in ("classical", "hybrid"):
            cands = self.candidates_tfidf([mentions[i] for i in pending], k=2)
            still = []
            for i, c in zip(pending, cands):
                if c and c[0][2] >= self.tfidf_threshold:
                    results[i] = self._resolved(mentions[i], c[0][0], "tfidf_char", c[0][2])
                else:
                    still.append(i)
            pending = still
        if pending and strategy in ("embedding", "hybrid") and self.emb_index:
            cands = self.emb_index.query([mentions[i] for i in pending], k=5)
            for i, c in zip(pending, cands):
                if not c:
                    continue
                top, second = c[0], (c[1] if len(c) > 1 else (None, None, 0.0))
                if top[2] >= self.emb_threshold and top[2] - second[2] >= self.emb_margin:
                    results[i] = self._resolved(mentions[i], top[0], "embedding", top[2])
                elif top[2] >= self.emb_threshold and self.llm_tiebreak:
                    pick = self._llm_tiebreak(mentions[i], c[:4])
                    if pick:
                        results[i] = self._resolved(mentions[i], pick, "llm_tiebreak", top[2])
        return [r or SkillMention(mention=m, method="unresolved", confidence=0.0) for r, m in zip(results, mentions)]

    def _llm_tiebreak(self, mention: str, cands: list[tuple[str, str, float]]) -> Optional[str]:
        from llm.router import get_router

        router = get_router()
        options = "\n".join(f"{i + 1}. {name}" for i, (_, name, _) in enumerate(cands))
        prompt = (f"Which canonical skill is the SAME skill as the mention (not merely related)?\n"
                  f"Mention: {mention!r}\nOptions:\n{options}\n0. none of them\n"
                  'Answer as JSON: {"choice": <number>}')
        try:
            resp = router.complete_json("tiebreak", [{"role": "user", "content": prompt}])
            choice = int(resp.data.get("choice", 0))
        except Exception:
            return None
        return cands[choice - 1][0] if 1 <= choice <= len(cands) else None

    # ------------------------------------------------------------------ extract
    def extract(self, text: str, doc_id: str, extractor: str = "gazetteer", section_of=None,
                base_offset: int = 0) -> list[SkillMention]:
        """Find taxonomy skills in ``text`` with exact character offsets.

        ``section_of(offset) -> str|None`` lets callers relax list-context rules
        inside a SKILLS section.
        """
        toks = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
        found: list[tuple[int, int, _Entry]] = []
        i = 0
        while i < len(toks):
            match = None
            for n in range(min(MAX_NGRAM, len(toks) - i), 0, -1):
                s, e = toks[i][0], toks[i + n - 1][1]
                raw = text[s:e]
                key = normalize_key(raw)
                if key not in self.spot_keys:
                    continue
                for entry in self.lexicon[key]:
                    required = _CS_KEYS.get(key) or (entry.surface if entry.case_sensitive else None)
                    if required and raw != required:
                        # 'pandas' written 'Pandas' (or lower-cased) in a skills list is the same tool
                        relaxed = entry.lower_ok_in_list and (raw.islower() or raw.lower() == required.lower())
                        if not (relaxed and self._list_context(text, s, e, section_of)):
                            continue
                    if entry.list_context and not self._list_context(text, s, e, section_of):
                        continue
                    match = (s, e, entry)
                    break
                if match:
                    break
            if match:
                found.append(match)
                i += len([t for t in toks[i:] if t[1] <= match[1]])
            else:
                i += 1
        for pat, name in _SPECIAL:
            sid = next((e.skill_id for e in self.lexicon.get(normalize_key(name), [])), None)
            if sid:
                for m in pat.finditer(text):
                    if not any(s < m.end() and m.start() < e for s, e, _ in found):
                        found.append((m.start(), m.end(), _Entry(sid, name, "alias:special", True, False)))
        out = []
        for s, e, entry in sorted(found, key=lambda x: x[0]):
            info = self.tax.skill_by_id.get(entry.skill_id, {})
            conf = 0.95 if entry.method == "exact" else (0.9 if entry.method.startswith("alias:curated") else 0.8)
            ev = Evidence(source_document_id=doc_id, source_text=text[s:e], start_offset=s + base_offset,
                          end_offset=e + base_offset, extractor=extractor, extractor_version=VERSION, confidence=conf)
            out.append(SkillMention(mention=text[s:e], canonical_id=entry.skill_id, canonical_name=info.get("name"),
                                    taxonomy=info.get("taxonomy"), method=f"gazetteer_{entry.method.split(':')[0]}",
                                    confidence=conf, evidence=[ev]))
        return out

    @staticmethod
    def _list_context(text: str, s: int, e: int, section_of) -> bool:
        if section_of and section_of(s) == "skills":
            return True
        left = text[:s].rstrip(" ")
        right = text[e:].lstrip(" ")
        lc = left[-1:] or "\n"
        rc = right[:1] or "\n"
        left_ok = lc in _LIST_PUNCT or left.lower().endswith((" and", " or", " &"))
        right_ok = rc in _LIST_PUNCT or right.lower().startswith(("and ", "or ", "& "))
        return left_ok and right_ok


def merge_mentions(mentions: Iterable[SkillMention]) -> list[SkillMention]:
    """Collapse mentions of the same canonical skill, keeping all evidence."""
    merged: dict[str, SkillMention] = {}
    for m in mentions:
        k = m.key
        if k in merged:
            cur = merged[k]
            seen = {e.evidence_id for e in cur.evidence}
            cur.evidence.extend(e for e in m.evidence if e.evidence_id not in seen)
            cur.confidence = max(cur.confidence, m.confidence)
        else:
            merged[k] = m.model_copy(deep=True)
    return list(merged.values())


@lru_cache(maxsize=1)
def get_skill_normalizer() -> SkillNormalizer:
    return SkillNormalizer()
