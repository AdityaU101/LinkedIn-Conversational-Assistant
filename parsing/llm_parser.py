"""Ontology-constrained LLM extraction for resumes and job postings.

Pipeline per document:
  1. split into ~350-word chunks with 10% overlap (configs/settings.yaml)
  2. prompt the routed model with a fixed ontology + JSON shape
  3. validate with Pydantic (one repair retry, handled by the router)
  4. GROUND every item: its verbatim evidence quote must be found in the chunk
     and the extracted value must occur in that quote; otherwise it is rejected
  5. normalise skills / occupations against the taxonomy
Results are cached by content hash (model + prompt version + chunk text).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from common.config import get_settings
from common.schemas import (
    Candidate, Credential, Education, Evidence, Experience, Job, OccupationMention, Project, Requirement, SkillMention,
)
from common.text import find_span, normalize_key, sha1, word_chunks
from llm.prompts.extraction import PROMPT_VERSION, job_messages, resume_messages
from llm.router import LLMOutputError, ModelRouter, NoProviderAvailable, get_router
from llm.schemas.extraction import JobExtraction, ResumeExtraction
from normalization.occupation_normalizer import get_occupation_normalizer
from normalization.skill_normalizer import get_skill_normalizer, merge_mentions
from parsing.classical_parser import DEGREE_LEVEL, months_between

log = logging.getLogger(__name__)
VERSION = "1.0"


@dataclass
class ExtractionStats:
    chunks: int = 0
    items: int = 0
    grounded: int = 0
    rejected_ungrounded: int = 0
    rejected_value_not_in_evidence: int = 0
    rejected_invalid: int = 0  # items the model returned without required fields (e.g. no evidence quote)
    failed_chunks: int = 0
    cache_hits: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    provider: Optional[str] = None
    model: Optional[str] = None
    errors: list[str] = field(default_factory=list)


def _value_in_quote(value: Optional[str], quote: str) -> bool:
    if not value:
        return True
    v, q = normalize_key(value), normalize_key(quote)
    if not v:
        return True
    if v in q:
        return True
    vt = set(v.split())
    return bool(vt) and len(vt & set(q.split())) / len(vt) >= 0.6


_PLACEHOLDER = {"not specified", "n a", "na", "none", "unknown", "null", "not mentioned", "not available",
                "not provided", "unspecified", "tbd"}


def _attr(value: Optional[str], ev: Evidence) -> Optional[str]:
    """A secondary value (company, institution, dates …) is kept only when it is not a placeholder and appears in
    the item's own evidence quote; small models fill gaps with 'Not specified' or guesses."""
    if not value or normalize_key(value) in _PLACEHOLDER:
        return None
    return value if _value_in_quote(value, ev.source_text) else None


class LLMParser:
    def __init__(self, router: Optional[ModelRouter] = None, task: str = "parser",
                 provider: Optional[str] = None, model: Optional[str] = None, extractor_name: str = "llm"):
        self.router = router or get_router()
        self.task = task
        self.provider, self.model = provider, model
        self.extractor_name = extractor_name
        cfg = get_settings().section("parsing")
        self.chunk_words = int(cfg.get("chunk_words", 350))
        self.overlap = float(cfg.get("chunk_overlap", 0.10))
        self.cache_dir = get_settings().path("cache") / "llm_extraction"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.skills = get_skill_normalizer()
        self.occupations = get_occupation_normalizer()

    def available(self) -> bool:
        if self.provider and self.model:
            p = self.router.providers.get(self.provider)
            return bool(p and p.available(self.model) and self.router.permitted(self.task, self.provider))
        return self.router.is_available(self.task)

    def _target(self) -> tuple[str, str]:
        if self.provider and self.model:
            return self.provider, self.model
        chain = self.router.available_chain(self.task)
        if not chain:
            raise NoProviderAvailable(f"No available model for task {self.task!r}")
        return chain[0]

    def _call(self, kind: str, chunk: str, schema, messages, stats: ExtractionStats):
        prov, model = self._target()
        stats.provider, stats.model = prov, model
        key = sha1(f"{prov}/{model}|{PROMPT_VERSION}|{kind}|{chunk}")
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            stats.cache_hits += 1
            out = schema.model_validate(json.loads(path.read_text())["data"])
            stats.rejected_invalid += getattr(out, "invalid_items", 0)
            return out
        res = self.router.complete_json(self.task, messages, schema=schema, provider=prov, model=model)
        stats.latency_ms += res.latency_ms
        stats.input_tokens += res.input_tokens or 0
        stats.output_tokens += res.output_tokens or 0
        path.write_text(json.dumps({"provider": prov, "model": model, "prompt_version": PROMPT_VERSION,
                                    "kind": kind, "repaired": res.repaired, "data": res.data}))
        stats.rejected_invalid += getattr(res.parsed, "invalid_items", 0)
        return res.parsed

    def _ground(self, doc_id: str, text: str, chunk_start: int, chunk: str, quote: str, value: Optional[str],
                stats: ExtractionStats) -> Optional[Evidence]:
        stats.items += 1
        span = find_span(chunk, quote)
        if not span:
            stats.rejected_ungrounded += 1
            return None
        s, e = chunk_start + span[0], chunk_start + span[1]
        if not _value_in_quote(value, text[s:e]):
            stats.rejected_value_not_in_evidence += 1
            return None
        stats.grounded += 1
        return Evidence(source_document_id=doc_id, source_text=text[s:e], start_offset=s, end_offset=e,
                        extractor=self.extractor_name, extractor_version=f"{VERSION}/{PROMPT_VERSION}", confidence=0.8)

    # ------------------------------------------------------------------ resume
    def parse_resume(self, text: str, candidate_id: str, source_document: str) -> tuple[Candidate, ExtractionStats]:
        stats = ExtractionStats()
        skills, occs, exps, edus, creds, projs = [], {}, [], [], [], []
        for cs, ce, chunk in word_chunks(text, self.chunk_words, self.overlap):
            stats.chunks += 1
            try:
                out: ResumeExtraction = self._call("resume", chunk, ResumeExtraction, resume_messages(chunk), stats)
            except (LLMOutputError, NoProviderAvailable) as e:
                stats.failed_chunks += 1
                stats.errors.append(str(e)[:200])
                continue
            for it in out.skills:
                ev = self._ground(candidate_id, text, cs, chunk, it.evidence, it.name, stats)
                if ev:
                    sm = self.skills.normalize(it.name)
                    sm.mention, sm.evidence = it.name, [ev]
                    skills.append(sm)
            for it in out.occupations:
                ev = self._ground(candidate_id, text, cs, chunk, it.evidence, it.title, stats)
                if ev:
                    om = self.occupations.normalize(it.title)
                    om.evidence = [ev]
                    if om.code:
                        occs.setdefault(om.code, om)
            for it in out.experiences:
                ev = self._ground(candidate_id, text, cs, chunk, it.evidence, it.title or it.company, stats)
                if ev:
                    occ = self.occupations.normalize(it.title) if it.title else None
                    sd, ed = _attr(it.start_date, ev), _attr(it.end_date, ev)
                    exps.append(Experience(title=_attr(it.title, ev), company=_attr(it.company, ev), start_date=sd,
                                           end_date=ed, duration_months=months_between(sd, ed) if sd and ed else None,
                                           occupation=occ if occ and occ.code else None, evidence=[ev]))
            for it in out.education:
                ev = self._ground(candidate_id, text, cs, chunk, it.evidence, it.degree or it.institution, stats)
                if ev:
                    level = next((lvl for pat, lvl in DEGREE_LEVEL if it.degree and pat.search(it.degree)), None)
                    edus.append(Education(degree=_attr(it.degree, ev), level=level, field=_attr(it.field, ev),
                                          institution=_attr(it.institution, ev), year=_attr(it.year, ev),
                                          evidence=[ev]))
            for it in out.credentials:
                ev = self._ground(candidate_id, text, cs, chunk, it.evidence, it.name, stats)
                if ev:
                    creds.append(Credential(name=it.name, issuer=_attr(it.issuer, ev), evidence=[ev]))
            for it in out.projects:
                ev = self._ground(candidate_id, text, cs, chunk, it.evidence, it.name, stats)
                if ev:
                    projs.append(Project(name=it.name, description=it.description, evidence=[ev]))
        cand = Candidate(candidate_id=candidate_id, source_document=source_document, raw_text=text,
                         occupations=list(occs.values()), skills=merge_mentions(skills), experiences=_dedup(exps),
                         education=_dedup(edus), credentials=_dedup(creds), projects=_dedup(projs),
                         extractor=f"{self.extractor_name}:{stats.provider}/{stats.model}")
        return cand, stats

    # --------------------------------------------------------------------- job
    def parse_job(self, job: Job) -> tuple[Job, ExtractionStats]:
        stats = ExtractionStats()
        text = job.description or ""
        reqs: list[Requirement] = []
        occ_title = None
        for cs, ce, chunk in word_chunks(text, self.chunk_words, self.overlap):
            stats.chunks += 1
            try:
                out: JobExtraction = self._call("job", chunk, JobExtraction, job_messages(chunk, job.title), stats)
            except (LLMOutputError, NoProviderAvailable) as e:
                stats.failed_chunks += 1
                stats.errors.append(str(e)[:200])
                continue
            occ_title = occ_title or out.occupation
            for r in out.requirements:
                ev = self._ground(job.job_id, text, cs, chunk, r.evidence, r.text if r.kind == "skill" else None, stats)
                if not ev:
                    continue
                req = Requirement(kind=r.kind, importance=r.importance, text=r.text, years=r.years, evidence=[ev])
                if r.kind == "skill":
                    sm = self.skills.normalize(r.text)
                    sm.evidence = [ev]
                    req.skill = sm
                if r.kind == "education":
                    req.education_level = next((lvl for pat, lvl in DEGREE_LEVEL if pat.search(r.text)), None)
                reqs.append(req)
            for r in out.responsibilities:
                ev = self._ground(job.job_id, text, cs, chunk, r.evidence, None, stats)
                if ev:
                    reqs.append(Requirement(kind="responsibility", importance="required", text=ev.source_text, evidence=[ev]))
        # merge skill requirements found in overlapping chunks
        merged: dict[str, Requirement] = {}
        final = []
        for r in reqs:
            if r.kind == "skill" and r.skill:
                k = r.skill.key
                if k in merged:
                    merged[k].evidence.extend(e for e in r.evidence if e.evidence_id not in {x.evidence_id for x in merged[k].evidence})
                    if r.importance == "required":
                        merged[k].importance = "required"
                    continue
                merged[k] = r
            final.append(r)
        for i, r in enumerate(final):
            r.requirement_id = f"req:{job.job_id}:llm{i}"
        occ = self.occupations.normalize(job.title or occ_title or "")
        return job.model_copy(update={"requirements": final, "occupation": occ if occ.code else None}), stats


def _dedup(items: list) -> list:
    seen, out = set(), []
    for it in items:
        k = (it.evidence[0].start_offset, it.evidence[0].end_offset) if it.evidence else id(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out
