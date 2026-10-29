"""Resume parsing orchestrator: file -> text -> structured Candidate.

Classical extraction always runs (fast, local, deterministic). The LLM
extractor runs when ``use_llm`` is True, or "auto" and a permitted local model
is reachable; its grounded facts are merged in. Every fact keeps its own
extractor label in its Evidence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from common.config import get_settings
from common.privacy import detect_injection
from common.schemas import Candidate
from normalization.skill_normalizer import merge_mentions
from parsing.classical_parser import ClassicalResumeParser
from parsing.text_extraction import extract_text

log = logging.getLogger(__name__)
UseLLM = Union[bool, Literal["auto"]]


@dataclass
class ResumeParseResult:
    candidate: Candidate
    file_type: str
    warnings: list[str] = field(default_factory=list)
    injection_flags: list[str] = field(default_factory=list)
    extractors: list[str] = field(default_factory=list)
    llm_stats: Optional[dict] = None


_classical: Optional[ClassicalResumeParser] = None


def classical_parser() -> ClassicalResumeParser:
    global _classical
    if _classical is None:
        _classical = ClassicalResumeParser()
    return _classical


def merge_candidates(base: Candidate, extra: Candidate) -> Candidate:
    """Union of facts; duplicates (same span or same canonical skill) are collapsed."""
    def spans(items):
        return {(e.start_offset, e.end_offset) for it in items for e in it.evidence}

    out = base.model_copy(deep=True)
    out.skills = merge_mentions(list(base.skills) + list(extra.skills))
    codes = {o.code for o in out.occupations}
    out.occupations += [o for o in extra.occupations if o.code not in codes]
    # an extra fact whose evidence overlaps anything the base already extracted is the same fact (or the same
    # text read as another type — e.g. a dated project offered again as a job) and is not added twice
    taken = [sp for attr in ("experiences", "education", "credentials", "projects")
             for sp in spans(getattr(out, attr))]

    def overlaps(it) -> bool:
        return any(s < b and a < e for ev in it.evidence for s, e in [(ev.start_offset, ev.end_offset)]
                   for a, b in taken)

    for attr in ("experiences", "education", "credentials", "projects"):
        getattr(out, attr).extend(it for it in getattr(extra, attr) if not overlaps(it))
    out.extractor = f"{base.extractor}+{extra.extractor}"
    return out


def parse_resume_text(text: str, candidate_id: str, source_document: str, use_llm: UseLLM = "auto") -> ResumeParseResult:
    flags = detect_injection(text)
    if flags:
        # The text is still parsed as data; flags are surfaced to the user and the critic.
        log.warning("Possible prompt-injection phrases in %s: %d", candidate_id, len(flags))
    cand = classical_parser().parse(text, candidate_id, source_document)
    res = ResumeParseResult(candidate=cand, file_type="txt", injection_flags=flags, extractors=["classical"])
    if use_llm == "auto":
        use_llm = get_settings().get("parsing.use_llm", "auto") in ("auto", True)
        auto = True
    else:
        auto = False
    if use_llm:
        from parsing.llm_parser import LLMParser

        parser = LLMParser()
        if parser.available():
            llm_cand, stats = parser.parse_resume(text, candidate_id, source_document)
            res.candidate = merge_candidates(cand, llm_cand)
            res.extractors.append(f"llm:{stats.provider}/{stats.model}")
            res.llm_stats = stats.__dict__
        elif not auto:
            res.warnings.append("LLM extraction requested but no permitted local model is available; "
                                "classical extraction only.")
    return res


def parse_resume_file(data: bytes, filename: str, candidate_id: str, use_llm: UseLLM = "auto") -> ResumeParseResult:
    ext = extract_text(data, filename)
    if not ext.text.strip():
        raise ValueError("No text could be extracted from the file")
    res = parse_resume_text(ext.text, candidate_id, filename, use_llm)
    res.file_type = ext.file_type
    res.warnings = (ext.warnings or []) + res.warnings
    return res
