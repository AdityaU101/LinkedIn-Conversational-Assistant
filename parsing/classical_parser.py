"""Classical (non-LLM) extraction baseline for resumes and job postings.

Techniques: section segmentation, regular expressions (dates, degrees,
credentials, years of experience), a taxonomy gazetteer for skills (a
dictionary-based NER) and, when installed, spaCy ``en_core_web_sm`` NER for
organisations. Every fact carries Evidence with exact character offsets.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from functools import lru_cache
from typing import Optional

from common.schemas import (
    Candidate, Credential, Education, Evidence, Experience, Job, OccupationMention, Project, Requirement,
)
from normalization.occupation_normalizer import OccupationNormalizer, get_occupation_normalizer
from normalization.skill_normalizer import SkillNormalizer, get_skill_normalizer, merge_mentions
from parsing.sections import JOB_HEADERS, RESUME_HEADERS, SectionMap, segment

log = logging.getLogger(__name__)
EXTRACTOR = "classical"
VERSION = "1.2"  # job parser (keys the job-parse cache); 1.2: lexicon adds Java, LightGBM, Streamlit, Next.js, …
# resume parser — 1.2: pipe-delimited role lines, '|' certification lists, wrapped projects;
# 1.3: education entries per block (layout-scored school pairing, own-line field, nearest year), dated project titles
# 1.4: stacked projects (title / own-line date range / tech-stack line), tab-separated title dates
RESUME_VERSION = "1.4"

MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
_DATE = rf"(?:{_MONTH}\s*,?\s*(?:19|20)\d{{2}}|\d{{1,2}}\s*/\s*(?:19|20)\d{{2}}|(?:19|20)\d{{2}})"
DATE_RANGE = re.compile(
    rf"(?P<start>{_DATE})\s*(?:-|–|—|to|until|through|thru)\s*(?P<end>{_DATE}|present|current|now|today|date)\b", re.I)
_SEG_SPLIT = re.compile(r"\n+|[ \t ]{3,}|\s+[－–—]\s+")

# A bare 'Master(s)' / 'Bachelor(s)' counts only when it stands alone (followed by punctuation, a column gap or the
# line end) — never inside a name like 'Road Master Truck Driving School'.
_BARE_END = r"(?=\s*(?:[:,(|–—-]|\n|$)|\s{2,})"
DEGREE_LONG = re.compile(
    r"\b(?P<deg>Ph\.?\s?D\.?|Doctor(?:ate)?\s+of\s+[A-Z][a-z]+|"
    r"Master(?:['’]s|s)?(?:\s+(?:of|in)\s+[A-Z][a-z]+(?: [A-Z][a-z]+)?|\s+[Dd]egree|" + _BARE_END + r")|"
    r"M\.B\.A\.?|MBA|Bachelor(?:['’]s|s)?(?:\s+(?:of|in)\s+[A-Z][a-z]+(?: [A-Z][a-z]+)?|\s+[Dd]egree|" + _BARE_END + r")|"
    r"Associate(?:['’]s|s)?\s+(?:of|in)\s+[A-Z][a-z]+(?: (?:and )?[A-Z][a-z]+)?(?:\s+[Dd]egree)?|Associate(?:['’]s|s)?\s+[Dd]egree|"
    r"High School (?:Diploma|Degree)|GED|"
    r"B\.\s?S\.|B\.\s?A\.|M\.\s?S\.|M\.\s?A\.|B\.\s?Sc\.?|M\.\s?Sc\.?|B\.?Tech|M\.?Tech|B\.\s?E\.|"
    r"[BM]\.\s?F\.\s?A\.?|M\.\s?Ed\.?|B\.\s?Ed\.?|LL\.?\s?B\.?|B\.\s?Com\.?|M\.\s?Com\.?|B\.?C\.?A\.|M\.?C\.?A\.)(?![A-Za-z])")
CERTIFICATE_EDU = re.compile(r"(?i)(?<![A-Za-z])(?P<deg>certificate(?:\s+of\s+[A-Za-z]+)?|diploma(?:\s+in\s+[A-Za-z]+)?)(?![A-Za-z])")
FIELD_AFTER = re.compile(r"^\s*[:,\-–]\s*(?P<field>[A-Z][A-Za-z&]+(?: (?:and |& )?[A-Z][A-Za-z&]+){0,4})")
DEGREE_SHORT = re.compile(r"(?<![A-Za-z])(?P<deg>BS|BA|MS|MA|BSc|MSc|BBA|BSN|MSN|AAS|AS|AA|BE|ME|PhD|JD|MD|BFA|MFA|"
                          r"MEd|BEd|LLB|BCom|MCom|BCA|MCA)(?![A-Za-z])")
FIELD = re.compile(r"\b(?:in|of) (?P<field>[A-Z][A-Za-z&]+(?: (?:and )?[A-Z][A-Za-z&]+){0,4})")
_INST_KEY = (r"(?:University|College|Institute|School|Academy|Polytechnic|(?:Senior|Junior) High|"
             r"UNIVERSITY|COLLEGE|INSTITUTE|SCHOOL|ACADEMY|POLYTECHNIC)")
_INST_WORD = r"(?:[A-Z][A-Za-z&.'\-]+|\d{1,2})"  # 'Don Bosco 2 Senior High School'; never a 4-digit year
# 'University of Illinois' — but not '… of Technology Jul 2020': a month never continues a name
_OF_WORDS = r"(?:(?!(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\b)[A-Z][A-Za-z&.'\-]+(?: |(?=\W|$))){1,4}"
# A name around the keyword is required: a bare 'College' / 'School' (e.g. 'College Prep') is not an institution.
INSTITUTION = re.compile(
    r"(?P<inst>(?:" + _INST_WORD + r" ){1,5}" + _INST_KEY + r"(?: School)?(?: of " + _OF_WORDS + r")?"
    r"|" + _INST_KEY + r" of " + _OF_WORDS + r")")
# 'M.S. Computer Science | GPA' — capitalised words right after an abbreviated degree, single-spaced, same line
FIELD_DIRECT = re.compile(r"^[ ,]\s?(?P<field>[A-Z][A-Za-z&]+(?: (?:and |& |of )?[A-Z][A-Za-z&]+){0,4})")
INSTITUTION_WORD = re.compile(r"(?i)\b(?:University|College|Institute|School|Academy|Polytechnic)\b")
YEAR = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")

CREDENTIAL_PATTERNS = [
    re.compile(r"(?<![A-Za-z])(?:PMP|CAPM|CPA|CFA|CISSP|CISM|CISA|CCNA|CCNP|CCIE|CEH|OSCP|MCSE|MCSA|PHR|SPHR|"
               r"SHRM-CP|SHRM-SCP|CSM|PSM|LPN|CNA|EMT|BLS|ACLS|PALS|CDL|ServSafe|ITIL(?:\s+v?\d|\s+Foundation)?)(?![A-Za-z])"),
    re.compile(r"\bCompTIA\s+(?:A\+|Network\+|Security\+|Linux\+|Cloud\+|CySA\+|Server\+|Project\+)"),
    re.compile(r"\b(?:Lean\s+)?Six Sigma\s+(?:Green|Black|Yellow|White)\s+Belt\b"),
    re.compile(r"\bAWS Certified(?:\s+[A-Z][\w\-]*){1,5}"),
    re.compile(r"\b(?:Microsoft|Google|Oracle|Cisco|Salesforce) Certified(?:\s+[A-Z][\w\-:]*){1,5}"),
    re.compile(r"\bCertified\s+(?:[A-Z][\w\-]*\s?){1,5}"),
    re.compile(r"\b(?:Registered Nurse|Licensed (?:Practical|Vocational) Nurse)\b"),
    re.compile(r"\b(?:[A-Z][\w&\-]*\s){1,4}(?:Certification|Certificate|License)\b"),
]

DEGREE_LEVEL = [
    (re.compile(r"ph\.?\s?d|doctor|\bjd\b|\bmd\b", re.I), "doctorate"),
    (re.compile(r"master|m\.?b\.?a|\bm\.?\s?s\.?c?\b|\bm\.?\s?a\.?\b|m\.?tech|\bmsn\b|\bme\b", re.I), "master"),
    (re.compile(r"bachelor|\bb\.?\s?s\.?c?\b|\bb\.?\s?a\.?\b|b\.?tech|\bbba\b|\bbsn\b|\bb\.?\s?e\.?\b", re.I), "bachelor"),
    (re.compile(r"associate|\baas\b|\ba\.?a\.?\b|\bas\b", re.I), "associate"),
    (re.compile(r"high school|ged", re.I), "high_school"),
]

YEARS_REQ = re.compile(
    r"(?P<min>\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*(?P<max>\d{1,2})\s*)?\+?\s*(?:years?|yrs?)\.?"
    r"(?:\s+of)?(?P<ctx>(?:\s+[\w\-/+#.]+){0,6}?)\s+(?:experience|exp\b)", re.I)
SUMMARY_TITLE = re.compile(r"(?:(?:An?|Experienced|Senior|Results-driven)\s+)?(?P<title>(?:[A-Z][\w/+&.\-]*\s){0,4}"
                           r"[A-Z][\w/+&.\-]*)\s+with\s+(?:over\s+|more than\s+)?\d{1,2}\+?\s+years")
STATED_YEARS = re.compile(r"(?:over|more than|\b)\s*(?P<n>\d{1,2})\+?\s*years?\s+(?:of\s+)?(?:\w+\s+){0,4}?experience", re.I)
PREFERRED_CUE = re.compile(r"\b(prefer(?:red|ably)?|nice[- ]to[- ]have|a plus|is a plus|bonus|desired|desirable|ideally|"
                           r"advantage(?:ous)?|familiarity with is helpful)\b", re.I)
REQUIRED_CUE = re.compile(r"\b(required|must|minimum|at least|need to have|mandatory|essential)\b", re.I)


def _ev(doc_id: str, text: str, s: int, e: int, conf: float, extractor: str = EXTRACTOR,
        version: str = VERSION) -> Evidence:
    return Evidence(source_document_id=doc_id, source_text=text[s:e], start_offset=s, end_offset=e,
                    extractor=extractor, extractor_version=version, confidence=conf)


def _rev(doc_id: str, text: str, s: int, e: int, conf: float) -> Evidence:
    return _ev(doc_id, text, s, e, conf, version=RESUME_VERSION)


_URL = re.compile(r"https?://|www\.|\b[\w\-]+\.(?:com|io|org|net|dev|ai|me)/", re.I)
_LINK_LABEL = re.compile(r"^(?:demo|link|repo|code|video|website|live|github|url)\s*:", re.I)
_NAME_TOKEN = re.compile(r"^(?:[A-Za-z][A-Za-z'’\-]*|[A-Za-z]\.)$")
# words that head a resume but are a role or a document label, never part of a person's name
_NOT_A_NAME = {"resume", "curriculum", "vitae", "cv", "profile", "summary", "engineer", "developer", "manager",
               "analyst", "scientist", "consultant", "designer", "intern", "student", "senior", "junior", "lead",
               "software", "data", "sales", "marketing", "teacher", "specialist", "administrator", "officer",
               "associate", "director", "assistant", "technician", "architect", "accountant", "leader", "team"}


def candidate_name(text: str) -> Optional[str]:
    """The person's name from the top of a resume, or None when it is not clearly one.

    Only the first line is considered, and only when contact details (an e-mail address or phone number) follow
    in the header: 'KRISH NAIK' above '602-… | krish@…' is a name; 'BPO TEAM LEADER   Summary …' is not.
    Abstains rather than guesses — the caller then writes without a name.
    """
    from common.privacy import EMAIL, PHONE

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines or not (EMAIL.search("\n".join(lines[:6])) or PHONE.search("\n".join(lines[:6]))):
        return None
    first = re.split(r"\s*[|•·,]\s*|\s{3,}", lines[0])[0].strip()
    tokens = first.split()
    if not 2 <= len(tokens) <= 4 or len(first) > 40 or not all(_NAME_TOKEN.match(t) for t in tokens):
        return None
    if any(t.lower().strip(".") in _NOT_A_NAME for t in tokens):
        return None
    if first.isupper() or first.islower():      # 'KRISH NAIK' -> 'Krish Naik'; mixed case is kept as written
        first = " ".join(t.capitalize() if "-" not in t and "'" not in t else t.title() for t in tokens)
    return first


_LOCATION = re.compile(r"^(?:remote|hybrid|[A-Z][a-zA-Z .'\-]+,\s*(?:[A-Z]{2}|[A-Z][a-zA-Z .'\-]+))$")


def _parse_point(tok: str, is_end: bool) -> Optional[tuple[int, int]]:
    t = tok.strip().lower()
    if t in {"present", "current", "now", "today", "date"}:
        d = date.today()
        return d.year, d.month
    m = re.match(r"(\d{1,2})\s*/\s*(\d{4})", t)
    if m:
        mo = int(m.group(1))
        return int(m.group(2)), mo if 1 <= mo <= 12 else (12 if is_end else 1)
    m = re.match(r"([a-z]{3})[a-z]*\.?\s*,?\s*(\d{4})", t)
    if m and m.group(1) in MONTHS:
        return int(m.group(2)), MONTHS[m.group(1)]
    m = re.match(r"(\d{4})", t)
    if m:
        return int(m.group(1)), 12 if is_end else 1
    return None


def months_between(start: str, end: str) -> Optional[int]:
    a, b = _parse_point(start, False), _parse_point(end, True)
    if not a or not b:
        return None
    n = (b[0] - a[0]) * 12 + (b[1] - a[1]) + 1
    return n if 0 < n < 600 else None


@lru_cache(maxsize=1)
def _spacy():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _segments_around(text: str, start: int, end: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Spans of the text segment just before ``start`` and just after ``end``."""
    left = text[max(0, start - 300):start]
    cuts = list(_SEG_SPLIT.finditer(left))
    ls = max(0, start - 300) + (cuts[-1].end() if cuts else 0)
    if ls >= start - 1 and len(cuts) >= 2:  # date sits right after a separator: take the segment before it
        ls = max(0, start - 300) + cuts[-2].end()
    prev = (ls, start)
    right = text[end:end + 300]
    m = _SEG_SPLIT.search(right.lstrip(" ,|\t"))
    off = end + (len(right) - len(right.lstrip(" ,|\t")))
    nxt = (off, off + (m.start() if m else min(len(right), 120)))
    return prev, nxt


_HEADER_WORDS = {h for hs in RESUME_HEADERS.values() for h in hs}


def _clean_title(s: str) -> Optional[str]:
    s = re.sub(r"[\s,|:;\-–—]+$", "", s).strip(" ,|:-–—\t")
    s = re.sub(r"\s+", " ", s)
    if not s or len(s.split()) > 9 or not re.search(r"[A-Za-z]{2}", s) or "@" in s:
        return None
    return s


class ClassicalResumeParser:
    def __init__(self, skills: Optional[SkillNormalizer] = None, occupations: Optional[OccupationNormalizer] = None,
                 use_spacy: bool = True):
        self.skills = skills or get_skill_normalizer()
        self.occupations = occupations or get_occupation_normalizer()
        self.use_spacy = use_spacy

    def parse(self, text: str, candidate_id: str, source_document: str) -> Candidate:
        sections = segment(text, RESUME_HEADERS)
        smap = SectionMap(sections)
        doc_id = candidate_id
        skills = self.skills.extract(text, doc_id, extractor="classical_gazetteer", section_of=smap)
        experiences = self._experiences(text, doc_id, sections)
        education = self._education(text, doc_id, sections)
        credentials = self._credentials(text, doc_id, smap)
        projects = self._projects(text, doc_id, sections)
        occupations = self._occupations(text, doc_id, sections, experiences)
        summary = None
        for s in smap.of("summary"):
            summary = text[s.start:s.end].strip()[:1500] or None
            break
        months = self._total_months(experiences)
        stated = [int(m.group("n")) for m in STATED_YEARS.finditer(text[:3000])]
        total_years = round(months / 12, 1) if months else (float(max(stated)) if stated else None)
        return Candidate(candidate_id=candidate_id, source_document=source_document, raw_text=text,
                         occupations=occupations, skills=merge_mentions(skills), experiences=experiences,
                         education=education, credentials=credentials, projects=projects,
                         total_experience_years=total_years, summary=summary, extractor=f"{EXTRACTOR}/{RESUME_VERSION}")

    # -- experience -------------------------------------------------------------
    def _experiences(self, text: str, doc_id: str, sections) -> list[Experience]:
        exp_secs = [s for s in sections if s.name == "experience"] or [s for s in sections if s.name not in {"education"}]
        anchors = []
        for sec in exp_secs:
            for m in DATE_RANGE.finditer(text, sec.start, sec.end):
                anchors.append((m, sec))
        out: list[Experience] = []
        nlp = _spacy() if self.use_spacy else None
        for idx, (m, sec) in enumerate(anchors):
            (ps, pe), (ns, ne) = _segments_around(text, m.start(), m.end())
            line_start = text.rfind("\n", 0, m.start()) + 1
            prefix = text[line_start:m.start()]
            if "|" in prefix and prefix.strip(" |\t"):
                # 'Title | Company | City, ST | Jan 2021 - Aug 2024'
                parts = [(line_start + p.start(), line_start + p.end()) for p in re.finditer(r"[^|]+", prefix)
                         if p.group().strip()]
                parts = [(a + len(text[a:b]) - len(text[a:b].lstrip()), a + len(text[a:b].rstrip())) for a, b in parts]
                title = _clean_title(text[parts[0][0]:parts[0][1]])
                if title and title.lower() not in _HEADER_WORDS:
                    company = None
                    if len(parts) >= 2 and not _LOCATION.match(text[parts[1][0]:parts[1][1]]):
                        company = text[parts[1][0]:parts[1][1]]
                    out.append(self._experience(text, doc_id, anchors, idx, m, sec, title, parts[0][0], company,
                                                desc_start=max(ne, m.end())))
                    continue
            title_raw = text[ps:pe]
            # 'Title , 09/2012' or 'Title at Company - City'
            title_raw = re.split(r"\s+(?:at|@)\s+", title_raw)[0]
            title = _clean_title(title_raw)
            company = None
            nxt = text[ns:ne].strip(" ,|")
            if title is None or title.lower().strip(": ") in _HEADER_WORDS or DATE_RANGE.search(title_raw):
                # date-first layout: 'Experience   05/2014 to 01/2016   Graphic Designer   Company Name'
                cand = _clean_title(nxt)
                if cand and cand.lower() not in _HEADER_WORDS:
                    title, ps, pe = cand, ns, ne
                    (ns, ne) = _segments_around(text, ns, ne)[1]
                    nxt = text[ns:ne].strip(" ,|")
                else:
                    title = None
            if nxt and len(nxt.split()) <= 8 and not DATE_RANGE.search(nxt):
                company = re.split(r"\s+[-－–—]\s+|\s{2,}|,\s+[A-Z][a-z]+\s*,", nxt)[0].strip(" ,") or None
            if title and " - " in title:  # 'Senior Sysadmin Trico Products - Brownsville, TX'
                title = title.split(" - ")[0].strip()
            if nlp is not None and title:
                ents = [e for e in nlp(title).ents if e.label_ == "ORG"]
                if ents and ents[-1].end_char == len(title) and ents[-1].start_char > 0:
                    company = company or ents[-1].text
                    title = title[: ents[-1].start_char].strip(" ,-") or title
            title_start = text.find(title, ps, pe + 1) if title else m.start()
            out.append(self._experience(text, doc_id, anchors, idx, m, sec, title, title_start, company,
                                        desc_start=ne if ne > m.end() else m.end()))
        return out

    def _experience(self, text: str, doc_id: str, anchors, idx: int, m, sec, title: Optional[str], title_start: int,
                    company: Optional[str], desc_start: int) -> Experience:
        ev_start = title_start if title_start >= 0 else m.start()
        desc_end = anchors[idx + 1][0].start() if idx + 1 < len(anchors) else sec.end
        desc_end = min(desc_end, m.end() + 3000)
        if idx + 1 < len(anchors):  # stop before the next role's title segment / line
            nxt_m = anchors[idx + 1][0]
            nps, _ = _segments_around(text, nxt_m.start(), nxt_m.end())[0]
            desc_end = max(desc_start, min(desc_end, nps, text.rfind("\n", 0, nxt_m.start()) + 1))
        desc = text[desc_start:desc_end].strip() or None
        exp_skills = self.skills.extract(text[desc_start:desc_end], doc_id, extractor="classical_gazetteer",
                                         base_offset=desc_start) if desc else []
        for sm in exp_skills:  # evidence source_text must match the document
            for ev in sm.evidence:
                ev.source_text = text[ev.start_offset:ev.end_offset]
        occ = self.occupations.normalize(title) if title else None
        occ_len = len(title) if title else 0
        if title and "," in title and not (occ and occ.code and (occ.method or "").startswith("exact")):
            head = title.split(",")[0].strip()  # 'Software Engineer Intern, AI and Automation'
            # a bare one-word head ('Analyst, AI and Data Engineering') is too generic to name an occupation
            alt = self.occupations.normalize(head) if len(head.split()) >= 2 else None
            if alt and alt.code and (alt.confidence or 0) > ((occ.confidence or 0) if occ and occ.code else 0):
                occ, occ_len = alt, len(head)
        if occ and occ.code:
            occ.evidence = [_rev(doc_id, text, title_start, title_start + occ_len, 0.7)] if title_start >= 0 else []
        return Experience(
            title=title, company=company, start_date=m.group("start").strip(), end_date=m.group("end").strip(),
            duration_months=months_between(m.group("start"), m.group("end")),
            description=desc[:2000] if desc else None, skills=merge_mentions(exp_skills),
            occupation=occ if occ and occ.code else None,
            evidence=[_rev(doc_id, text, ev_start, m.end(), 0.75 if title else 0.5)],
        )

    @staticmethod
    def _total_months(exps: list[Experience]) -> int:
        months: set[tuple[int, int]] = set()
        for e in exps:
            a, b = _parse_point(e.start_date or "", False), _parse_point(e.end_date or "", True)
            if not a or not b or (b[0] - a[0]) > 50:
                continue
            y, mo = a
            while (y, mo) <= b:
                months.add((y, mo))
                mo += 1
                if mo > 12:
                    y, mo = y + 1, 1
        return len(months)

    # -- education --------------------------------------------------------------
    def _education(self, text: str, doc_id: str, sections) -> list[Education]:
        """One entry per degree mention, each confined to its own block of the section.

        Layout is read from the section: a school before the first degree means 'School (dates) / Degree' blocks,
        otherwise 'Degree / School (dates)'. A degree takes the school on its layout side and never reaches past a
        neighbouring degree, and its year comes only from its own block — so one school or date range is never
        handed to two degrees.
        """
        secs = [s for s in sections if s.name == "education"]
        spans: list[list[int]] = []
        for sec in secs:  # a field like 'Education' on its own line can split one section into two: rejoin them
            if spans and sec.start - spans[-1][1] <= 25:
                spans[-1][1] = sec.end
            else:
                spans.append([sec.start, sec.end])
        spans = spans or [[0, len(text)]]
        out = []
        for lo, hi in spans:
            found = list(DEGREE_LONG.finditer(text, lo, hi))
            if secs:
                found += list(DEGREE_SHORT.finditer(text, lo, hi))
                if not found:
                    found += list(CERTIFICATE_EDU.finditer(text, lo, hi))
            degs = []
            for m in sorted(found, key=lambda x: (x.start(), -x.end())):
                if any(d.start() <= m.start() < d.end() for d in degs):
                    continue
                if degs and re.fullmatch(r"\s*\(\s*", text[degs[-1].end():m.start()]):
                    continue  # 'Bachelor of Laws (LLB)' — the abbreviation restates the same degree
                degs.append(m)
            if not degs:
                continue
            insts = [i for i in INSTITUTION.finditer(text, lo, hi)
                     if not any(d.start() <= i.start() < d.end() for d in degs)]

            def pair(inst_first: bool) -> list:
                out_ = []
                for k, m in enumerate(degs):
                    prev_end = degs[k - 1].end() if k else lo
                    nxt = degs[k + 1].start() if k + 1 < len(degs) else hi
                    if inst_first:
                        cands = [i for i in insts if prev_end <= i.start() < m.start()]
                        out_.append((m, cands[-1] if cands else None))
                    else:
                        cands = [i for i in insts if m.end() <= i.start() < nxt]
                        out_.append((m, cands[0] if cands else None))
                return out_

            def fit(ps: list) -> tuple:  # more degrees with a school, then schools closer to their degree
                gap = sum(abs(i.start() - m.end()) if i.start() > m.start() else abs(m.start() - i.end())
                          for m, i in ps if i)
                return (sum(1 for _, i in ps if i), -gap)

            default_inst_first = bool(insts) and insts[0].start() < degs[0].start()
            a_, b_ = pair(default_inst_first), pair(not default_inst_first)
            inst_first = default_inst_first if fit(a_) >= fit(b_) else not default_inst_first
            pairs = a_ if inst_first == default_inst_first else b_

            def anchor(pos: int) -> int:  # start of the line, unless the 'line' is a flattened paragraph
                ls = text.rfind("\n", lo, pos) + 1 if text.rfind("\n", lo, pos) >= 0 else lo
                le = text.find("\n", pos, hi)
                return ls if (le if le >= 0 else hi) - ls < 200 else pos

            for k, (m, inst) in enumerate(pairs):
                if inst_first:
                    b0 = anchor(inst.start()) if inst else anchor(m.start())
                    if k + 1 < len(pairs):
                        nm, ni = pairs[k + 1]
                        b1 = anchor(ni.start()) if ni else anchor(nm.start())
                    else:
                        b1 = hi
                else:
                    b0 = anchor(m.start())
                    b1 = anchor(pairs[k + 1][0].start()) if k + 1 < len(pairs) else hi
                b0, b1 = max(lo, min(b0, m.start())), max(b1, m.end())
                block = text[b0:b1]
                deg = m.group("deg")
                level = next((lvl for pat, lvl in DEGREE_LEVEL if pat.search(deg)), None)
                # field: only from the rest of the degree's own line, before any school on it
                # (may run past the section end: a field like 'Education' on its own is sometimes taken for a header)
                line_end = text.find("\n", m.end(), m.end() + 120)
                nd = DEGREE_LONG.search(text, m.end(), m.end() + 120)
                nxt_deg = pairs[k + 1][0].start() if k + 1 < len(pairs) else (nd.start() if nd else len(text))
                tail_end = min([x for x in (line_end if line_end >= 0 else m.end() + 120, nxt_deg,
                                            inst.start() if inst and inst.start() > m.end() else len(text)) if x > m.end()]
                               or [m.end()])
                tail = text[m.end():tail_end]
                fa = FIELD_AFTER.search(tail)
                fm = FIELD.search(tail[:80])
                fd = FIELD_DIRECT.match(tail)
                field = fa.group("field") if fa else (fm.group("field") if fm else
                                                       (fd.group("field") if fd and not INSTITUTION_WORD.search(
                                                           fd.group("field")) else None))
                if not field and inst and inst.start() > m.end():  # 'Associate of Arts : Irvine Valley College - Finance'
                    fi = FIELD_AFTER.search(text[inst.end():inst.end() + 80])
                    if fi and fi.group("field") not in {"City", "State", "USA", "United States"}:
                        field = fi.group("field")
                if not field and re.search(r"\b(?:of|in)\s+[A-Z]", deg):
                    field = re.split(r"\s+(?:of|in)\s+", deg, maxsplit=1)[-1]
                # year: the date nearest the degree (or its school) within the entry — a range counts as its end
                # (graduation / expected graduation); a year just before the degree ('May 2009  Master of …') counts
                prev_stop = max([pairs[k - 1][0].end()] + ([pairs[k - 1][1].end()] if pairs[k - 1][1] else [])) \
                    if k else lo
                ls = text.rfind("\n", lo, m.start()) + 1 if text.rfind("\n", lo, m.start()) >= 0 else lo
                flat = anchor(m.start()) == m.start() and ls < m.start()  # a flattened paragraph, not a line start
                pre = m.start() - 30 if flat else ls  # the degree's own line; in flattened text, 30 chars before it
                y0 = max(prev_stop, min(b0, pre))
                dates, covered = [], []
                for dr in DATE_RANGE.finditer(text, y0, b1):
                    yr = YEAR.search(dr.group("end"))
                    dates.append((dr.start(), dr.end(), yr.group(1) if yr else None))
                    covered.append((dr.start(), dr.end()))
                for ym in YEAR.finditer(text, y0, b1):
                    if not any(a <= ym.start() < b for a, b in covered):
                        dates.append((ym.start(), ym.end(), ym.group(1)))
                refs = [(m.start(), m.end())] + ([(inst.start(), inst.end())] if inst else [])

                def dist(d):
                    return min(max(r0 - d[1], d[0] - r1, 0) for r0, r1 in refs)

                year = min(dates, key=dist)[2] if dates else None
                ev_end = line_end if 0 <= line_end <= b1 and line_end - m.start() <= 250 else min(b1, m.start() + 250)
                out.append(Education(degree=deg, level=level, field=field,
                                     institution=inst.group("inst").strip() if inst else None, year=year,
                                     evidence=[_rev(doc_id, text, m.start(), ev_end, 0.75)]))
        return out

    # -- credentials ------------------------------------------------------------
    def _credentials(self, text: str, doc_id: str, smap: SectionMap) -> list[Credential]:
        found: list[tuple[int, int]] = []
        for pat in CREDENTIAL_PATTERNS:
            for m in pat.finditer(text):
                s, e = m.start(), m.end()
                name = text[s:e].strip()
                if len(name) < 2 or any(a <= s < b or s <= a < e for a, b in found):
                    continue
                if smap(s) == "education" and "Certif" not in name:
                    continue
                found.append((s, s + len(name)))
        for sec in smap.of("certifications"):
            # one per line, or several on one line separated by '|', ';' or '•'
            for seg in re.finditer(r"[^\n|;•]+", text[sec.start:sec.end]):
                s = sec.start + seg.start() + (len(seg.group()) - len(seg.group().lstrip(" •*-\t")))
                name = seg.group().strip(" •*-\t")
                if name and len(name.split()) <= 12 and not any(a <= s < b for a, b in found):
                    found.append((s, s + len(name)))
        out = []
        for s, e in sorted(found):
            out.append(Credential(name=re.sub(r"\s+", " ", text[s:e]).strip(),
                                  evidence=[_rev(doc_id, text, s, e, 0.7)]))
        return out

    # -- projects ---------------------------------------------------------------
    _BULLET_VERB = re.compile(r"^(?:[A-Z][a-z]+ed|Built|Led|Wrote|Ran|Made|Won|Taught|Drove|Set|Grew|Cut|Met|"
                              r"Worked|Using|Used|Responsible|Assisted|Helped)\b")

    @staticmethod
    def _date_only(line: str) -> bool:
        """'Aug 2025 – Dec 2025' (optionally followed by a location) on a line of its own."""
        dm = DATE_RANGE.search(line)
        if not dm or dm.start() > 3:
            return False
        rest = line[dm.end():].strip(" ,|·–—-\t")
        return not rest or bool(_LOCATION.match(rest))

    @staticmethod
    def _tech_stack(line: str) -> bool:
        """'Python, Apache Kafka, MongoDB, Docker' — a comma list of short names, not a sentence."""
        items = [x.strip() for x in re.split(r"\s*[,;]\s*", line.removeprefix("Tech").lstrip(" :|"))]
        return (len(items) >= 3 and not line.rstrip().endswith((".", "!", "?"))
                and all(x and len(x.split()) <= 4 for x in items))

    def _projects(self, text: str, doc_id: str, sections) -> list[Project]:
        """A short title-like line starts a project; action-verb bullets that follow are its description."""
        out: list[Project] = []
        for sec in [s for s in sections if s.name == "projects"]:
            current, prev_line = None, ""
            segs = [g for g in re.finditer(r"[^\n]+", text[sec.start:sec.end]) if len(g.group().strip(" •*-\t")) >= 4]
            for i, seg in enumerate(segs):
                raw = seg.group()
                line = raw.strip(" •*-\t")
                s = sec.start + seg.start() + (len(raw) - len(raw.lstrip(" •*-\t")))
                e = s + len(line)
                # dates on a line of their own belong to the project above; they never open one ('Aug 2025 – Dec 2025'
                # read as a project named 'Aug 2025')
                if self._date_only(line):
                    prev_line = ""
                    if current is not None:
                        current.evidence[0] = _rev(doc_id, text, current.evidence[0].start_offset, e, 0.6)
                    continue
                # 'Python, Kafka, MongoDB' under a title is that project's stack: its skills, not a new project
                if current is not None and not current.description and self._tech_stack(line):
                    prev_line = ""
                    current.evidence[0] = _rev(doc_id, text, current.evidence[0].start_offset, e, 0.6)
                    continue
                # a short line whose next line is only a date range is a title, whatever its first word
                titled_by_date = (i + 1 < len(segs) and len(line.split()) <= 14
                                  and not raw.lstrip().startswith(("•", "*", "-", "–", "▪", "●", "◦"))
                                  and self._date_only(segs[i + 1].group().strip(" •*-\t")))
                # a line-wrapped continuation: previous line ran long without closing punctuation, or this starts
                # in lower case
                wrapped = bool(prev_line) and not raw.lstrip().startswith(("•", "*", "-")) and (
                    line[0].islower() or (len(prev_line) >= 60 and not prev_line.rstrip().endswith((".", "!", "?"))))
                is_link = bool(_URL.search(line))
                prev_line = "" if is_link else line  # a link line ends a block; it never wraps into the next
                if is_link and not _LINK_LABEL.match(line):
                    continue  # 'Other Projects GitHub: github.com/…' — a link, not a project
                glyph = raw.lstrip().startswith(("•", "*", "-", "–", "▪", "●", "◦"))
                # 'Distributed Traffic Management System   Aug 2025 – Dec 2025' — a line ending in a date range is a
                # project title, even when its first word looks like an action verb ('Distributed')
                dm = DATE_RANGE.search(line)
                dated = bool(dm) and not glyph and dm.start() > 3 and len(line) - dm.end() <= 3
                # 'Name: what it does …' / 'Label | Name: …' opens a project whatever its length
                titled = dated or titled_by_date or (not wrapped and not is_link and not glyph
                                   and bool(re.match(r"^[^:|.]{3,80}?\s*[:|]\s", line))
                                   and not self._BULLET_VERB.match(line))
                is_bullet = not titled and (glyph or wrapped or is_link or bool(self._BULLET_VERB.match(line))
                                            or len(line.split()) > 14)
                if is_bullet and current is not None:
                    current.description = ((current.description or "") + " " + line).strip()[:800]
                    current.evidence[0] = _rev(doc_id, text, current.evidence[0].start_offset, e, 0.6)
                    continue
                if is_bullet or len(out) >= 15:
                    continue
                if dated:  # the dates are not part of the name; dashes inside it are ('Lights! — Movie Analytics')
                    title_line = line[:dm.start()].rstrip(" \t,|–—-")
                    parts = re.split(r"\s*[:|]\s*", title_line)
                    line = title_line
                elif titled_by_date:  # the whole line is the name, dashes included
                    parts = re.split(r"\s*[:|]\s*", line)
                else:
                    parts = re.split(r"\s*[:–—|]\s*|\s+-\s+", line)
                # 'Freelance Project | Property Maintenance Agent: Built …' — skip a generic leading label
                k = 1 if len(parts) > 1 and re.search(r"(?i)\bproject\b", parts[0]) and parts[1] else 0
                name = parts[k][:120]
                rest = line[line.find(parts[k]) + len(parts[k]):].lstrip(" :–—|-")
                current = Project(name=name, description=rest or None, evidence=[_rev(doc_id, text, s, e, 0.6)])
                out.append(current)
        for p in out:  # skills mentioned anywhere in the project's span
            ev = p.evidence[0]
            sk = self.skills.extract(text[ev.start_offset:ev.end_offset], doc_id, extractor="classical_gazetteer",
                                     base_offset=ev.start_offset)
            for sm in sk:
                for x in sm.evidence:
                    x.source_text = text[x.start_offset:x.end_offset]
            p.skills = merge_mentions(sk)
        return out

    # -- occupations ------------------------------------------------------------
    def _occupations(self, text: str, doc_id: str, sections, experiences: list[Experience]) -> list[OccupationMention]:
        out: dict[str, OccupationMention] = {}
        head = sections[0] if sections and sections[0].name == "header" else None
        if head:
            for seg in _SEG_SPLIT.split(text[head.start:head.end])[:3]:
                seg = seg.strip()
                if seg and len(seg.split()) <= 7 and not re.search(r"[@\d]", seg):
                    occ = self.occupations.normalize(seg.title() if seg.isupper() else seg)
                    if occ.code and (occ.method.startswith("exact") or occ.confidence >= 0.75):
                        s = text.find(seg, head.start)
                        occ.mention = seg
                        occ.evidence = [_rev(doc_id, text, s, s + len(seg), 0.7)]
                        out[occ.code] = occ
                        break
        if not out and head and len(text[head.start:head.end].split()) > 7:
            # flattened exports open with a headline title ('Network Administrator Network Administrator Stennis
            # Space Center, MS …'): the shortest leading 2–6 word phrase resolving to an O*NET title — an exact title
            # first, else a fuzzy match at the experience-title bar (>= 0.8)
            lead = text[head.start:head.start + 160]
            words = list(re.finditer(r"[A-Za-z][\w.+#&'/-]*", lead))
            if words and not lead[:words[0].start()].strip():
                best = None
                for n in range(2, min(6, len(words)) + 1):
                    phrase = lead[words[0].start():words[n - 1].end()]
                    if re.search(r"[@|\d,]", phrase):
                        break
                    occ = self.occupations.normalize(phrase)
                    if not occ.code:
                        continue
                    if (occ.method or "").startswith("exact"):
                        best = (occ, phrase)
                        break
                    if best is None and (occ.confidence or 0) >= 0.8:
                        best = (occ, phrase)
                if best:
                    occ, phrase = best
                    s = head.start + words[0].start()
                    occ.mention = phrase
                    occ.evidence = [_rev(doc_id, text, s, s + len(phrase), 0.65)]
                    out[occ.code] = occ
        if not out:  # 'SUMMARY  AI Engineer with 3+ years specializing in …'
            for sec in [s for s in sections if s.name == "summary"][:1]:
                m = SUMMARY_TITLE.match(text, sec.start + len(text[sec.start:sec.end]) - len(text[sec.start:sec.end].lstrip()))
                if m:
                    occ = self.occupations.normalize(m.group("title"))
                    if occ.code and (occ.method.startswith("exact") or occ.confidence >= 0.75):
                        occ.mention = m.group("title")
                        occ.evidence = [_rev(doc_id, text, m.start("title"), m.end("title"), 0.7)]
                        out[occ.code] = occ
        for exp in experiences:
            occ = exp.occupation
            # weak fuzzy mappings stay on the experience (visible, correctable) but do not become
            # the candidate's target occupations
            if occ and occ.code and not (occ.method or "").startswith("exact") and occ.confidence < 0.8:
                continue
            if exp.occupation and exp.occupation.code and exp.occupation.code not in out:
                out[exp.occupation.code] = exp.occupation
            elif exp.occupation and exp.occupation.code in out:
                out[exp.occupation.code].evidence.extend(exp.occupation.evidence)
        return list(out.values())


class ClassicalJobParser:
    def __init__(self, skills: Optional[SkillNormalizer] = None, occupations: Optional[OccupationNormalizer] = None):
        self.skills = skills or get_skill_normalizer()
        self.occupations = occupations or get_occupation_normalizer()

    def parse(self, job: Job) -> Job:
        text = job.description or ""
        doc_id = job.job_id
        sections = segment(text, JOB_HEADERS, default="intro")
        smap = SectionMap(sections)
        reqs: list[Requirement] = []
        sentences = [(m.start(), m.end()) for m in re.finditer(r"[^.\n;•]+[.\n;]?", text)]

        def sentence_of(off: int) -> tuple[int, int]:
            for s, e in sentences:
                if s <= off < e:
                    return s, e
            return off, off

        def importance(off: int) -> tuple[str, float]:
            sec = smap(off)
            s, e = sentence_of(off)
            sent = text[s:e]
            if sec == "preferred" or PREFERRED_CUE.search(sent):
                return "preferred", 0.8
            if sec == "requirements" or REQUIRED_CUE.search(sent):
                return "required", 0.85
            return "required", 0.6  # mentioned without explicit cue

        mentions = self.skills.extract(text, doc_id, extractor="classical_gazetteer", section_of=smap)
        by_skill: dict[str, Requirement] = {}
        for sm in mentions:
            off = sm.evidence[0].start_offset
            if smap(off) in {"benefits", "about"}:
                continue
            imp, conf = importance(off)
            k = sm.key
            if k in by_skill:
                r = by_skill[k]
                r.evidence.extend(sm.evidence)
                if r.importance == "preferred" and imp == "required" and conf > 0.6:
                    r.importance = "required"
                continue
            by_skill[k] = Requirement(kind="skill", importance=imp, text=sm.mention, skill=sm,
                                      evidence=list(sm.evidence))
            by_skill[k].evidence[0].confidence = conf
        reqs.extend(by_skill.values())
        # source-provided tags (e.g. RemoteOK) are normalised but have no span in the text
        for tag in job.skills:
            sm = self.skills.normalize(tag)
            if sm.canonical_id and sm.key not in by_skill:
                by_skill[sm.key] = Requirement(kind="skill", importance="required", text=tag, skill=sm, evidence=[])
                reqs.append(by_skill[sm.key])
        for m in YEARS_REQ.finditer(text):
            imp, conf = importance(m.start())
            years = float(m.group("min"))
            if years > 30:
                continue
            reqs.append(Requirement(kind="experience", importance=imp, text=text[m.start():m.end()], years=years,
                                    evidence=[_ev(doc_id, text, m.start(), m.end(), conf)]))
        for m in DEGREE_LONG.finditer(text):
            deg = m.group("deg")
            level = next((lvl for pat, lvl in DEGREE_LEVEL if pat.search(deg)), None)
            if not level:
                continue
            s, e = sentence_of(m.start())
            imp, conf = importance(m.start())
            reqs.append(Requirement(kind="education", importance=imp, text=text[s:e].strip()[:300] or deg,
                                    education_level=level, evidence=[_ev(doc_id, text, m.start(), m.end(), conf)]))
        for pat in CREDENTIAL_PATTERNS[:5]:
            for m in pat.finditer(text):
                imp, conf = importance(m.start())
                reqs.append(Requirement(kind="credential", importance=imp, text=m.group(0).strip(),
                                        evidence=[_ev(doc_id, text, m.start(), m.end(), conf)]))
        for sec in smap.of("responsibilities"):
            for seg in re.finditer(r"[^\n•]+", text[sec.start:sec.end]):
                line = seg.group().strip(" -*\t")
                if 15 <= len(line) <= 400:
                    s = sec.start + seg.start()
                    reqs.append(Requirement(kind="responsibility", importance="required", text=line,
                                            evidence=[_ev(doc_id, text, s, s + len(seg.group()), 0.6)]))
        for i, r in enumerate(reqs):
            r.requirement_id = f"req:{job.job_id}:{i}"
        occ = self.occupations.normalize(job.title) if job.title else None
        return job.model_copy(update={"requirements": reqs, "occupation": occ if occ and occ.code else None})
