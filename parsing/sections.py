"""Section segmentation for resumes and job descriptions.

Resumes arrive in different layouts: newline-separated (PDF/DOCX), runs of
spaces (Kaggle Resume.csv) or a single line (resume_corpus). A segment is a
piece of text between newlines or runs of >= 3 spaces; a short segment that
matches a header vocabulary starts a new section. If no header is found, a
fallback looks for distinctive inline headers.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

RESUME_HEADERS: dict[str, list[str]] = {
    "summary": ["summary", "professional summary", "profile", "professional profile", "objective", "career objective",
                "about me", "executive summary", "career overview", "overview", "highlights", "career summary"],
    "experience": ["experience", "work experience", "professional experience", "employment history", "work history",
                   "employment", "career history", "relevant experience", "experience details"],
    "education": ["education", "education and training", "academic background", "academic qualifications",
                  "educational background", "education & training", "academics"],
    "skills": ["skills", "technical skills", "core competencies", "key skills", "skill highlights", "competencies",
               "core qualifications", "areas of expertise", "expertise", "computer skills", "technical proficiencies",
               "skills & abilities", "skills and abilities", "qualifications", "tools", "technologies",
               "summary of skills", "skills summary", "technical summary", "software", "it skills"],
    "certifications": ["certifications", "certification", "certificates", "licenses", "licenses and certifications",
                       "certifications/licenses", "certifications and licenses", "professional certifications",
                       "accreditations"],
    "projects": ["projects", "academic projects", "personal projects", "key projects", "selected projects"],
    "awards": ["awards", "honors", "achievements", "accomplishments", "honors and awards"],
    "publications": ["publications", "presentations"],
    "languages": ["languages"],
    "interests": ["interests", "hobbies", "activities", "activites", "extracurricular activities",
                  "leadership", "leadership activities"],
    "volunteer": ["volunteer", "volunteer experience", "community service", "affiliations",
                  "professional affiliations", "groups"],
    "additional": ["additional information", "additional skills", "other information", "links", "references"],
}

JOB_HEADERS: dict[str, list[str]] = {
    "requirements": ["requirements", "qualifications", "required qualifications", "minimum qualifications",
                     "basic qualifications", "what you'll need", "what you need", "what we're looking for",
                     "what we are looking for", "who you are", "must have", "must haves", "required skills",
                     "skills and qualifications", "skills & qualifications", "job requirements", "your profile",
                     "you have", "experience", "education", "knowledge, skills and abilities", "required experience",
                     "minimum requirements", "key qualifications", "about you", "skills"],
    "preferred": ["preferred qualifications", "preferred", "nice to have", "nice-to-have", "bonus points", "bonus",
                  "pluses", "desired qualifications", "preferred skills", "preferred experience", "desired skills",
                  "additional qualifications", "it's a plus if you have", "extra credit"],
    "responsibilities": ["responsibilities", "key responsibilities", "duties", "job duties", "what you'll do",
                         "what you will do", "the role", "your role", "role overview", "essential functions",
                         "essential duties", "duties and responsibilities", "job responsibilities", "day to day",
                         "in this role you will", "job description", "position summary", "job summary"],
    "benefits": ["benefits", "what we offer", "perks", "compensation", "why join us", "pay", "salary"],
    "about": ["about us", "about the company", "who we are", "company overview", "our company", "about"],
}

_SEG = re.compile(r"\n+|[ \t ]{3,}")


@dataclass
class Section:
    name: str
    start: int          # offset of first content char (after the header)
    end: int
    header_start: int
    header_text: str


def _lookup(vocab: dict[str, list[str]]) -> dict[str, str]:
    return {h: name for name, hs in vocab.items() for h in hs}


JOB_HEADER_KEYWORDS = [
    ("preferred", re.compile(r"prefer|nice[- ]to[- ]have|bonus|desired|plus")),
    ("requirements", re.compile(r"requirement|qualification|must have|skills|experience|education|knowledge|abilities|you bring|competenc")),
    ("responsibilities", re.compile(r"responsibilit|duties|what you.?ll do|accountabilit|functions|the role|day[- ]to[- ]day")),
    ("benefits", re.compile(r"benefit|perks|compensation|salary|pay range|what we offer")),
    ("about", re.compile(r"about|who we are|our company|our mission|equal opportunity|eeo")),
]
RESUME_HEADER_KEYWORDS = [
    ("experience", re.compile(r"experience|employment|work history")),
    ("education", re.compile(r"education|academic")),
    ("skills", re.compile(r"skills|competenc|proficienc|technolog")),
    ("certifications", re.compile(r"certif|licen")),
    ("projects", re.compile(r"project")),
]


def _fuzzy_header(key: str, rules) -> str | None:
    for name, pat in rules:
        if pat.search(key):
            return name
    return None


def segment(text: str, vocab: dict[str, list[str]], default: str = "header", fuzzy=None) -> list[Section]:
    lookup = _lookup(vocab)
    if fuzzy is None:
        fuzzy = JOB_HEADER_KEYWORDS if vocab is JOB_HEADERS else RESUME_HEADER_KEYWORDS
    heads: list[tuple[int, int, str, str]] = []
    pos = 0
    for m in list(_SEG.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        raw = text[pos:end]
        seg = raw.strip()
        if seg:
            key = re.sub(r"[\s:–\-•*_|]+$", "", seg.lower()).strip()
            key = re.sub(r"^[\s•*_|#]+", "", key)
            name = lookup.get(key) if len(key.split()) <= 6 else None
            if name is None and fuzzy and len(key.split()) <= 6 and (seg.rstrip().endswith(":") or seg.isupper()):
                name = _fuzzy_header(key, fuzzy)
            if name:
                s = pos + (len(raw) - len(raw.lstrip()))
                heads.append((s, s + len(seg), name, seg))
        if m:
            pos = m.end()
    if not heads:
        heads = _inline_headers(text, lookup)
    sections: list[Section] = []
    first = heads[0][0] if heads else len(text)
    if first > 0:
        sections.append(Section(default, 0, first, 0, ""))
    for i, (hs, he, name, raw) in enumerate(heads):
        nxt = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        sections.append(Section(name, he, nxt, hs, raw))
    return sections


_INLINE = ["Work Experience", "Professional Experience", "Education", "Skills", "Certifications/Licenses",
           "Certifications", "Additional Information", "Links", "Awards", "Groups", "Summary", "Projects",
           "Requirements", "Qualifications", "Responsibilities", "Preferred Qualifications", "Nice to have", "Benefits"]


def _inline_headers(text: str, lookup: dict[str, str]) -> list[tuple[int, int, str, str]]:
    out = []
    for h in _INLINE:
        name = lookup.get(h.lower())
        if not name:
            continue
        for m in re.finditer(rf"(?:(?<=\s)|^){re.escape(h)}(?=\s+[A-Z0-9]|:)", text):
            out.append((m.start(), m.end(), name, h))
    out.sort()
    dedup = []
    for item in out:  # drop overlapping matches ('Work Experience' vs 'Experience')
        if dedup and item[0] < dedup[-1][1]:
            continue
        dedup.append(item)
    return dedup


class SectionMap:
    """Offset -> section-name lookup."""

    def __init__(self, sections: list[Section]):
        self.sections = sections
        self._starts = [s.header_start for s in sections]

    def __call__(self, offset: int) -> str | None:
        i = bisect.bisect_right(self._starts, offset) - 1
        return self.sections[i].name if i >= 0 else None

    def of(self, name: str) -> list[Section]:
        return [s for s in self.sections if s.name == name]
