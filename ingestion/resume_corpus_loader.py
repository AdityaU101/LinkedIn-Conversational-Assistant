"""Loader for resume_corpus (Jiechieu & Tsopze, 2020).

Actual structure (verified by scripts/inspect_data.py):
* resume_samples.zip/resume_samples.txt — records ``id:::labels:::text`` with
  ``;``-separated raw occupation labels, separated by ``::::::`` lines.
* resumes_corpus.zip — ``NNNNN.txt`` resume + ``NNNNN.lab`` normalised labels.
* normlized_classes.txt — ``raw occupation title:Class[,Class]`` expert mappings.
* skills_it.txt — same record layout as resume_samples.txt (a resume subset).
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import re

from common.config import get_settings
from common.text import strip_html

SOURCE = "resume_corpus"


@dataclass
class CorpusResume:
    resume_id: str
    text: str
    labels: list[str] = field(default_factory=list)       # normalised classes (.lab)
    raw_labels: list[str] = field(default_factory=list)   # raw occupation titles (samples file)
    source: str = SOURCE


_HTML = re.compile(r"</?[a-zA-Z][^>]*>")


def _clean(text: str) -> str:
    """Indeed exports carry search-highlight markup (<span class="hl">Java</span>): keep the words only."""
    return strip_html(text) if _HTML.search(text) else text


def base_dir() -> Path:
    return get_settings().path("raw") / "resume_corpus"


def _parse_record_lines(lines: Iterator[str]) -> Iterator[tuple[str, list[str], str]]:
    for line in lines:
        line = line.strip()
        if not line or set(line) == {":"}:
            continue
        parts = line.split(":::", 2)
        if len(parts) == 3:
            rid, labels, text = parts
            yield rid.strip(), [l.strip() for l in labels.split(";") if l.strip()], text.strip()


def iter_samples(limit: Optional[int] = None, base: Optional[Path] = None) -> Iterator[CorpusResume]:
    base = base or base_dir()
    with zipfile.ZipFile(base / "resume_samples.zip") as zf, zf.open("resume_samples.txt") as fh:
        lines = (raw.decode("utf-8", "replace") for raw in fh)
        for i, (rid, raw_labels, text) in enumerate(_parse_record_lines(lines)):
            if limit and i >= limit:
                return
            yield CorpusResume(resume_id=f"{SOURCE}:sample:{rid}", text=_clean(text), raw_labels=raw_labels)


def iter_corpus(limit: Optional[int] = None, base: Optional[Path] = None) -> Iterator[CorpusResume]:
    """Resumes with expert-normalised class labels (.lab files)."""
    base = base or base_dir()
    with zipfile.ZipFile(base / "resumes_corpus.zip") as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".txt"))
        for i, name in enumerate(names):
            if limit and i >= limit:
                return
            stem = name[:-4]
            lab = stem + ".lab"
            labels = []
            if lab in zf.NameToInfo:
                labels = [l.strip() for l in zf.read(lab).decode("utf-8", "replace").splitlines() if l.strip()]
            text = zf.read(name).decode("utf-8", "replace").strip()
            yield CorpusResume(resume_id=f"{SOURCE}:{Path(stem).name}", text=_clean(text), labels=labels)


def normalized_classes(base: Optional[Path] = None) -> list[tuple[str, list[str]]]:
    """Expert mappings raw occupation title -> normalised class(es)."""
    base = base or base_dir()
    out = []
    for line in (base / "normlized_classes.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        raw, classes = line.rsplit(":", 1)
        cls = [c.strip() for c in classes.split(",") if c.strip()]
        if raw.strip() and cls:
            out.append((raw.strip(), cls))
    return out


def iter_skills_it(limit: Optional[int] = None, base: Optional[Path] = None) -> Iterator[CorpusResume]:
    base = base or base_dir()
    with open(base / "skills_it.txt", encoding="utf-8", errors="replace") as fh:
        for i, (rid, raw_labels, text) in enumerate(_parse_record_lines(fh)):
            if limit and i >= limit:
                return
            yield CorpusResume(resume_id=f"{SOURCE}:skills_it:{rid}", text=_clean(text), raw_labels=raw_labels)
