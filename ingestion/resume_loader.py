"""Kaggle Resume Dataset loader (Resume.csv + per-category PDF folders)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from common.config import get_settings

SOURCE = "kaggle_resumes"


@dataclass
class ResumeRecord:
    resume_id: str
    category: str
    text: str
    pdf_path: Optional[Path]
    source: str = SOURCE


def base_dir() -> Path:
    return get_settings().path("raw") / "resume_dataset"


def categories(base: Optional[Path] = None) -> list[str]:
    """Discover categories from the folder layout (not from a hard-coded list)."""
    root = (base or base_dir()) / "data"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def iter_resumes(limit: Optional[int] = None, base: Optional[Path] = None,
                 category: Optional[str] = None) -> Iterator[ResumeRecord]:
    base = base or base_dir()
    df = pd.read_csv(base / "Resume.csv", usecols=["ID", "Resume_str", "Category"])
    if category:
        df = df[df["Category"] == category]
    for i, r in enumerate(df.itertuples(index=False)):
        if limit and i >= limit:
            return
        pdf = base / "data" / str(r.Category) / f"{r.ID}.pdf"
        yield ResumeRecord(resume_id=f"{SOURCE}:{r.ID}", category=str(r.Category),
                           text=str(r.Resume_str).strip(), pdf_path=pdf if pdf.exists() else None)
