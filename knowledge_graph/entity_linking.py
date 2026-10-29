"""Stable node IDs (entity linking to canonical identifiers)."""
from __future__ import annotations

from typing import Optional

from common.text import sha1, slug


def skill_node_id(canonical_id: str) -> str:
    return f"skill:{canonical_id}"


def occupation_node_id(code: str) -> str:
    return f"occ:{code}"


def company_node_id(name: Optional[str]) -> Optional[str]:
    if not name or slug(name) in {"unknown", "company-name"}:   # Kaggle resumes anonymise employers
        return None
    return f"company:{slug(name)}"


def location_node_id(city: Optional[str], state: Optional[str], country: Optional[str]) -> Optional[str]:
    parts = [p for p in (city, state, country) if p]
    return f"loc:{slug('-'.join(parts))}" if parts else None


def job_node_id(job_id: str) -> str:
    return f"job:{job_id}"


def candidate_node_id(candidate_id: str) -> str:
    return f"cand:{candidate_id}"


def fact_node_id(prefix: str, owner: str, *parts) -> str:
    return f"{prefix}:{sha1(owner + '|' + '|'.join(str(p) for p in parts))[:16]}"
