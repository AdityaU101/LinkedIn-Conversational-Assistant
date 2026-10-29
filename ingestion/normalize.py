"""Field normalisers used by every source loader.

Each function returns ``None`` when a value cannot be determined — values are
never guessed. Callers keep the raw value in ``Job.raw``.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Optional

from common.text import collapse_ws

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
_STATE_NAMES = {v.lower(): k for k, v in US_STATES.items()}


def is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def clean_str(v: Any) -> Optional[str]:
    if is_missing(v):
        return None
    s = collapse_ws(str(v))
    return s or None


_MASKED = re.compile(r"[*\s,.\-·/()]*\*[*\s,.\-·/()]*")


def is_masked(v: Any) -> bool:
    """True for values the source redacted with asterisks ('**** *********'). The real value is unknown."""
    return isinstance(v, str) and bool(_MASKED.fullmatch(v))


def parse_location(raw: Any) -> dict[str, Optional[str]]:
    """Parse free-text locations like 'Princeton, NJ', 'Brisbane, Queensland, Australia', 'United States'."""
    out = {"location": clean_str(raw), "city": None, "state": None, "country": None}
    s = out["location"]
    if not s:
        return out
    low = s.lower()
    if low in {"remote", "anywhere", "worldwide"}:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if low in {"united states", "usa", "us"}:
        out["country"] = "US"
        return out
    if len(parts) >= 2 and parts[-1].upper() in US_STATES:
        out.update(city=parts[0] if len(parts) > 1 else None, state=parts[-1].upper(), country="US")
    elif len(parts) >= 2 and parts[-1].lower() in _STATE_NAMES:
        out.update(city=parts[0], state=_STATE_NAMES[parts[-1].lower()], country="US")
    elif len(parts) >= 2 and parts[-1].lower() in {"united states", "usa", "us"}:
        st = parts[-2]
        code = st.upper() if st.upper() in US_STATES else _STATE_NAMES.get(st.lower())
        out.update(city=parts[0] if len(parts) == 3 else None, state=code, country="US")
    elif len(parts) == 1 and parts[0].lower() in _STATE_NAMES:
        out.update(state=_STATE_NAMES[parts[0].lower()], country="US")
    elif len(parts) >= 2:
        out.update(city=parts[0], country=parts[-1])
    return out


_PERIOD_FACTOR = {"HOURLY": 2080, "DAILY": 260, "WEEKLY": 52, "BIWEEKLY": 26, "MONTHLY": 12, "YEARLY": 1, "ANNUAL": 1}


def annualize_salary(amount: Any, period: Any) -> Optional[float]:
    """Annualise using standard full-time factors (2080 h/yr). None if unknown."""
    if is_missing(amount) or is_missing(period):
        return None
    try:
        val = float(amount)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    factor = _PERIOD_FACTOR.get(str(period).upper())
    return round(val * factor, 2) if factor else None


_EMPLOYMENT = {
    "full-time": "full_time", "full_time": "full_time", "full time": "full_time", "fulltime": "full_time",
    "part-time": "part_time", "part_time": "part_time", "part time": "part_time",
    "contract": "contract", "contractor": "contract", "temporary": "temporary", "temp": "temporary",
    "internship": "internship", "intern": "internship", "volunteer": "volunteer", "other": "other",
    "permanent": "full_time",
}


def normalize_employment_type(raw: Any) -> Optional[str]:
    s = clean_str(raw)
    return _EMPLOYMENT.get(s.lower()) if s else None


_SENIORITY = {
    "internship": "internship", "entry level": "entry", "entry": "entry", "associate": "associate",
    "mid-senior level": "mid_senior", "mid-senior": "mid_senior", "director": "director",
    "executive": "executive",
}
_TITLE_SENIORITY = [
    (re.compile(r"\b(intern|internship)\b", re.I), "internship"),
    (re.compile(r"\b(chief|cxo|ceo|cto|cfo|coo|vp|vice president|president)\b", re.I), "executive"),
    (re.compile(r"\b(director|head of)\b", re.I), "director"),
    (re.compile(r"\b(senior|sr\.?|lead|principal|staff|manager)\b", re.I), "mid_senior"),
    (re.compile(r"\b(junior|jr\.?|entry[- ]level|graduate|trainee)\b", re.I), "entry"),
]


def normalize_seniority(raw: Any) -> Optional[str]:
    s = clean_str(raw)
    return _SENIORITY.get(s.lower()) if s else None


def infer_seniority_from_title(title: Optional[str]) -> Optional[str]:
    """Rule-based fallback; used only when the source has no seniority field."""
    if not title:
        return None
    for pat, level in _TITLE_SENIORITY:
        if pat.search(title):
            return level
    return None


_REMOTE = re.compile(r"\b(fully remote|100% remote|remote[- ]first|work from home|wfh|remote)\b", re.I)
_HYBRID = re.compile(r"\bhybrid\b", re.I)
_ONSITE = re.compile(r"\b(on[- ]?site|in[- ]office|in person)\b", re.I)


def infer_work_mode(*texts: Optional[str], remote_flag: Any = None) -> Optional[str]:
    """Work mode from an explicit flag, else from title/location text. None if no signal."""
    if not is_missing(remote_flag):
        try:
            if float(remote_flag) == 1.0:
                return "remote"
        except (TypeError, ValueError):
            pass
    head = " ".join(t for t in texts if t)
    if not head:
        return None
    if _HYBRID.search(head):
        return "hybrid"
    if _REMOTE.search(head):
        return "remote"
    if _ONSITE.search(head):
        return "onsite"
    return None


def epoch_ms_to_date(v: Any) -> Optional[str]:
    if is_missing(v):
        return None
    try:
        return datetime.fromtimestamp(float(v) / 1000, timezone.utc).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def parse_date(v: Any) -> Optional[str]:
    if is_missing(v):
        return None
    if hasattr(v, "date"):
        try:
            return v.date().isoformat()
        except Exception:
            pass
    s = str(v).strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None


def normalize_title(title: Optional[str]) -> Optional[str]:
    """Light title clean-up for display/dedup; the raw title is preserved separately."""
    s = clean_str(title)
    if not s:
        return None
    s = re.sub(r"\s*[\(\[][^)\]]*(remote|hybrid|onsite|on-site)[^)\]]*[\)\]]", "", s, flags=re.I)
    s = re.sub(r"\s+-\s+(remote|hybrid)$", "", s, flags=re.I)
    return s.strip() or None


def normalize_company(name: Optional[str]) -> Optional[str]:
    s = clean_str(name)
    if not s:
        return None
    return re.sub(r",?\s+(inc\.?|llc|ltd\.?|corp\.?|corporation|co\.)$", "", s, flags=re.I).strip() or s
