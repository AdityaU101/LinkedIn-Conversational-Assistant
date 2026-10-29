"""Reusable Streamlit UI pieces."""
from __future__ import annotations

import html
from typing import Optional

import streamlit as st

from frontend.api_client import APIClient, APIError

SOURCE_LABEL = {"linkedin_kaggle": "LinkedIn (Kaggle 2023–24)", "datastax_linkedin": "LinkedIn (DataStax)",
                "remoteok": "Remote OK", "adzuna": "Adzuna", "linkedin_insights": "LinkedIn Insights"}
WORK_MODE_ICON = {"remote": "Remote", "hybrid": "Hybrid", "onsite": "On-site"}


@st.cache_resource
def client() -> APIClient:
    return APIClient()


def call(fn, *args, spinner: Optional[str] = None, **kw):
    """Run an API call, rendering errors instead of raising. Returns None on failure."""
    try:
        if spinner:
            with st.spinner(spinner):
                return fn(*args, **kw)
        return fn(*args, **kw)
    except Exception as e:
        # matched by name, not identity: after a Streamlit hot reload the cached client still raises the APIError
        # class of the module it was created from, which `except APIError` no longer catches
        if not isinstance(e, APIError) and type(e).__name__ != "APIError":
            raise
        if e.status == 0:
            st.error(f"{e.detail}")
        elif e.status == 404:
            st.warning(f"Not found: {e.detail}")
        elif e.status == 503:
            st.warning(f"Data not ready: {e.detail}")
        else:
            st.error(f"Request failed ({e.status}): {e.detail}")
        return None


def candidate_id() -> Optional[str]:
    return st.session_state.get("candidate_id")


def require_candidate() -> Optional[str]:
    cid = candidate_id()
    if not cid:
        st.info("Upload a resume on **Resume Analyzer** or pick a profile in the sidebar to continue.")
        st.page_link("pages/resume_analyzer.py", label="Go to Resume Analyzer")
    return cid


# One palette, one meaning, on every page: green = matched / done, amber = partial / related, grey = missing /
# not yet, red = blocked or destructive, blue = neutral information.
TONES = {"good": ("#e7f6ec", "#1a7f37"), "warn": ("#fff4e5", "#9a5b00"), "neutral": ("#eef2f7", "#475467"),
         "bad": ("#fdecec", "#b42318"), "info": ("#e8f1fd", "#1849a9")}
STATUS_TONE = {"evaluated": "good", "partially_evaluated": "warn", "not_evaluated": "neutral", "not_run": "neutral",
               "blocked": "bad", "parsed": "good", "imported": "neutral", "pending_approval": "warn",
               "approved": "info", "done": "good"}


def nav_link(page: str, label: str) -> None:
    """st.page_link that is skipped when the page runs outside the app's navigation (single-page tests)."""
    try:
        st.page_link(page, label=label)
    except Exception:
        pass


def badge_html(text, kind: str = "neutral", size: str = "0.82rem") -> str:
    bg, fg = TONES.get(kind, TONES["neutral"])
    return (f'<span style="background:{bg};color:{fg};padding:2px 9px;margin:2px 4px 2px 0;border-radius:12px;'
            f'font-size:{size};display:inline-block;white-space:nowrap">{html.escape(str(text))}</span>')


def badge(text, kind: str = "neutral") -> None:
    st.markdown(badge_html(text, kind), unsafe_allow_html=True)


def status_badge_html(status: Optional[str]) -> str:
    status = status or "not_run"
    return badge_html(status.replace("_", " "), STATUS_TONE.get(status, "neutral"))


def score_tone(score: Optional[float]) -> str:
    """Match score / coverage (0–1) -> the tone it is shown in."""
    if score is None:
        return "neutral"
    return "good" if score >= 0.7 else "warn" if score >= 0.4 else "neutral"


def chips_html(items: list, kind: str = "neutral", limit: Optional[int] = None) -> str:
    shown = items if limit is None else items[:limit]
    out = "".join(badge_html(i, kind) for i in shown)
    if limit is not None and len(items) > limit:
        out += f'<span style="color:#667085;font-size:0.8rem">+{len(items) - limit} more</span>'
    return out


def chips(items: list[str], kind: str = "neutral", empty: str = "—", limit: Optional[int] = None) -> None:
    if not items:
        st.caption(empty)
        return
    st.markdown(chips_html(items, kind, limit), unsafe_allow_html=True)


def money(v) -> Optional[str]:
    return None if v is None else f"${v:,.0f}"


def salary_text(job: dict) -> Optional[str]:
    lo, hi = job.get("salary_min"), job.get("salary_max")
    if lo is None and hi is None:
        return None
    cur = job.get("salary_currency") or ""
    rng = money(lo) if lo == hi or hi is None else (f"{money(lo)} – {money(hi)}" if lo else f"up to {money(hi)}")
    return f"{rng} {cur}/yr (annualised)".strip()


def attribution(job: dict) -> None:
    url = job.get("source_url")
    src = SOURCE_LABEL.get(job.get("source"), job.get("source"))
    if job.get("source") == "remoteok":
        st.markdown(f"Source: **[Remote OK]({url})** — job listing provided by Remote OK" if url else
                    "Source: **Remote OK**")
    elif url:
        st.markdown(f"Source: [{src}]({url})")
    else:
        st.caption(f"Source: {src}")


def evidence_list(items: list[dict], title: str) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    for ev in items:
        txt = ev.get("source_text") or ev.get("text") or ""
        meta = []
        if ev.get("extractor"):
            meta.append(ev["extractor"])
        if ev.get("start_offset") is not None:
            meta.append(f"chars {ev['start_offset']}–{ev['end_offset']}")
        if ev.get("confidence") is not None:
            meta.append(f"conf {ev['confidence']:.2f}")
        st.markdown(f"> “{html.escape(txt[:300])}”  \n<small>{' · '.join(meta)}</small>", unsafe_allow_html=True)


def empty_state(text: str) -> None:
    st.markdown(f"<div style='padding:1.2rem;border:1px dashed #cbd5e1;border-radius:10px;color:#475569;"
                f"text-align:center'>{html.escape(text)}</div>", unsafe_allow_html=True)


def not_evaluated(label: str = "Not evaluated") -> None:
    badge(label, "neutral")
