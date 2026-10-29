import html
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components.ui import (SOURCE_LABEL, STATUS_TONE, badge_html, call, candidate_id, chips, client, empty_state,
                                    nav_link, score_tone)

st.title("Dashboard")
cid = candidate_id()
d = call(client().dashboard, cid, spinner="Loading dashboard…")
if not d:
    st.stop()

c = d.get("candidate")
cov = d.get("mean_required_coverage")
BLUES = ["#0a66c2", "#70b5f9", "#a8d4ff", "#d0e8ff", "#e8f3ff"]
ACTIVITY = {"recommendation": ("Recommendations", "info"), "resume_upload": ("Resume uploaded", "good"),
            "resume_reparse": ("Resume re-parsed", "good"), "profile_update": ("Profile corrected", "warn"),
            "resume_delete": ("Profile deleted", "neutral")}


def tile(key: str, height: int):
    """One bento tile: a bordered box of fixed height, so tiles in a row line up."""
    return st.container(border=True, height=height, key=f"bento-{key}")


def stat(label: str, value, note: str = "") -> None:
    st.markdown(f"<div class='om-stat'><div class='om-stat-label'>{html.escape(label)}</div>"
                f"<div class='om-stat-value'>{html.escape(str(value))}</div>"
                f"<div class='om-stat-note'>{html.escape(note)}</div></div>", unsafe_allow_html=True)


# ---- row 1: profile (wide) · coverage (hero) · three small stats
a, b, s = st.columns([2, 1, 1])
with a, tile("profile", 344):
    st.subheader("Your profile")
    if not c:
        empty_state("No active profile. Upload a resume to get started.")
        st.page_link("pages/resume_analyzer.py", label="Upload a resume")
    else:
        r = c.get("resume") or {}
        status = r.get("status", "imported")
        st.markdown(f"**{html.escape(r.get('original_filename') or c['source_document'])}** "
                    + badge_html(status, STATUS_TONE.get(status, "neutral"))
                    + (badge_html("corrected by you", "warn") if c["user_corrected"] else ""), unsafe_allow_html=True)
        years = c["experience_years"]
        occ = (c["occupations"] or [{}])[0].get("title")
        st.markdown(" · ".join(x for x in [f"{years:g} years of experience" if years is not None else None,
                                           f"closest occupation: **{occ}**" if occ else None] if x)
                    or "No dated roles or occupation found.")
        st.markdown("**Skills**")
        chips(c["top_skills"], "info", limit=14)
        with st.expander("How this was computed"):
            st.caption(f"Extractors: {', '.join(r.get('extractors') or ['classical'])}. Experience is computed from "
                       "dated roles only.")
            if c["occupations"]:
                st.dataframe(pd.DataFrame(c["occupations"]), hide_index=True, use_container_width=True)
with b, tile("hero", 344):
    st.markdown(f"<div class='om-hero'><div class='om-hero-label'>Required-skill coverage</div>"
                f"<div class='om-hero-value'>{f'{cov:.0%}' if cov is not None else '—'}</div>"
                f"<div class='om-hero-note'>Share of each recommended job's required skills that your resume "
                f"shows, averaged over your top 10 matches.</div></div>", unsafe_allow_html=True)
with s:
    with tile("stat-jobs", 104):
        stat("Indexed jobs", f"{d['jobs_indexed']:,}")
    with tile("stat-skills", 104):
        stat("Skills in your resume", c["skills"] if c else "—")
    with tile("stat-profiles", 104):
        stat("Profiles", d["candidates"])

# ---- row 2: top recommendations (wide) · job sources
a, b = st.columns([2, 1])
with a, tile("recs", 400):
    st.subheader("Top recommendations")
    recs = d.get("top_recommendations")
    if not recs:
        empty_state("No recommendations yet.")
        if c:
            st.page_link("pages/job_recommendations.py", label="Get recommendations")
    else:
        rows = []
        for r in recs[:5]:
            cv = r.get("coverage_required")
            rows.append(
                f"<div class='om-row'><span class='om-rank'>{r['rank']}</span>"
                f"<div class='om-row-main'><b>{html.escape(r['title'] or '')}</b><br>"
                f"<small>{html.escape(r.get('company') or '—')} · "
                f"{html.escape(SOURCE_LABEL.get(r['source'], r['source']) or '')}</small></div>"
                f"<div>{badge_html(f'{r['score']:.2f} match', score_tone(r['score']))}"
                f"{badge_html(f'{cv:.0%} skills', score_tone(cv)) if cv is not None else ''}</div></div>")
        st.markdown("".join(rows), unsafe_allow_html=True)
        nav_link("pages/job_recommendations.py", "See all recommendations")
        with st.expander("How this was ranked"):
            st.caption(f"Last run: {d['last_run']['created_at']} · ranker: {d['last_run']['ranker']}")
with b, tile("sources", 400):
    st.subheader("Job sources")
    src = pd.DataFrame([{"source": SOURCE_LABEL.get(k, k), "jobs": v} for k, v in d["job_sources"].items()])
    st.plotly_chart(px.pie(src, names="source", values="jobs", hole=0.6, color_discrete_sequence=BLUES).update_layout(
        height=290, margin=dict(t=6, b=6, l=6, r=6), legend=dict(orientation="h", y=-0.05)), use_container_width=True)

# ---- row 3: work mode · recent activity (wide)
a, b = st.columns([1, 2])
with a, tile("modes", 330):
    st.subheader("Work mode")
    wm = pd.DataFrame([{"work mode": k, "jobs": v} for k, v in d["work_modes"].items()])
    st.plotly_chart(px.bar(wm, x="work mode", y="jobs", color_discrete_sequence=BLUES).update_layout(
        height=200, margin=dict(t=6, b=6, l=6, r=6), xaxis_title=None), use_container_width=True)
    st.caption("'unknown' = the source gave no work-mode signal; values are never guessed.")
with b, tile("activity", 330):
    st.subheader("Recent activity")
    act = d.get("recent_activity") or []
    if not act:
        st.caption("Nothing yet.")
    rows = []
    for x in act[:10]:
        label, tone = ACTIVITY.get(x["kind"], (x["kind"].replace("_", " "), "neutral"))
        summary = re.sub(r"\s*\([^\s()]+\)", "", x["summary"])     # drop '(ranker+names)' style internals
        rows.append(f"<div class='om-row'><div class='om-row-main'>{badge_html(label, tone)} "
                    f"{html.escape(summary)}</div><small class='om-when'>{html.escape(x['created_at'][:16])}</small></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)
