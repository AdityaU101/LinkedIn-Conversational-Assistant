import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components.ui import badge_html, call, client, empty_state, evidence_list, require_candidate

st.title("Skill Gap Analysis")
cid = require_candidate()
if not cid:
    st.stop()

options = {}
try:
    run = client().results(cid)
    options = {r["job"]["job_id"]: f"#{r['rank']} {r['job']['title']} — {r['job'].get('company') or '?'}" for r in run["results"]}
except Exception:
    pass
sel = st.session_state.get("selected_job")
if sel and sel not in options:
    options = {sel: sel, **options}
if not options:
    empty_state("Run Job Recommendations first, or open a job from its card.")
    st.page_link("pages/job_recommendations.py", label="Job Recommendations")
    st.stop()
ids = list(options)
job_id = st.selectbox("Target job", ids, index=ids.index(sel) if sel in ids else 0, format_func=options.get)
st.session_state["selected_job"] = job_id

res = call(client().gaps, cid, job_id, spinner="Comparing your skills with the job requirements…")
if not res:
    st.stop()
g = res["gap"]
n_req = len(g["matched_required"]) + len(g["missing_required"])

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    cov = g["coverage_required"]
    fig = go.Figure(go.Indicator(mode="gauge+number", value=(cov or 0) * 100, number={"suffix": "%"},
                                 title={"text": "Required-skill coverage"},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1a7f37"}}))
    st.plotly_chart(fig.update_layout(height=220, margin=dict(t=40, b=0, l=40, r=40)), use_container_width=True)
    if cov is None:
        st.caption("This posting lists no normalisable required skills.")
c2.metric("Matched required", f"{len(g['matched_required'])}/{n_req}")
c3.metric("Missing required", len(g["missing_required"]))
c4.metric("Missing preferred", len(g["missing_preferred"]))
if g.get("experience_required_years") is not None:
    have = g.get("candidate_experience_years")
    st.caption(f"Experience: posting asks for ≥ {g['experience_required_years']:g} years; your dated roles total "
               f"{have if have is not None else 'unknown'} years.")

t1, t2, t3, t4, t5 = st.tabs(["Matched", "Missing", "Transferable", "Your skills", "Related occupations"])


def skill_picker(items: list[dict], key: str):
    """Skills as chips; the chosen one is returned so its evidence can be shown underneath."""
    by_name = {m["name"]: m for m in items}
    picked = st.pills("Skills", list(by_name), key=key, label_visibility="collapsed")
    return by_name.get(picked)


def asked_for(m: dict) -> None:
    if m["job_evidence"]:
        evidence_list(m["job_evidence"], f"Where the posting asks for {m['name']}")
    else:
        st.caption("Source-provided tag (no text span).")


with t1:
    matched = g["matched_required"] + g["matched_preferred"]
    if not matched:
        st.caption("No explicit skill matches.")
    else:
        st.markdown(badge_html(f"{len(matched)} matched", "good")
                    + "<span style='color:#667085;font-size:.86rem'>Click a skill to see the evidence on both "
                      "sides.</span>", unsafe_allow_html=True)
        m = skill_picker(matched, f"pick-matched-{job_id}")
        if m:
            with st.container(border=True):
                st.markdown(f"**{m['name']}** " + badge_html("required" if m["importance"] == "required"
                                                              else "nice to have", "neutral"), unsafe_allow_html=True)
                a, b = st.columns(2)
                with a:
                    evidence_list(m["candidate_evidence"], "Your resume")
                with b:
                    evidence_list(m["job_evidence"], "Job posting")
with t2:
    for title, items, key in (("Required", g["missing_required"], "req"),
                              ("Nice to have", g["missing_preferred"], "pref")):
        st.markdown(f"**{title}** " + badge_html(f"{len(items)} missing", "neutral"), unsafe_allow_html=True)
        if not items:
            st.caption("Nothing missing here.")
            continue
        m = skill_picker(items, f"pick-missing-{key}-{job_id}")
        if m:
            with st.container(border=True):
                asked_for(m)
    if g["missing_required"] or g["missing_preferred"]:
        st.caption("Click a skill to see where the posting asks for it. Skills you have something related to are "
                   "listed under Transferable.")
with t3:
    st.caption("Related skills are inferred relations (O*NET co-occurrence or embedding similarity) — related, not identical.")
    rows = [{"missing skill": m["name"], "your related skill": b["candidate_skill_name"], "method": b["method"],
             "confidence": b["confidence"]} for m in g["transferable"] for b in m["bridges"]]
    if rows:
        st.markdown(badge_html(f"{len(g['transferable'])} partly covered", "warn"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1)})
    else:
        st.caption("No related-skill bridges for the missing skills.")
    if g["occupation_transferable"]:
        st.markdown("**Shared O*NET skill/knowledge importance (your occupation vs. target occupation, 1–5)**")
        ot = pd.DataFrame(g["occupation_transferable"])
        fig = go.Figure([go.Bar(name="your occupation", y=ot["name"], x=ot["candidate_importance"], orientation="h"),
                         go.Bar(name="target occupation", y=ot["name"], x=ot["target_importance"], orientation="h")])
        st.plotly_chart(fig.update_layout(barmode="group", height=320, margin=dict(t=10, b=10)), use_container_width=True)
with t4:
    df = pd.DataFrame(res["candidate_skills"])
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1)})
with t5:
    rel = pd.DataFrame(res["related_occupations"])
    if rel.empty:
        st.caption("No O*NET occupation mapped for your profile.")
    else:
        st.dataframe(rel, hide_index=True, use_container_width=True)
        st.caption("Source: O*NET related_occupations.csv (relatedness tiers).")
