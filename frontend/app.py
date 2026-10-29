"""OpenMatch (the LinkedIn Assistant project) — Streamlit frontend.

  streamlit run frontend/app.py        (backend: uvicorn backend.main:app --port 8000)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from frontend.components.ui import call, client  # noqa: E402

st.set_page_config(page_title="OpenMatch", page_icon=str(ROOT / "frontend" / "assets" / "logo_mark.svg"), layout="wide")
css = (ROOT / "frontend" / "styles" / "theme.css")
if css.exists():
    st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)

HERE = Path(__file__).parent
pages = {
    "Career": [
        st.Page(HERE / "pages/dashboard.py", title="Dashboard", default=True),
        st.Page(HERE / "pages/resume_analyzer.py", title="Resume Analyzer"),
        st.Page(HERE / "pages/job_recommendations.py", title="Job Recommendations"),
        st.Page(HERE / "pages/skill_gap.py", title="Skill Gap"),
        st.Page(HERE / "pages/knowledge_graph.py", title="Knowledge Graph"),
        st.Page(HERE / "pages/career_assistant.py", title="Career Assistant"),
    ],
    "Research": [
        st.Page(HERE / "pages/research_dashboard.py", title="Research Dashboard"),
        st.Page(HERE / "pages/annotation.py", title="Annotation"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### OpenMatch")
    st.caption("Explainable, open-weight job matching")
    health = call(client().health)
    if health:
        st.caption(f"API ✓ · {health.get('jobs_indexed', 0):,} jobs · graph: {health.get('graph_backend')}")
    profiles = call(client().list_resumes, 200) or {"items": []}
    items = profiles["items"]
    if items:
        labels = {p["candidate_id"]: f"{p.get('original_filename') or p['candidate_id']}  ·  {p['source']}" for p in items}
        ids = list(labels)
        cur = st.session_state.get("candidate_id")
        idx = ids.index(cur) if cur in ids else 0
        choice = st.selectbox("Active profile", ids, index=idx, format_func=lambda x: labels[x])
        if choice != cur:
            st.session_state["candidate_id"] = choice
            st.session_state.pop("selected_job", None)
    else:
        st.caption("No profiles yet — upload a resume.")
    # a conversation belongs to one sitting with one profile: switching profile or uploading a resume starts a
    # fresh one, and coming back to a profile does not bring its old conversation (or optimizer output) back
    if st.session_state.get("chat_profile") != st.session_state.get("candidate_id"):
        st.session_state["chat_profile"] = st.session_state.get("candidate_id")
        for k in ("chat", "opt", "outreach_res"):
            st.session_state.pop(k, None)
    st.divider()
    st.caption("Resume text is processed locally. Nothing is sent to hosted LLMs unless you enable it in "
               "configs/settings.yaml.")

# top-left of every page, above its title. Our own mark — the LinkedIn logo is a trademark and is not used
st.image(str(HERE / "assets" / "logo.svg"), width=190)
nav.run()
