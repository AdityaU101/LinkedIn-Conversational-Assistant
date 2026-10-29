import streamlit as st

from frontend.components.ui import call, client, require_candidate

st.title("Relevance Annotation")
st.caption("Judge pooled jobs for the active profile: 0 = irrelevant, 1 = partially relevant, 2 = highly relevant. "
           "Jobs are pooled from all ranking methods and shuffled; method names and scores are hidden to avoid bias. "
           "Judgments are appended to Data/annotations/relevance/judgments.csv and train LambdaMART once enough exist.")
cid = require_candidate()
if not cid:
    st.stop()
annotator = st.text_input("Annotator ID (required)", st.session_state.get("annotator", ""), max_chars=40)
st.session_state["annotator"] = annotator
if st.button("Load / refresh pool"):
    st.session_state["pool"] = call(client().relevance_pool, cid, spinner="Pooling results from all methods…")
pool = st.session_state.get("pool")
if not pool or pool.get("candidate_id") != cid:
    st.info("Load the pool to start judging.")
    st.stop()
done = pool.get("judged", {})
st.progress(len(done) / max(1, len(pool["jobs"])), text=f"{len(done)} / {len(pool['jobs'])} judged for this profile")
for j in pool["jobs"]:
    with st.container(border=True):
        st.markdown(f"**{j['title']}** — {j.get('company') or '?'} · {j.get('location') or ''}")
        with st.expander("Description"):
            st.write(j.get("description") or "")
        prev = done.get(j["job_id"])
        c1, c2 = st.columns([3, 1])
        label = c1.radio("Relevance", [0, 1, 2], horizontal=True, key=f"r-{j['job_id']}",
                         index=prev["label"] if prev else None,
                         format_func={0: "0 irrelevant", 1: "1 partial", 2: "2 highly relevant"}.get)
        if c2.button("Save", key=f"s-{j['job_id']}", disabled=label is None or not annotator.strip()):
            if call(client().add_judgment, cid, j["job_id"], int(label), annotator):
                done[j["job_id"]] = {"label": int(label), "annotator_id": annotator}
                st.toast("Saved")
