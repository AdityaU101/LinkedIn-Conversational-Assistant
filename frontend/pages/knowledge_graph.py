import pandas as pd
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

import html

from frontend.components.ui import call, client, require_candidate

st.title("Knowledge Graph")
cid = require_candidate()
if not cid:
    st.stop()

COLORS = {"Candidate": "#7c3aed", "Skill": "#2563eb", "Occupation": "#059669", "JobPosting": "#d97706",
          "Company": "#6b7280", "Requirement": "#f59e0b", "Experience": "#db2777", "Credential": "#0891b2",
          "Project": "#65a30d", "Education": "#9333ea", "Location": "#64748b", "Evidence": "#94a3b8"}
ALL = list(COLORS)

c1, c2, c3 = st.columns([3, 2, 1])
labels = c1.multiselect("Show node types", ALL, default=["Candidate", "Skill", "Occupation", "JobPosting", "Requirement",
                                                         "Experience", "Credential", "Company"])
jobs = {}
try:
    jobs = {r["job"]["job_id"]: f"#{r['rank']} {r['job']['title']}" for r in client().results(cid)["results"]}
except Exception:
    pass
job_id = c2.selectbox("Include job", [None] + list(jobs), format_func=lambda x: "— none —" if x is None else jobs[x],
                      index=(list(jobs).index(st.session_state["selected_job"]) + 1)
                      if st.session_state.get("selected_job") in jobs else 0)
max_nodes = c3.slider("Max nodes", 20, 300, 120)

g = call(client().graph, cid, job_id, labels + ["Candidate"], max_nodes, spinner="Querying the graph…")
if not g:
    st.stop()
shown = [l for l in ALL if l in set(labels) | {"Candidate"} and any(n["label"] == l for n in g["nodes"])]
# colour legend for the node types on screen
st.markdown(" ".join(f"<span style='display:inline-block;margin:0 .9rem .3rem 0;font-size:.86rem;white-space:nowrap'>"
                     f"<span style='display:inline-block;width:.75rem;height:.75rem;border-radius:50%;background:"
                     f"{COLORS[l]};margin-right:.35rem'></span>{html.escape(l)}</span>" for l in shown)
            + f"<span style='color:#667085;font-size:.82rem'>· {len(g['nodes'])} nodes, {len(g['edges'])} links · "
              "hover a link to see what it means · dashed = inferred</span>", unsafe_allow_html=True)


def label_of(n):
    return (n.get("name") or n.get("title") or n.get("text") or n.get("degree") or n["id"])[:40]


nodes = [Node(id=n["id"], label=label_of(n), size=22 if n["label"] in ("Candidate", "JobPosting") else 13,
              color=COLORS.get(n["label"], "#999"), title=f"{n['label']}: {label_of(n)}") for n in g["nodes"]]
# the relation name is a hover tooltip, not a label: drawn on every link they pile up and hide the graph
edges = [Edge(source=e["src"], target=e["dst"], title=e["type"].replace("_", " ").lower(),
              color="#d1d5db" if (e.get("props") or {}).get("inferred") else "#9ca3af",
              dashes=bool((e.get("props") or {}).get("inferred"))) for e in g["edges"]]
cfg = Config(height=720, directed=True, physics=True, hierarchical=False)
cfg.width = "100%"                 # fill the column instead of a fixed pixel width
if cfg.groups is None:
    del cfg.groups                 # vis-network rejects `groups: null`
left, right = st.columns([5, 2])
with left:
    clicked = agraph(nodes=nodes, edges=edges, config=cfg)
with right:
    st.subheader("Inspector")
    node = next((n for n in g["nodes"] if n["id"] == clicked), None)
    if not node:
        st.caption("Click a node to inspect it and its evidence.")
    else:
        st.markdown(f"**{node['label']}** — {label_of(node)}")
        st.json({k: v for k, v in node.items() if k not in ("label",)}, expanded=False)
        rel = [e for e in g["edges"] if clicked in (e["src"], e["dst"])]
        rows = [{"relation": e["type"], "direction": "out" if e["src"] == clicked else "in",
                 "other": label_of(next((n for n in g["nodes"] if n["id"] == (e["dst"] if e["src"] == clicked else e["src"])),
                                        {"id": "?"})),
                 "inferred": (e.get("props") or {}).get("inferred"), "method": (e.get("props") or {}).get("method"),
                 "confidence": (e.get("props") or {}).get("confidence"),
                 "evidence_ids": ", ".join((e.get("props") or {}).get("evidence_ids", [])[:3])} for e in rel]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        ev_ids = sorted({i for e in rel for i in (e.get("props") or {}).get("evidence_ids", [])})[:5]
        for eid in ev_ids:
            ev = call(client().evidence, eid)
            if ev:
                st.markdown(f"> “{ev['text'][:300]}”  \n<small>{ev['extractor']} · {ev['document_id']} · chars "
                            f"{ev['start']}–{ev['end']}</small>", unsafe_allow_html=True)

if job_id and g.get("paths"):
    st.subheader("Explanation paths (candidate → job)")
    for p in g["paths"][:15]:
        what = p.get("skill") or p.get("occupation")
        line = " → ".join(p["path"])
        extra = f" (inferred via {p.get('via')}, {p.get('method')})" if p["type"] == "related_skill_bridge" else ""
        st.markdown(f"- **{p['type']}** · {what}{extra}  \n  <small>{line}</small>", unsafe_allow_html=True)
with st.expander("Related occupations (O*NET, via your occupations)"):
    st.dataframe(pd.DataFrame(g.get("related_occupations") or []), hide_index=True, use_container_width=True)
