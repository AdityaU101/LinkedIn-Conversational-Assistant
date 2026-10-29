import copy

import pandas as pd
import streamlit as st

from frontend.components.ui import badge_html, call, candidate_id, chips, chips_html, client, empty_state

st.title("Resume Analyzer")

# open while there is no profile; afterwards it is one click away and the analysis has the page
with st.expander("Upload a resume" if not candidate_id() else "Upload a different resume",
                 expanded=not candidate_id()):
    c1, c2 = st.columns([3, 1])
    up = c1.file_uploader("PDF, DOCX or TXT (max 5 MB)", type=["pdf", "docx", "txt"])
    mode = c2.radio("LLM extraction", ["auto", "false", "true"], index=0,
                    format_func={"auto": "Auto (if local model)", "false": "Classical only",
                                 "true": "Require LLM"}.get,
                    help="Resume text is only ever sent to a LOCAL model (Ollama).")
    if up and st.button("Analyze resume", type="primary"):
        res = call(client().upload_resume, up.name, up.getvalue(), mode, spinner="Extracting and parsing…")
        if res:
            st.session_state["candidate_id"] = res["candidate_id"]
            if res.get("duplicate"):
                st.info("This exact file was already analysed — opened the existing profile.")
            else:
                st.success(f"Parsed with: {', '.join(res.get('extractors', []))}")
            for w in res.get("warnings") or []:
                st.warning(w)
            if res.get("injection_flags"):
                st.warning("The document contains instruction-like text (" + "; ".join(res["injection_flags"][:3]) +
                           "). It was treated purely as data.")
            st.rerun()

cid = candidate_id()
if not cid:
    empty_state("Upload a resume to see the structured analysis.")
    st.stop()

a = call(client().analysis, cid, spinner="Loading analysis…")
if not a:
    st.stop()
cand = a["candidate"]
s = a["stats"]
m = st.columns(6)
for col, (k, label) in zip(m, [("skills", "Skills"), ("experiences", "Roles"), ("education", "Education"),
                               ("credentials", "Credentials"), ("projects", "Projects"), ("evidence_spans", "Evidence spans")]):
    col.metric(label, s[k])
if st.session_state.get("reparse_note"):
    st.success(st.session_state.pop("reparse_note"))
years = cand.get("total_experience_years")
st.markdown((f"**{years:g} years** of experience" if years is not None else "Experience: unknown")
            + (" " + badge_html("corrected by you", "warn") if cand.get("user_corrected") else ""),
            unsafe_allow_html=True)
with st.expander("How this was computed"):
    st.caption(f"Extractor: `{cand.get('extractor')}`. Experience is the total of dated roles. Every item below "
               "keeps the exact resume text it was read from (its evidence span).")


def ev_text(item) -> str:
    evs = item.get("evidence") or []
    return " | ".join(e["source_text"][:120] for e in evs[:2])


def ev_span(item) -> str:
    evs = item.get("evidence") or []
    return ", ".join(f"{e['start_offset']}–{e['end_offset']}" for e in evs[:2])


tabs = st.tabs(["Skills", "Experience", "Education", "Credentials", "Projects", "Occupations", "Extracted text"])
edited = {}
with tabs[0]:
    st.caption("Untick wrong skills, or add skills you actually have (they are marked as user-asserted, not resume evidence).")
    df = pd.DataFrame([{"keep": True, "mention": x["mention"], "canonical": x.get("canonical_name"),
                        "taxonomy": x.get("taxonomy"), "method": x.get("method"), "confidence": x.get("confidence"),
                        "evidence": ev_text(x), "span": ev_span(x)} for x in cand["skills"]])
    edited["skills"] = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True,
                                      disabled=["canonical", "taxonomy", "method", "confidence", "evidence", "span"],
                                      column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1)},
                                      key="ed_skills")
with tabs[1]:
    df = pd.DataFrame([{"keep": True, "title": x.get("title"), "company": x.get("company"), "start": x.get("start_date"),
                        "end": x.get("end_date"), "months": x.get("duration_months"),
                        "occupation": (x.get("occupation") or {}).get("canonical_title"),
                        "skills used": ", ".join(sk.get("canonical_name") or sk["mention"] for sk in x.get("skills", [])),
                        "evidence": ev_text(x)} for x in cand["experiences"]])
    if df.empty:
        st.caption("No dated roles found.")
    for x in cand["experiences"]:
        with st.container(border=True):
            dates = " – ".join(str(v) for v in (x.get("start_date"), x.get("end_date")) if v)
            st.markdown(f"**{x.get('title') or 'Untitled role'}**" + (f" · {x['company']}" if x.get("company") else "")
                        + (f"  \n<small style='color:#667085'>{dates}"
                           + (f" · {x['duration_months']} months" if x.get("duration_months") else "") + "</small>"
                           if dates else ""), unsafe_allow_html=True)
            used = [sk.get("canonical_name") or sk["mention"] for sk in x.get("skills", [])]
            if used:
                st.markdown(chips_html(used, "info"), unsafe_allow_html=True)
    with st.expander("Edit or remove roles"):
        edited["experiences"] = st.data_editor(df, use_container_width=True, hide_index=True,
                                               disabled=["months", "occupation", "skills used", "evidence"],
                                               key="ed_exp")
with tabs[2]:
    df = pd.DataFrame([{"keep": True, "degree": x.get("degree"), "level": x.get("level"), "field": x.get("field"),
                        "institution": x.get("institution"), "year": x.get("year"), "evidence": ev_text(x)}
                       for x in cand["education"]])
    if df.empty:
        st.caption("No education entries found.")
    edited["education"] = st.data_editor(df, use_container_width=True, hide_index=True, disabled=["evidence"], key="ed_edu")
with tabs[3]:
    df = pd.DataFrame([{"keep": True, "name": x["name"], "evidence": ev_text(x)} for x in cand["credentials"]])
    if df.empty:
        st.caption("No credentials found.")
    edited["credentials"] = st.data_editor(df, use_container_width=True, hide_index=True, disabled=["evidence"], key="ed_cred")
with tabs[4]:
    df = pd.DataFrame([{"keep": True, "name": x["name"], "description": x.get("description"),
                        "skills": ", ".join(sk.get("canonical_name") or sk["mention"] for sk in x.get("skills", []))}
                       for x in cand["projects"]])
    if df.empty:
        st.caption("No projects section found.")
    for x in cand["projects"]:
        with st.container(border=True):
            st.markdown(f"**{x['name']}**")
            if x.get("description"):
                st.write(x["description"])
            used = [sk.get("canonical_name") or sk["mention"] for sk in x.get("skills", [])]
            if used:
                st.markdown(chips_html(used, "info"), unsafe_allow_html=True)
    with st.expander("Edit or remove projects"):
        edited["projects"] = st.data_editor(df, use_container_width=True, hide_index=True, disabled=["skills"],
                                            key="ed_proj")
with tabs[5]:
    df = pd.DataFrame([{"keep": True, "mention": x["mention"], "O*NET code": x.get("code"), "title": x.get("canonical_title"),
                        "method": x.get("method"), "confidence": x.get("confidence")} for x in cand["occupations"]])
    if df.empty:
        st.caption("No occupation mapped confidently.")
    edited["occupations"] = st.data_editor(df, use_container_width=True, hide_index=True,
                                           disabled=["mention", "O*NET code", "title", "method", "confidence"], key="ed_occ")
with tabs[6]:
    st.text_area("Text extracted from your file", cand["raw_text"], height=420, disabled=True)

c1, c3, _, c2 = st.columns([2, 2, 3, 1.4])
if c1.button("Save corrected profile", type="primary"):
    prof = copy.deepcopy(cand)
    sk_rows = edited["skills"].to_dict("records")
    kept, by_mention = [], {x["mention"]: x for x in cand["skills"]}
    for r in sk_rows:
        if not r.get("keep") or not isinstance(r.get("mention"), str) or not r["mention"].strip():
            continue
        orig = by_mention.get(r["mention"])
        kept.append(orig if orig else {"mention": r["mention"].strip(), "method": "user", "confidence": 1.0, "evidence": []})
    prof["skills"] = kept

    def keep_rows(key, items, fields):
        rows = edited[key].to_dict("records") if len(edited[key]) else []
        out = []
        for item, r in zip(items, rows):
            if r.get("keep"):
                for src, dst in fields.items():
                    item[dst] = r.get(src) if r.get(src) not in ("", None) else None
                out.append(item)
        return out

    prof["experiences"] = keep_rows("experiences", prof["experiences"],
                                    {"title": "title", "company": "company", "start": "start_date", "end": "end_date"})
    prof["education"] = keep_rows("education", prof["education"],
                                  {"degree": "degree", "field": "field", "institution": "institution", "year": "year"})
    prof["credentials"] = keep_rows("credentials", prof["credentials"], {"name": "name"})
    prof["projects"] = keep_rows("projects", prof["projects"], {"name": "name", "description": "description"})
    prof["occupations"] = keep_rows("occupations", prof["occupations"], {})
    res = call(client().save_profile, cid, prof, spinner="Saving and updating the knowledge graph…")
    if res:
        st.success("Profile saved. Recommendations will use the corrected profile.")
        st.rerun()

with c2.popover("Delete profile", type="tertiary"):
    st.write("Deletes the uploaded file, the parsed profile, recommendations, chat history and graph nodes. "
             "This cannot be undone.")
    if st.button("Delete permanently", type="primary"):
        res = call(client().delete_resume, cid)
        if res:
            st.session_state.pop("candidate_id", None)
            st.success(f"Deleted ({res.get('files_removed', 0)} file, {res.get('graph_nodes_removed', 0)} graph nodes).")
            st.rerun()

if a.get("resume"):  # only uploaded resumes keep the original file
    with c3.popover("Re-parse with current parser"):
        st.write("Parses the stored file again with the current extractor — useful after a parser update. "
                 "The structured profile is replaced; the file itself is unchanged.")
        force = False
        if cand.get("user_corrected"):
            st.warning("You saved corrections to this profile. Re-parsing replaces them with the new machine parse.")
            force = st.checkbox("Discard my corrections and re-parse")
        if st.button("Re-parse now", disabled=bool(cand.get("user_corrected")) and not force):
            res = call(client().reparse_resume, cid, mode, force, spinner="Re-parsing…")
            if res:
                b, n = res["before"], res["after"]
                st.session_state["reparse_note"] = (
                    f"Re-parsed with `{res['extractor']}` — roles {b['experiences']} → {n['experiences']}, "
                    f"education {b['education']} → {n['education']}, credentials {b['credentials']} → "
                    f"{n['credentials']}, projects {b['projects']} → {n['projects']}, skills {b['skills']} → {n['skills']}. "
                    "Run Job Recommendations again to use the new profile.")
                st.rerun()
