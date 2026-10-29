import html

import pandas as pd
import streamlit as st

from frontend.components.ui import (SOURCE_LABEL, WORK_MODE_ICON, attribution, badge_html, call, chips_html, client,
                                    require_candidate, salary_text, score_tone)

st.title("Job Recommendations")
cid = require_candidate()
if not cid:
    st.stop()

METHODS = {"final": "Final pipeline (RRF → ranker → cross-encoder → MMR)", "ltr": "RRF + LambdaMART / fallback ranker",
           "rrf": "RRF (lexical + dense fusion)", "bm25": "BM25", "tfidf": "TF-IDF cosine", "jaccard": "Jaccard skill overlap",
           "dense": "Dense (Sentence-Transformers + FAISS)"}

analysis = call(client().analysis, cid) or {"candidate": {"skills": []}}
cand_skills = {s["canonical_id"]: s.get("canonical_name") or s["mention"] for s in analysis["candidate"]["skills"]
               if s.get("canonical_id")}

with st.expander("Filters", expanded=False):
    f1, f2, f3 = st.columns(3)
    location = f1.text_input("Location contains", placeholder="e.g. TX, New York, Remote", key="f_location")
    work_modes = f2.multiselect("Work mode", ["remote", "hybrid", "onsite"], key="f_work_modes")
    seniorities = f3.multiselect("Seniority", ["internship", "entry", "associate", "mid_senior", "director", "executive"],
                                 key="f_seniorities")
    f4, f5, f6 = st.columns(3)
    min_salary = f4.number_input("Minimum annual salary (USD, jobs with known salary only)", min_value=0, step=5000,
                                 value=0, key="f_min_salary")
    max_years = f5.number_input("Max required years of experience", min_value=0, max_value=40, value=0,
                                help="0 = no limit. Jobs that state no requirement are kept.", key="f_max_years")
    title = f6.text_input("Title contains", key="f_title")
    f7, f8 = st.columns(2)
    sources = f7.multiselect("Source", list(SOURCE_LABEL), format_func=SOURCE_LABEL.get, key="f_sources")
    skills = f8.multiselect("Must require these of my skills", [s for s in cand_skills], format_func=cand_skills.get,
                            key="f_skills")
filters = {"location": location or None, "work_modes": work_modes, "seniorities": seniorities,
           "min_salary": float(min_salary) or None, "title_contains": title or None, "sources": sources,
           "required_skill_ids": skills, "max_required_years": float(max_years) if max_years else None}



def drop_filter(key: str, value=None) -> None:
    """Chip callback: clear one filter (or one value of a multi-select) and re-run the recommendations."""
    cur = st.session_state.get(key)
    if isinstance(cur, list):
        st.session_state[key] = [v for v in cur if v != value]
    else:
        st.session_state[key] = "" if isinstance(cur, str) else 0
    st.session_state["rerun_rec"] = True


# active filters as chips; clicking one removes it
active = ([("f_location", None, f"Location: {location}")] if location else []) \
    + [("f_work_modes", v, WORK_MODE_ICON.get(v, v)) for v in work_modes] \
    + [("f_seniorities", v, v.replace("_", "-")) for v in seniorities] \
    + ([("f_min_salary", None, f"Salary ≥ ${min_salary:,.0f}")] if min_salary else []) \
    + ([("f_max_years", None, f"≤ {max_years:g} years required")] if max_years else []) \
    + ([("f_title", None, f"Title: {title}")] if title else []) \
    + [("f_sources", v, SOURCE_LABEL.get(v, v)) for v in sources] \
    + [("f_skills", v, f"Requires {cand_skills.get(v, v)}") for v in skills]
if active:
    with st.container(horizontal=True, key="filter-chips"):
        for i, (key, value, label) in enumerate(active):
            st.button(f"{label}  ✕", key=f"chip-{i}-{key}", on_click=drop_filter, args=(key, value),
                      help="Remove this filter")

tab_rec, tab_cmp = st.tabs(["Recommendations", "Compare ranking methods"])
with tab_rec:
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    method = c1.selectbox("Ranking method", list(METHODS), format_func=METHODS.get)
    k = c2.number_input("Results", 3, 30, 10)
    llm_exp = c3.toggle("LLM explanations", value=False, help="Citation-constrained LLM explanation when a model is "
                                                               "available; always verified. Otherwise template.")
    use_ce = c4.toggle("Cross-encoder", value=True)
    if st.button("Get recommendations", type="primary") or st.session_state.pop("rerun_rec", False):
        st.session_state["rec"] = call(client().recommend, cid, filters, method, int(k), llm_exp, use_ce,
                                       spinner="Retrieving, fusing, reranking…")
    rec = st.session_state.get("rec")
    if rec is None or rec.get("candidate_id") != cid:
        stored = None
        try:
            stored = client().results(cid)
        except Exception:
            pass
        rec = stored
    if not rec:
        st.info("Press **Get recommendations** to run the pipeline.")
    else:
        with st.expander("How this was ranked"):
            st.caption(f"Ranker: **{rec['ranker']}** · jobs left after filters: {rec.get('filtered_pool', '—')}")
            if rec.get("stages"):
                st.dataframe(pd.DataFrame([rec["stages"]], index=["jobs kept at each stage"]), use_container_width=True)
            if "fallback" in (rec.get("ranker") or ""):
                st.caption("LambdaMART is awaiting human relevance judgments; a documented weighted-feature fallback "
                           "ranker is used.")
        if not rec["results"]:
            st.warning("No jobs matched these filters.")
        for r in rec["results"]:
            j = r["job"]
            with st.container(border=True):
                meta = " · ".join(html.escape(str(x)) for x in [
                    j.get("company"), j.get("location"), WORK_MODE_ICON.get(j.get("work_mode")),
                    (j.get("employment_type") or "").replace("_", " ") or None,
                    (j.get("seniority") or "").replace("_", "-") or None, salary_text(j),
                    f"posted {j['posting_date']}" if j.get("posting_date") else None] if x)
                cov = r.get("coverage_required")
                # one tight block: title + score badges on a line, then the facts, then skills as chips
                st.markdown(
                    f"<div class='om-job'><div class='om-job-head'><span class='om-job-title'>{r['rank']}. "
                    f"{html.escape(j['title'] or '')}</span><span>"
                    f"{badge_html(f'{r['score']:.2f} match', score_tone(r['score']), '0.88rem')}"
                    f"{badge_html(f'{cov:.0%} of required skills', score_tone(cov), '0.88rem') if cov is not None else ''}"
                    f"</span></div><div class='om-job-meta'>{meta}</div>"
                    f"<div class='om-job-skills'><span class='om-job-label'>Matched</span><span>"
                    + (chips_html([m["name"] for m in r["matched_skills"]], "good", limit=8)
                       or "<span class='om-job-none'>no explicit skill matches</span>")
                    + "</span></div><div class='om-job-skills'><span class='om-job-label'>Missing</span><span>"
                    + (chips_html([m["name"] for m in r["missing_required"]]
                                  + [m["name"] + " (nice to have)" for m in r["missing_preferred"]], "neutral", limit=8)
                       or "<span class='om-job-none'>nothing required is missing</span>")
                    + "</span></div></div>", unsafe_allow_html=True)
                exp = r.get("explanation")
                with st.expander("Why this job?"):
                    if exp:
                        st.markdown(exp["text"].replace("\n", "  \n") or "_No verified statements._")
                        st.markdown("**Evidence**")
                        rows = [{"cite": i["cite_id"], "from": i["kind"], "skills": ", ".join(i["skill_names"]),
                                 "text": i["text"]} for i in exp["registry"]["items"]]
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                        rej = exp["verification"].get("rejected") or []
                        if rej:
                            st.warning(f"{len(rej)} unsupported sentence(s) were removed by the verifier.")
                        st.caption(f"Explanation method: {exp['method']} · faithfulness "
                                   f"{exp['verification'].get('faithfulness')}")
                    st.markdown("**How the score was computed**")
                    cs, cr = r.get("component_scores") or {}, r.get("component_ranks") or {}
                    st.dataframe(pd.DataFrame({"score": cs, "rank in that method": cr}), use_container_width=True)
                    if r.get("features"):
                        st.caption("Ranker features")
                        st.dataframe(pd.Series(r["features"], name="value").to_frame(), use_container_width=True)
                b1, b2, b3 = st.columns([2, 1, 1])
                with b1:
                    attribution(j)
                if b2.button("Skill gap", key=f"gap-{j['job_id']}"):
                    st.session_state["selected_job"] = j["job_id"]
                    st.switch_page("pages/skill_gap.py")
                if b3.button("Outreach draft", key=f"out-{j['job_id']}"):
                    st.session_state["selected_job"] = j["job_id"]
                    st.session_state["pending_outreach"] = j["job_id"]
                    st.switch_page("pages/career_assistant.py")

with tab_cmp:
    st.caption("Runs every method with the same filters. These rankings are not judged for relevance here — use the "
               "Annotation page to create judgments; the Research Dashboard reports nDCG/MRR once they exist.")
    kc = st.slider("Top-k per method", 5, 20, 10)
    if st.button("Compare all methods"):
        out = {}
        prog = st.progress(0.0)
        for i, m in enumerate(METHODS):
            res = call(client().recommend, cid, filters, m, kc, False, m == "final")
            if res:
                out[m] = res
            prog.progress((i + 1) / len(METHODS), text=f"{m} done")
        st.session_state["cmp"] = out
    cmp = st.session_state.get("cmp")
    if cmp:
        table = {m: [f"{r['job']['title'][:38]} ({r['job'].get('company') or '?'})" for r in res["results"]]
                 for m, res in cmp.items()}
        n = max(len(v) for v in table.values())
        st.dataframe(pd.DataFrame({m: v + [""] * (n - len(v)) for m, v in table.items()}, index=range(1, n + 1)),
                     use_container_width=True)
        cov = {m: [r["coverage_required"] for r in res["results"] if r.get("coverage_required") is not None]
               for m, res in cmp.items()}
        st.bar_chart(pd.Series({m: (sum(v) / len(v) if v else 0) for m, v in cov.items()},
                               name="mean required-skill coverage of top-k (descriptive)"))
