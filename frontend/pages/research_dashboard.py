import html

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components.ui import badge_html, call, client, not_evaluated, status_badge_html

st.title("Research Dashboard")
st.caption("Every number below is read from Data/results/*.json written by `python scripts/evaluate.py`. "
           "Metrics that need missing labels or providers are shown as **Not evaluated** — never estimated. "
           "Descriptive statistics are labelled as such and are not quality measures.")

data = call(client().evaluation_results, spinner="Loading results…")
if not data:
    st.stop()
exps, res = data["experiments"], data["results"]

with st.expander("Run evaluations", expanded=not res):
    sel = st.multiselect("Experiments", ["classifiers", "A", "B", "C", "D", "E", "F", "G"],
                         default=["classifiers", "B", "C", "D", "E", "F", "G"])
    if st.button("Run selected (background)"):
        r = call(client()._req, "POST", "/api/admin/evaluation/run", json={"only": sel})
        if r:
            st.session_state["eval_job"] = r["job"]
    job = st.session_state.get("eval_job")
    if job:
        j = call(client().admin_job, job)
        if j:
            st.write(f"Job `{job}`: **{j['status']}**")
            if j["status"] == "running":
                st.button("Refresh")
            st.code("\n".join(j.get("log", []))[-3000:] or "…")

# one row per experiment: a coloured status and the full reason it is (in)complete — never cut off
rows = []
for k, e in exps.items():
    r = res.get(k) or {}
    rows.append(
        f"<tr><td><b>{html.escape(k.split('_')[0])}</b></td>"
        f"<td>{html.escape(e['title'])}<br><small>{html.escape(str(e['rq']))}"
        f"{' · run ' + html.escape(str(r['generated_at'])[:16].replace('T', ' ')) if r.get('generated_at') else ''}"
        f"</small></td>"
        f"<td>{status_badge_html(r.get('status'))}{badge_html('silver results', 'warn') if r.get('silver') else ''}</td>"
        f"<td class='om-note'>{html.escape(r.get('reason') or '')}</td></tr>")
st.markdown("<table class='om-table'><thead><tr><th></th><th>Experiment</th><th>Status</th><th>What is missing</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>", unsafe_allow_html=True)

ANNOTATION_LABEL = {"extraction_documents_double_annotated": "Extraction — documents labelled by two annotators",
                    "normalization_mentions": "Normalisation — skill mentions",
                    "graph_edges": "Graph — edges validated", "relevance_candidate_profiles": "Relevance — profiles judged",
                    "relevance_pooled_jobs_per_profile": "Relevance — jobs pooled per profile",
                    "relevance_judgments": "Relevance — judgments", "explanations": "Explanations — rated",
                    "user_study_participants": "User study — participants"}
ann = call(client().annotations_status)
if ann:
    st.subheader("Human annotation progress")
    cols = st.columns(2)
    for i, (k, target) in enumerate(ann["targets"].items()):
        if k not in ann["progress"]:
            continue                      # a design parameter (e.g. jobs pooled per profile), not something to count
        have = ann["progress"].get(k, 0)
        cols[i % 2].progress(min(1.0, have / target), text=f"{ANNOTATION_LABEL.get(k, k.replace('_', ' '))}: "
                                                           f"{have:,} / {target:,}")
    st.caption("Targets from the proposal. Counts are read from Data/annotations/*; nothing is pre-filled. "
               "The human-label metrics below stay “Not evaluated” until these are filled.")


def metric_or_ne(col, label, value, fmt="{:.3f}"):
    if value is None:
        with col:
            st.markdown(f"**{label}**")
            not_evaluated()
    else:
        col.metric(label, fmt.format(value) if isinstance(value, (int, float)) else value)


SILVER_NOTE = ("**Silver** = scored against resume_corpus expert occupation labels (Jiechieu & Tsopze, 2020), "
               "not against this project's human annotations. Reported separately; never used to train LambdaMART.")


def silver_header(title: str) -> None:
    st.markdown(f"**{title}** <span style='background:#fff4e5;color:#9a5b00;padding:1px 8px;border-radius:10px;"
                f"font-size:0.78rem'>silver</span>", unsafe_allow_html=True)


def ranking_table(metrics: dict) -> pd.DataFrame:
    labels = {"ltr": "LambdaMART / fallback", "final": "Final (+cross-encoder, MMR)"}
    return pd.DataFrame([{"method": labels.get(m, m), "nDCG@10": v.get("ndcg@10"), "MRR": v.get("mrr"),
                          "Precision@5": v.get("precision@5"), "queries": v.get("queries")}
                         for m, v in metrics.items()])


tabs = st.tabs(["Extraction (A)", "Normalisation (B)", "Graph (C)", "Ranking (D)", "QLoRA (E)", "Explanations (F)",
                "Providers (G)", "Baselines"])

with tabs[0]:
    r = res.get("A_extraction")
    for name in ("classical", "llm"):
        st.markdown(f"**{ {'classical': 'Classical', 'llm': 'LLM'}[name] } extraction (micro, exact spans)**")
        m = ((r or {}).get("metrics", {}).get(name) or {}).get("exact", {}).get("micro")
        c = st.columns(3)
        for col, key in zip(c, ("precision", "recall", "f1")):
            metric_or_ne(col, key.title(), (m or {}).get(key))
    if r and r.get("metrics", {}).get("classical"):
        per = pd.DataFrame(r["metrics"]["classical"]["overlap"]).T
        st.markdown("Per-label (relaxed overlap, classical)")
        st.dataframe(per, use_container_width=True)
    occ = ((r or {}).get("silver") or {}).get("occupation_extraction")
    if occ:
        st.divider()
        silver_header("Occupation extraction vs expert occupation labels")
        rows = []
        for key, label in (("classical", "Classical"), ("classical_same_docs", "Classical (same documents as LLM)"),
                           ("llm", f"LLM ({(occ.get('llm') or {}).get('model', '')})")):
            v = occ.get(key)
            if v:
                rows.append({"extractor": label, "resumes": v["resumes"], "precision": v["precision"],
                             "recall": v["recall"], "F1": v["f1"], "top-1 accuracy": v["top1_accuracy"],
                             "ms / resume": v["latency_ms_per_doc"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if not occ.get("llm"):
            st.caption("LLM row not evaluated: no local model was running.")
        st.caption(occ.get("labels", "") + ". " + SILVER_NOTE)
    if r and r.get("reason"):
        st.caption(r["reason"])

with tabs[1]:
    r = res.get("B_normalization") or {}
    m = r.get("metrics", {})
    st.markdown("**Skill normalisation — top-1 accuracy vs human gold**")
    c = st.columns(3)
    for col, s in zip(c, ("classical", "embedding", "hybrid")):
        metric_or_ne(col, s.title(), ((m.get("skill_gold") or {}).get("results", {}).get(s) or {}).get("top1_accuracy"))
    st.markdown("**Graph edge precision (human validation)**")
    metric_or_ne(st.container(), "Edge precision", (m.get("edge_validation") or {}).get("edge_precision"))
    occ = m.get("occupation")
    st.markdown("**Occupation normalisation — O*NET held-out alternate titles (real reference labels)**")
    if occ:
        df = pd.DataFrame(occ["results"]).T
        st.dataframe(df, use_container_width=True)
        st.caption(f"{occ['test_items']} titles · {occ['gold']}. {occ['note']}")
    else:
        not_evaluated()
    diag = (r.get("descriptive") or {}).get("skill_synthetic_diagnostic")
    if diag:
        with st.expander("Synthetic robustness diagnostic (NOT human gold)"):
            st.dataframe(pd.DataFrame(diag["results"]).T, use_container_width=True)
            st.caption(diag["construction"])

with tabs[2]:
    r = res.get("C_graph")
    if not r:
        not_evaluated()
    else:
        df = pd.DataFrame(r["metrics"]).T
        cols = ["nodes", "edges", "duplicate_node_rate", "entity_fragmentation", "distinct_predicates",
                "predicate_fragmentation", "schema_conformance", "grounded_edge_rate", "latency_ms_per_doc"]
        st.dataframe(df[[c for c in cols if c in df]], use_container_width=True)
        long = df[["duplicate_node_rate", "predicate_fragmentation", "schema_conformance", "grounded_edge_rate"]] \
            .reset_index().melt(id_vars="index")
        st.plotly_chart(px.bar(long, x="variable", y="value", color="index", barmode="group").update_layout(height=320),
                        use_container_width=True)
        st.markdown("**Ranking impact (human judgments)**")
        not_evaluated("Not evaluated — needs relevance judgments")
        imp = (r.get("silver") or {}).get("ranking_impact")
        if imp:
            silver_header("Ranking impact of graph features")
            st.dataframe(ranking_table(imp), hide_index=True, use_container_width=True)
            st.caption(SILVER_NOTE)
        st.caption(r.get("reason") or "")

with tabs[3]:
    r = res.get("D_ranking") or {}
    methods = ["bm25", "tfidf", "jaccard", "dense", "rrf", "ltr", "final", "final_no_cross_encoder", "final_no_mmr",
               "final_no_graph_features"]
    labels = {"ltr": "LambdaMART / fallback", "final": "Final (+cross-encoder, MMR)"}
    table = []
    for mth in methods:
        mm = (r.get("metrics") or {}).get(mth) or {}
        table.append({"method": labels.get(mth, mth), "nDCG@10": mm.get("ndcg@10", "Not evaluated"),
                      "MRR": mm.get("mrr", "Not evaluated"), "Precision@5": mm.get("precision@5", "Not evaluated")})
    st.markdown("**Human relevance judgments**")
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    sv = r.get("silver") or {}
    if sv.get("metrics"):
        silver_header("All methods vs silver relevance")
        st.dataframe(ranking_table(sv["metrics"]), hide_index=True, use_container_width=True)
        d = sv.get("details") or {}
        st.caption(f"{d.get('queries')} resumes × {d.get('jobs_judged_per_query')} jobs, every job labelled "
                   f"(mean {d.get('relevant_jobs_per_query_mean')} relevant per resume). {sv.get('labels', '')}. "
                   "Lexical methods also read job titles, so silver scores favour them. " + SILVER_NOTE)
    lm = (r.get("details") or {}).get("lambdamart") or {}
    if lm:
        st.caption(f"LambdaMART: **{lm.get('state')}** — {lm.get('judgments', 0)} / {lm.get('required', {}).get('judgments')} "
                   f"judgments, {lm.get('candidates', 0)} / {lm.get('required', {}).get('candidates')} candidates.")
    if r.get("reason"):
        st.caption(r["reason"])
    desc = r.get("descriptive") or {}
    if desc:
        with st.expander("Descriptive statistics (not relevance metrics)"):
            st.markdown("Mean latency per query (ms)")
            st.bar_chart(pd.Series(desc.get("latency_ms_mean", {})))
            st.markdown("Mean required-skill coverage of top-10")
            st.bar_chart(pd.Series(desc.get("mean_required_skill_coverage@10", {})))
            ov = desc.get("top10_overlap", {})
            if ov:
                ms = sorted({x for k in ov for x in k.split("~")})
                mat = pd.DataFrame(1.0, index=ms, columns=ms)
                for k, v in ov.items():
                    a, b = k.split("~")
                    mat.loc[a, b] = mat.loc[b, a] = v
                st.plotly_chart(px.imshow(mat, text_auto=".2f", zmin=0, zmax=1, color_continuous_scale="Blues")
                                .update_layout(height=420), use_container_width=True)
                st.caption("Overlap@10 between methods' top-10 lists.")

with tabs[4]:
    r = res.get("E_qlora")
    st.markdown("**Status:** " + status_badge_html((r or {}).get("status")), unsafe_allow_html=True)
    if r:
        st.caption(r.get("reason") or "")
        st.json((r.get("details") or {}).get("environment", {}), expanded=False)
    c = st.columns(2)
    metric_or_ne(c[0], "QLoRA extractor F1", None)
    metric_or_ne(c[1], "Zero-shot LLM F1", None)

with tabs[5]:
    r = res.get("F_explanations") or {}
    m = r.get("metrics", {})
    for name, key in (("Template (automatic verifier)", "template_automatic"), ("LLM (automatic verifier)", "llm_automatic")):
        st.markdown(f"**{name}**")
        c = st.columns(3)
        for col, k in zip(c, ("faithfulness", "hallucination_rate", "citation_validity")):
            metric_or_ne(col, k.replace("_", " ").title(), (m.get(key) or {}).get(k))
    st.markdown("**Human faithfulness labels**")
    if m.get("human"):
        st.dataframe(pd.DataFrame(m["human"]).T, use_container_width=True)
    else:
        not_evaluated()
    if r.get("reason"):
        st.caption(r["reason"])

with tabs[6]:
    r = res.get("G_providers") or {}
    if r.get("metrics"):
        st.dataframe(pd.DataFrame(r["metrics"]).T, use_container_width=True)
    else:
        not_evaluated()
    if r.get("reason"):
        st.caption(r["reason"])
    for s in (r.get("details") or {}).get("skipped", []):
        st.caption(f"Skipped {s['provider']}/{s['model']}: {s['reason']}")
    mm = call(client().models_metrics)
    st.markdown("**Logged LLM calls (all tasks)**")
    if mm and mm["summary"]:
        st.dataframe(pd.DataFrame(mm["summary"]), hide_index=True, use_container_width=True)
    else:
        st.caption("No LLM calls logged yet.")
    ms = call(client().models_status)
    if ms:
        st.markdown("**Routing status**")
        st.dataframe(pd.DataFrame(ms["llm"]["tasks"]).T, use_container_width=True)

with tabs[7]:
    r = res.get("classifiers")
    if not r:
        not_evaluated()
    else:
        occ, cat = r["metrics"]["occupation_family"], r["metrics"]["resume_category"]
        c = st.columns(4)
        c[0].metric("SOC family accuracy", f"{occ['accuracy']:.3f}")
        c[1].metric("SOC family macro-F1", f"{occ['macro_f1']:.3f}")
        c[2].metric("Resume category accuracy", f"{cat['accuracy']:.3f}")
        c[3].metric("Resume category macro-F1", f"{cat['macro_f1']:.3f}")
        st.caption(f"{occ['test_set']} · {cat['cv']} over {cat['documents']} resumes.")
        st.bar_chart(pd.Series(cat["per_class_accuracy"]).sort_values())

