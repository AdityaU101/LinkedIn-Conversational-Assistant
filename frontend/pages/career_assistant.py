import time

import pandas as pd
import streamlit as st

from frontend.components.ui import call, client, require_candidate

st.title("Career Assistant")
cid = require_candidate()
if not cid:
    st.stop()

st.caption("Answers are composed from your stored profile, the job postings and O*NET, with citations. A critic "
           "removes any sentence that is not supported by the cited evidence. Nothing is sent or scheduled "
           "externally without your approval.")

EXAMPLES = ["Which jobs best match my resume?", "Why does job 1 match me?", "What skills am I missing for job 1?",
            "What transferable skills do I have for job 1?", "What should I improve in my resume?",
            "Which related occupations should I consider?", "Draft an outreach message for job 1",
            "Remind me to follow up in 7 days"]

chat_tab, opt_tab, out_tab, fu_tab = st.tabs(["Chat", "Resume optimizer", "Outreach drafts", "Follow-ups"])


def plain(text: str) -> str:
    """Answers are plain sentences, not Markdown: keep a literal '*' ('O*NET') from being read as emphasis."""
    return text.replace("*", "\\*")


def typed(text: str):
    """Reveal an answer word by word. The answer arrives complete — it is verified as a whole before any of it is
    shown — so this is how it is displayed, not how it is generated."""
    for line in text.split("\n"):
        for word in line.split(" "):
            yield word + " "
            time.sleep(0.012)
        yield "  \n"


def render_answer(res: dict, stream: bool = False, draft: bool = False) -> None:
    """``stream``: reveal a new answer progressively. ``draft``: an outreach message, shown with a copy button."""
    if draft or res.get("intent") == "outreach":
        st.caption("Draft only — review and edit before sending it yourself. Nothing has been sent. "
                   "Use the copy icon at the top right of the message.")
        st.code(res.get("body") or res["answer"], language=None, wrap_lines=True)
    elif stream and res["answer"]:
        st.write_stream(typed(plain(res["answer"])))
    else:
        st.markdown(plain(res["answer"]).replace("\n", "  \n") or "_No verified statements to show._")
    if res.get("requires_approval") and res.get("intent") != "outreach" and not draft:
        st.info("This is a pending action. Review and approve it yourself — nothing has been done yet.")
    with st.expander(f"Evidence & verification ({res.get('intent')})"):
        crit = res.get("critic") or {}
        st.caption(f"Factual sentences: {crit.get('n_factual')} · supported: {crit.get('n_supported')} · "
                   f"citation validity: {crit.get('citation_validity')}")
        if res.get("citations"):
            st.dataframe(pd.DataFrame([{"cite": c["cite_id"], "from": c["kind"], "text": c["text"]}
                                       for c in res["citations"]]), hide_index=True, use_container_width=True)
        for r in crit.get("rejected") or []:
            st.warning(f"Removed: “{r['sentence']}” — {'; '.join(r['reasons'])}")
        if res.get("injection_flags"):
            st.warning("Instruction-like text was detected and treated as data: " + "; ".join(res["injection_flags"]))


with chat_tab:
    # this sitting's conversation only — frontend/app.py clears it when the active profile changes
    chat = st.session_state.setdefault("chat", [])
    for m in chat:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                render_answer(m["res"])
            else:
                st.markdown(m["content"])
    pick, suggested = None, not chat
    if suggested:           # suggestions only for an empty conversation; afterwards the chat has the room
        st.caption("Try one of these, or type your own question below.")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
                pick = ex
    msg = st.chat_input("Ask about your matches, gaps, resume, or outreach…") or pick
    if msg:
        with st.chat_message("user"):
            st.markdown(msg)
        chat.append({"role": "user", "content": msg})
        with st.chat_message("assistant"):
            res = call(client().chat, cid, msg, None, spinner="Thinking with your data…")
            if res:
                render_answer(res, stream=True)
                chat.append({"role": "assistant", "res": res})
        if suggested:
            st.rerun()      # the suggestions were drawn before this message existed; redraw without them

with opt_tab:
    if st.button("Analyse my resume", type="primary"):
        st.session_state["opt"] = call(client().optimize, cid, spinner="Comparing your resume with your top matches…")
    opt = st.session_state.get("opt")
    if opt:
        for s in opt.get("suggestions", []):
            icon = {"missing_skill": "🎯", "add_context": "🧷", "quantify": "📏", "structure": "🧱"}.get(s["type"], "•")
            st.markdown(f"{icon} {s['advice']}")
        with st.expander("Verified summary & evidence"):
            render_answer(opt)
    st.divider()
    st.markdown("**Rephrase one existing bullet** (local model only; rejected if it adds skills or numbers)")
    bullet = st.text_area("Bullet from your resume", max_chars=1500)
    if st.button("Suggest rewrite") and bullet.strip():
        res = call(client().optimize, cid, bullet, spinner="Rewriting…")
        rw = (res or {}).get("rewrite") or {}
        if not rw.get("available"):
            st.info(rw.get("reason", "No local model available."))
        elif rw.get("accepted"):
            st.success(rw["rewrite"])
        else:
            st.error(f"Rewrite rejected — it introduced skills {rw['introduced_skills']} or numbers "
                     f"{rw['introduced_numbers']} not in your original.")

with out_tab:
    jobs = {}
    try:
        jobs = {r["job"]["job_id"]: f"#{r['rank']} {r['job']['title']} — {r['job'].get('company') or '?'}"
                for r in client().results(cid)["results"]}
    except Exception:
        pass
    pending = st.session_state.pop("pending_outreach", None)
    if pending and pending not in jobs:
        jobs = {pending: pending, **jobs}
    if not jobs:
        st.caption("Get recommendations first.")
    else:
        ids = list(jobs)
        jid = st.selectbox("Job", ids, index=ids.index(pending) if pending in ids else 0, format_func=jobs.get)
        if st.button("Draft outreach", type="primary") or pending:
            res = call(client().outreach, cid, jid, spinner="Drafting from verified evidence…")
            if res:
                st.session_state["outreach_res"] = {"cid": cid, "jid": jid, "res": res}
        saved = st.session_state.get("outreach_res")
        if saved and saved["cid"] == cid and saved["jid"] == jid:
            res = saved["res"]
            st.caption("Draft only — nothing has been sent. Edit it here, then copy your version below.")
            text = st.text_area("Draft (edit before you send it yourself)", res.get("body") or res["draft"], height=220,
                                key=f"draft-{jid}")
            st.caption("Your version — use the copy icon at the top right:")
            st.code(text, language=None, wrap_lines=True)
            info = (res.get("outreach") or {}).get("company_research")
            if info:
                with st.expander("Company facts (from stored LinkedIn company data)"):
                    st.json(info, expanded=False)
            with st.expander("Evidence & verification"):
                crit = res.get("critic") or {}
                st.caption(f"Factual sentences: {crit.get('n_factual')} · supported: {crit.get('n_supported')}")
                if res.get("citations"):
                    st.dataframe(pd.DataFrame([{"from": c["kind"], "text": c["text"]} for c in res["citations"]]),
                                 hide_index=True, use_container_width=True)

with fu_tab:
    f1, f2 = st.columns([2, 3])
    when = f1.text_input("When", "in 7 days", help="e.g. 'in 5 days', 'next week', '2026-11-01'")
    note = f2.text_input("Note", "Follow up on application")
    if st.button("Create reminder"):
        res = call(client().followup, cid, when, st.session_state.get("selected_job"), note)
        if res:
            st.success(res["answer"])
    fus = (call(client().followups, cid) or {"items": []})["items"]
    for f in fus:
        c1, c2, c3, c4 = st.columns([2, 4, 2, 2])
        c1.write(f["due_date"])
        c2.write(f"{f['note']} {('· ' + f['job_id']) if f.get('job_id') else ''}")
        c3.write(f"`{f['status']}`")
        if f["status"] == "pending_approval" and c4.button("Approve", key=f"ap{f['followup_id']}"):
            call(client().set_followup, f["followup_id"], "approved")
            st.rerun()
        elif f["status"] == "approved" and c4.button("Done", key=f"dn{f['followup_id']}"):
            call(client().set_followup, f["followup_id"], "done")
            st.rerun()
