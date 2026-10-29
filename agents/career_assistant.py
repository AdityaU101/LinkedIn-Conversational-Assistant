"""Career-assistant answer composer.

Intent is classified with transparent keyword rules (no LLM needed). Each
answer is a list of sentences with citations into an evidence registry built
from stored data; the Critic then verifies them. User messages are treated
as questions only — they cannot change these rules or trigger actions
without approval.
"""
from __future__ import annotations

import re
from typing import Optional

from agents.evidence_tools import RegistryBuilder
from agents.state import AgentState
from common.schemas import Candidate
from matching.skill_gap import compute_gap, related_occupations
from matching.store import get_store

# Requests to act on systems — run queries/commands, delete data, send to an address. The assistant has no
# tools that do any of these; it says so instead of guessing an intent from a keyword like "email".
RUN_COMMAND = re.compile(r"\b(?:run|execute|eval)\b[^.?!]{0,40}\b(?:cypher|sql|query|queries|command|script|code|shell)\b",
                         re.I)
QUESTION = re.compile(r"^\s*(?:how|what|why|where|when|which|should|can i|do i|is it)\b", re.I)
UNSUPPORTED = re.compile(
    r"\b(?:detach\s+delete|drop\s+table|match\s*\(\w*\)\s*(?:detach\s+)?delete)\b"
    r"|\b(?:delete|wipe|erase|drop)\b[^.?!]{0,30}\b(?:database|graph|all (?:data|nodes|records|users))\b"
    r"|\b(?:send|e-?mail|forward|submit)\b[^.?!]{0,60}\b[\w.+-]+@[\w-]+\.[\w.]+", re.I)
UNSUPPORTED_REPLY = ("I can't run database queries or commands, change or delete stored data, or send anything to "
                     "anyone — I only answer from your profile and the job postings, and every external action needs "
                     "your own approval. I can draft an outreach message for you to review and send yourself.")

INTENTS = [
    ("unsupported_action", UNSUPPORTED),
    ("outreach", re.compile(r"\b(outreach|draft|message|email|cover letter|reach out|write to)\b", re.I)),
    ("followup", re.compile(r"\b(remind|reminder|follow[- ]?up|schedule)\b", re.I)),
    ("improve_resume", re.compile(r"\b(improve|optimi[sz]e|better|strengthen|fix)\b.*\b(resume|cv|profile)\b|\bresume tips\b", re.I)),
    ("transferable", re.compile(r"\btransferable\b", re.I)),
    ("related_occupations", re.compile(r"\b(related|other|alternative|adjacent)\b.*\b(occupations?|roles?|careers?|jobs? titles?)\b", re.I)),
    ("missing_skills", re.compile(r"\b(missing|lack|gap|gaps|need to learn|should i learn)\b", re.I)),
    ("why_match", re.compile(r"\bwhy\b.*\b(match|fit|recommend|good)\b|\bexplain\b", re.I)),
    ("recommend", re.compile(r"\b(which|what|best|top|recommend|show)\b.*\bjobs?\b|\bmatch(es)?\b", re.I)),
]

HELP = ("I can answer: which jobs best match your resume, why a job matches, which skills you are missing, "
        "your transferable skills, related occupations, how to improve your resume, draft an outreach message, "
        "or set a follow-up reminder.")


def classify(message: str) -> str:
    if RUN_COMMAND.search(message or "") and not QUESTION.match(message or ""):  # 'how do I run SQL…' is a question
        return "unsupported_action"
    for name, pat in INTENTS:
        if pat.search(message or ""):
            return name
    return "help"


def resolve_job(message: str, state: AgentState) -> Optional[str]:
    if state.get("job_id"):
        return state["job_id"]
    results = (state.get("recommendations") or {}).get("results", [])
    m = re.search(r"\b(?:job|#|number|no\.?)\s*(\d{1,2})\b|\b(first|second|third|top)\b", message or "", re.I)
    if results:
        idx = 0
        if m and m.group(1):
            idx = int(m.group(1)) - 1
        elif m and m.group(2):
            idx = {"first": 0, "top": 0, "second": 1, "third": 2}[m.group(2).lower()]
        if 0 <= idx < len(results):
            return results[idx]["job"]["job_id"]
    return None


def compose(state: AgentState) -> AgentState:
    trace = state.get("trace", []) + ["career_assistant"]
    intent = state.get("intent") or classify(state.get("message", ""))
    if not state.get("candidate"):
        return {**state, "intent": intent, "trace": trace,
                "sentences": [{"text": "Please upload a resume first so I can answer from your own data.",
                               "citations": []}]}
    cand = Candidate.model_validate(state["candidate"])
    store = get_store()
    rb = RegistryBuilder(cand.candidate_id)
    sents: list[dict] = []
    allowed: set[str] = set()
    mask: list[str] = []
    results = (state.get("recommendations") or {}).get("results", [])

    if intent == "recommend":
        for r in results[:5]:
            job = store.by_id(r["job"]["job_id"])
            mask += [job.title or "", job.company or ""]
            gap = compute_gap(cand, job)
            rb.add_gap(gap)
            names = [m.name for m in gap.matched_required][:3]
            cites = [c for m in gap.matched_required[:3] for c in rb.cites_for(m.skill_id, "candidate")
                     + rb.cites_for(m.skill_id, "job", job.job_id)]
            n, tot = len(gap.matched_required), len(gap.matched_required) + len(gap.missing_required)
            allowed |= {str(n), str(tot), str(r["rank"])}
            where = f" at {job.company}" if job.company else ""
            if names:
                sents.append({"text": f"#{r['rank']} {job.title}{where}: you match {n} of {tot} required skills, "
                                      f"including {', '.join(names)}.", "citations": cites})
            else:
                sents.append({"text": f"#{r['rank']} {job.title}{where} ranks highly on overall text similarity, "
                                      "but none of its required skills are explicitly evidenced in your resume.",
                              "citations": []})
        if not results:
            sents.append({"text": "There are no recommendations yet — run Job Recommendations first.", "citations": []})
    elif intent in ("why_match", "missing_skills", "transferable"):
        jid = resolve_job(state.get("message", ""), state)
        if not jid:
            sents.append({"text": "Tell me which job (e.g. 'job 2') or pick one on the recommendations page.",
                          "citations": []})
        else:
            job = store.by_id(jid)
            mask += [job.title or "", job.company or ""]
            gap = compute_gap(cand, job)
            rb.add_gap(gap)
            state = {**state, "job_id": jid}
            if intent == "why_match":
                for m in (gap.matched_required + gap.matched_preferred)[:6]:
                    c = rb.cites_for(m.skill_id, "candidate") + rb.cites_for(m.skill_id, "job", jid)
                    sents.append({"text": f"The posting asks for {m.name}{' (preferred)' if m.importance == 'preferred' else ''}"
                                          f", and your resume shows it.", "citations": c})
                if gap.coverage_required is not None:
                    n, tot = len(gap.matched_required), len(gap.matched_required) + len(gap.missing_required)
                    allowed |= {str(n), str(tot)}
                    sents.append({"text": f"You cover {n} of {tot} required skills for {job.title}.", "citations": []})
                if not gap.matched_required and not gap.matched_preferred:
                    sents.append({"text": "No required skill is explicitly evidenced in your resume; the ranking is "
                                          "driven by overall text similarity.", "citations": []})
            if intent in ("missing_skills", "why_match"):
                for m in gap.missing_required[:6]:
                    sents.append({"text": f"Missing (required): {m.name}.", "citations": rb.cites_for(m.skill_id, "job", jid)})
                for m in gap.missing_preferred[:4]:
                    sents.append({"text": f"Missing (preferred): {m.name}.", "citations": rb.cites_for(m.skill_id, "job", jid)})
                if intent == "missing_skills" and not gap.missing_required and not gap.missing_preferred:
                    sents.append({"text": f"Your resume evidences every skill the {job.title} posting lists.",
                                  "citations": []})
            if intent == "transferable":
                for m in gap.transferable[:6]:
                    b = m.bridges[0]
                    sents.append({"text": f"{b.candidate_skill_name} is related to the missing skill {m.name} "
                                          f"(inferred relation, not the same skill).",
                                  "citations": rb.cites_for(m.skill_id, "graph", jid)})
                from explanations.evidence_registry import EvidenceItem
                from normalization.skill_normalizer import get_skill_normalizer

                for t in gap.occupation_transferable[:4]:
                    text = (f"O*NET rates {t.name} as important for both your occupation and the target occupation.")
                    ids = [t.skill_id] + [m.canonical_id for m in get_skill_normalizer().extract(text, "x")]
                    gid = rb.add_item(EvidenceItem(
                        cite_id="", kind="graph", skill_ids=ids, skill_names=[t.name] * len(ids),
                        text=f"O*NET importance (1-5) of {t.name}: {t.candidate_importance} for your occupation, "
                             f"{t.target_importance} for the target occupation", meta={"source": t.source}), jid)
                    sents.append({"text": text, "citations": [gid]})
                if not gap.transferable and not gap.occupation_transferable:
                    sents.append({"text": "I found no related-skill bridges for this job's missing skills.",
                                  "citations": []})
    elif intent == "related_occupations":
        codes = [o.code for o in cand.occupations if o.code]
        rel = related_occupations(codes, limit=6)
        if not rel:
            sents.append({"text": "Your resume did not map to an O*NET occupation, so I cannot list related ones.",
                          "citations": []})
        from explanations.evidence_registry import EvidenceItem

        for r in rel:
            mask += [r.title, r.via or ""]
            gid = rb.add_item(EvidenceItem(cite_id="", kind="graph", text=f"O*NET related_occupations.csv: {r.via} -> "
                                           f"{r.title} ({r.tier})", meta={"source": "O*NET"}))
            sents.append({"text": f"O*NET lists {r.title} as related to {r.via} ({r.tier}).", "citations": [gid]})
    elif intent == "improve_resume":
        sents = state.get("sentences") or [{"text": "See the suggestions below.", "citations": []}]
        rb_reg = state.get("registry")
        return {**state, "intent": intent, "trace": trace, "sentences": sents, "registry": rb_reg}
    elif intent in ("outreach", "followup"):
        return {**state, "intent": intent, "trace": trace}
    elif intent == "unsupported_action":
        sents.append({"text": UNSUPPORTED_REPLY, "citations": []})
    else:
        sents.append({"text": HELP, "citations": []})
    return {**state, "intent": intent, "trace": trace, "sentences": sents, "registry": rb.reg.model_dump(),
            "data": {**state.get("data", {}), "mask_phrases": [m for m in mask if m],
                     "allowed_numbers": sorted(allowed | set(state.get("data", {}).get("allowed_numbers", [])))}}
