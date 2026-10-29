"""Research & Outreach Agent — grounded company facts + outreach DRAFTS.

Company research uses only stored data (Kaggle companies.csv description,
specialities, industries). Drafts mention only skills the candidate's resume
evidences. Nothing is ever sent: drafts require the user's approval and the
user sends them personally.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pandas as pd

from agents.evidence_tools import RegistryBuilder
from agents.state import AgentState
from common.config import get_settings
from common.schemas import Candidate
from explanations.evidence_registry import EvidenceItem
from matching.skill_gap import compute_gap
from matching.store import get_store
from parsing.classical_parser import candidate_name


@lru_cache(maxsize=1)
def _companies() -> dict[str, dict]:
    base = get_settings().path("raw") / "linkedin_jobs" / "companies"
    if not (base / "companies.csv").exists():
        return {}
    df = pd.read_csv(base / "companies.csv", usecols=["company_id", "name", "description", "company_size", "city",
                                                       "state", "country", "url"])
    spec = pd.read_csv(base / "company_specialities.csv").groupby("company_id")["speciality"].apply(lambda s: list(s)[:8])
    ind = pd.read_csv(base / "company_industries.csv").groupby("company_id")["industry"].apply(list)
    out = {}
    for r in df.itertuples(index=False):
        out[str(r.company_id)] = {"name": r.name, "description": r.description if isinstance(r.description, str) else None,
                                  "company_size_bucket": None if pd.isna(r.company_size) else int(r.company_size),
                                  "location": ", ".join(str(x) for x in (r.city, r.state, r.country) if isinstance(x, str)),
                                  "linkedin_url": r.url, "specialities": spec.get(r.company_id, []),
                                  "industries": ind.get(r.company_id, [])}
    return out


def company_research(company_id: Optional[str]) -> Optional[dict]:
    if not company_id:
        return None
    info = _companies().get(str(company_id))
    if not info:
        return None
    return {**info, "source": "Kaggle LinkedIn companies.csv / company_specialities.csv / company_industries.csv"}


def outreach_draft(cand: Candidate, job_id: str) -> dict:
    job = get_store().by_id(job_id)
    if job is None:
        raise KeyError(job_id)
    gap = compute_gap(cand, job)
    rb = RegistryBuilder(cand.candidate_id, job_id)
    rb.add_gap(gap)
    research = company_research(job.company_id)
    if research and research.get("description"):
        rb.add_item(EvidenceItem(cite_id="", kind="job", text=research["description"][:400],
                                 meta={"source": research["source"]}), job_id)
    matched = (gap.matched_required + gap.matched_preferred)[:3]
    title, company = job.title or "the role", job.company or "your team"
    name = candidate_name(cand.raw_text)       # None when the resume does not open with a clear name
    intro = f"I, {name}, am interested" if name else "I am interested"
    sentences = [{"text": f"Hello {company} hiring team,", "citations": []},
                 {"text": f"{intro} in the {title} position.", "citations": []}]
    if matched:
        # one sentence naming every matched skill, carrying the citations of all of them
        names = [m.name for m in matched]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        cites = [c for m in matched
                 for c in rb.cites_for(m.skill_id, "candidate") + rb.cites_for(m.skill_id, "job", job_id)]
        sentences.append({"text": f"My resume shows experience with {listed}, which the posting asks for.",
                          "citations": list(dict.fromkeys(cites))})
    if gap.missing_required:
        sentences.append({"text": "I would welcome the chance to discuss how my background fits the remaining "
                                  "requirements.", "citations": []})
    sentences.append({"text": "Thank you for your time.", "citations": []})
    return {"job_id": job_id, "title": title, "company": company, "candidate_name": name, "sentences": sentences,
            "registry": rb.reg.model_dump(), "company_research": research,
            "status": "draft_requires_user_approval", "sent": False}


def research_outreach_agent(state: AgentState) -> AgentState:
    trace = state.get("trace", []) + ["research_outreach"]
    if not state.get("candidate") or not state.get("job_id"):
        return {**state, "trace": trace,
                "errors": state.get("errors", []) + ["Outreach needs a candidate and a job_id"]}
    d = outreach_draft(Candidate.model_validate(state["candidate"]), state["job_id"])
    return {**state, "trace": trace, "sentences": d["sentences"], "registry": d["registry"], "requires_approval": True,
            "data": {**state.get("data", {}), "outreach": {k: v for k, v in d.items() if k not in ("sentences", "registry")},
                     "mask_phrases": [p for p in (d["title"], d["company"], d["candidate_name"]) if p],
                     "hide_citations": True,
                     "preface": "Draft only — review and edit before sending it yourself. Nothing has been sent."}}
