"""Critic / Verifier Agent: every factual sentence must be supported by cited evidence.

Sentences are produced as {text, citations}. The critic renders them with
citation markers, runs the explanation verifier against the evidence registry
and drops unsupported sentences before anything is shown to the user.
"""
from __future__ import annotations

import re

from agents.state import AgentState
from explanations.evidence_registry import EvidenceRegistry
from explanations.verifier import verify


def render(sentences: list[dict]) -> str:
    return "\n".join(s["text"] + (" " + " ".join(f"[{c}]" for c in s.get("citations", [])) if s.get("citations") else "")
                     for s in sentences)


def strip_citations(text: str) -> str:
    """'… which the posting asks for. [C1] [J1]' -> '… which the posting asks for.'"""
    return re.sub(r"[ \t]*\[[A-Za-z]+\d+\]", "", text)


def critic_agent(state: AgentState) -> AgentState:
    trace = state.get("trace", []) + ["critic"]
    sentences = state.get("sentences") or []
    draft = state.get("draft") or render(sentences)
    reg_data = state.get("registry")
    if not reg_data:
        # no evidence registry: only non-factual text may pass
        reg = EvidenceRegistry(candidate_id=state.get("candidate_id") or "", job_id=state.get("job_id") or "")
    else:
        reg = EvidenceRegistry.model_validate(reg_data)
    allowed = set(state.get("data", {}).get("allowed_numbers", []))
    v = verify(draft, reg, allowed_numbers=allowed, mask_phrases=state.get("data", {}).get("mask_phrases", []))
    citations = [i.model_dump() for i in reg.items if any(i.cite_id in s["valid_citations"] for s in v["sentences"])]
    report = {k: v[k] for k in ("n_sentences", "n_factual", "n_supported", "citation_validity", "faithfulness",
                                "hallucination_rate")}
    report["rejected"] = [{"sentence": r["sentence"], "reasons": r["reasons"]} for r in v["rejected"]]
    preface = state.get("data", {}).get("preface", "")
    body = v["verified_text"] or ""
    if state.get("data", {}).get("hide_citations"):
        # text meant to be sent to someone else: verified with its markers, shown without them (the evidence
        # itself stays in ``citations``)
        body = strip_citations(body)
    answer = (preface + "\n\n" if preface else "") + body
    return {**state, "answer": answer.strip(), "body": body.strip(), "citations": citations, "critic": report,
            "trace": trace}
