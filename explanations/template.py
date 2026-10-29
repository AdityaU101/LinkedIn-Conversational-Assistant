"""Classical template explanation (every sentence carries citations by construction)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from explanations.evidence_registry import EvidenceRegistry, build_registry
from matching.skill_gap import SkillGap


class Explanation(BaseModel):
    method: str
    candidate_id: str
    job_id: str
    sentences: list[dict] = Field(default_factory=list)   # {text, citations}
    text: str = ""
    registry: EvidenceRegistry
    verification: dict = Field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    latency_ms: float | None = None


def _cites(reg: EvidenceRegistry, skill_id: str, kind: str) -> list[str]:
    return [i.cite_id for i in reg.items if i.kind == kind and skill_id in i.skill_ids][:2]


def template_explanation(gap: SkillGap, registry: EvidenceRegistry | None = None) -> Explanation:
    reg = registry or build_registry(gap)
    sents: list[dict] = []
    for m in gap.matched_required + gap.matched_preferred:
        c = _cites(reg, m.skill_id, "candidate") + _cites(reg, m.skill_id, "job")
        sents.append({"text": f"Matched{' (preferred)' if m.importance == 'preferred' else ''}: {m.name}.",
                      "citations": c, "claim": {"type": "matched", "skill_id": m.skill_id}})
    for ms in gap.missing_required:
        sents.append({"text": f"Missing (required): {ms.name}.", "citations": _cites(reg, ms.skill_id, "job"),
                      "claim": {"type": "missing", "skill_id": ms.skill_id}})
    for ms in gap.missing_preferred:
        sents.append({"text": f"Missing (preferred): {ms.name}.", "citations": _cites(reg, ms.skill_id, "job"),
                      "claim": {"type": "missing", "skill_id": ms.skill_id}})
    for ms in gap.transferable:
        g = _cites(reg, ms.skill_id, "graph")
        if g:
            b = ms.bridges[0]
            sents.append({"text": f"Transferable: {b.candidate_skill_name} is related to {ms.name} (inferred).",
                          "citations": g, "claim": {"type": "transferable", "skill_id": ms.skill_id}})
    if gap.coverage_required is not None:
        n = len(gap.matched_required)
        tot = n + len(gap.missing_required)
        sents.append({"text": f"Required-skill coverage: {n} of {tot} ({gap.coverage_required:.0%}).",
                      "citations": [], "claim": {"type": "summary"}})
    text = "\n".join(s["text"] + (" " + " ".join(f"[{c}]" for c in s["citations"]) if s["citations"] else "")
                     for s in sents)
    return Explanation(method="template", candidate_id=gap.candidate_id, job_id=gap.job_id, sentences=sents,
                       text=text, registry=reg)
