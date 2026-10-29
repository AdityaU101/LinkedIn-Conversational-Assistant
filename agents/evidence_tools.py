"""Helpers to assemble one evidence registry across several jobs."""
from __future__ import annotations

from explanations.evidence_registry import EvidenceItem, EvidenceRegistry, build_registry
from matching.skill_gap import SkillGap


class RegistryBuilder:
    def __init__(self, candidate_id: str, job_id: str = "*"):
        self.reg = EvidenceRegistry(candidate_id=candidate_id, job_id=job_id)
        self.counters = {"C": 0, "J": 0, "G": 0}
        self._by_key: dict[str, str] = {}

    def add_item(self, item: EvidenceItem, job_id: str | None = None) -> str:
        key = f"{item.kind}|{item.evidence_id or item.text}|{job_id if item.kind != 'candidate' else ''}"
        if key in self._by_key:
            cur = self.reg.get(self._by_key[key])
            for sid, name in zip(item.skill_ids, item.skill_names):
                if sid not in cur.skill_ids:
                    cur.skill_ids.append(sid)
                    cur.skill_names.append(name)
            return cur.cite_id
        p = {"candidate": "C", "job": "J", "graph": "G"}[item.kind]
        self.counters[p] += 1
        new = item.model_copy(update={"cite_id": f"{p}{self.counters[p]}", "meta": {**item.meta, "job_id": job_id}})
        self.reg.items.append(new)
        self._by_key[key] = new.cite_id
        return new.cite_id

    def add_gap(self, gap: SkillGap) -> dict[str, str]:
        """Add a gap's evidence; returns old cite id -> new cite id."""
        local = build_registry(gap)
        return {i.cite_id: self.add_item(i, gap.job_id) for i in local.items}

    def cites_for(self, skill_id: str, kind: str, job_id: str | None = None, limit: int = 1) -> list[str]:
        out = [i.cite_id for i in self.reg.items if i.kind == kind and skill_id in i.skill_ids
               and (job_id is None or kind == "candidate" or i.meta.get("job_id") == job_id)]
        return out[:limit]
