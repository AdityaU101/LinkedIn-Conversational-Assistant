"""Critic / Verifier for generated explanations and other factual text.

For each sentence:
  1. extract citations [C#]/[J#]/[G#]; any ID not in the registry is invalid
  2. detect skill mentions (taxonomy gazetteer) and numbers
  3. a sentence is SUPPORTED iff it is non-factual (no skills, no numbers,
     no experience claims) or every claim is backed by a cited item:
       - each mentioned skill appears in at least one cited item
       - a skill presented as something the candidate HAS must be cited by a
         candidate item [C#] containing it (or a graph item for "related")
       - every number appears in a cited item or the computed summary
  4. unsupported sentences are rejected and removed from the final text
Metrics: citation validity, faithfulness (supported / factual sentences),
hallucination rate (sentences asserting skills/experience absent from evidence).
"""
from __future__ import annotations

import re
from typing import Optional

from explanations.evidence_registry import EvidenceRegistry
from normalization.skill_normalizer import get_skill_normalizer

CITE = re.compile(r"\[([CJG]\d+)\]")
CITE_ANY = re.compile(r"\[([A-Za-z]+\d+)\]")
NUMBER = re.compile(r"(?<![\w\[])(\d+(?:\.\d+)?)\s*(\+|%|years?|yrs?)?", re.I)
HAS_CUES = re.compile(r"\b(you have|you've|your (?:experience|background|resume|skills?)|candidate has|has experience|"
                      r"experienced in|proficient|skilled|demonstrat|worked with|used|built|led|your)\b", re.I)
MISSING_CUES = re.compile(r"\b(missing|lack|lacks|not (?:mention|list|show)|no evidence|gap|would need|requires?|"
                          r"required|asks for|looking for|preferred|develop|learn)\b", re.I)
QUALIFICATION_CLAIM = re.compile(r"\b(you are (?:proficient|skilled|experienced|an expert|certified)|you have (?:experience|"
                                 r"expertise|a degree|\d)|proficient in|skilled in|expert in|certified in|"
                                 r"(?:extensive|strong|deep|hands-on) experience)\b", re.I)
EXPERIENCE_CLAIM = re.compile(r"\b(\d+\+?\s*years?|senior|expert|extensive experience)\b", re.I)


_ABBREV = re.compile(r"(?:\b(?:Sr|Jr|Mr|Ms|Mrs|Dr|St|Inc|Co|Corp|Ltd|vs|No|Mt|Ft|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|"
                     r"Sept|Oct|Nov|Dec|e\.g|i\.e|U\.S|approx)|\b[A-Z])\.$")


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in re.split(r"\n+", text.strip()):
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\-•*#])", line)
        merged: list[str] = []
        for p in parts:
            if merged and _ABBREV.search(merged[-1]):   # 'Sr. Python Developer' is one sentence
                merged[-1] += " " + p
            else:
                merged.append(p)
        out.extend(merged)
    return [p.strip(" -•*") for p in out if p and p.strip(" -•*")]


def verify(text: str, registry: EvidenceRegistry, allowed_numbers: Optional[set[str]] = None,
           mask_phrases: Optional[list[str]] = None) -> dict:
    """``mask_phrases``: entity names (job titles, companies, occupation titles) that are not skill claims;
    they are removed before skill spotting."""
    sk = get_skill_normalizer()
    allowed_numbers = allowed_numbers or set()
    results = []
    total_cites = valid_cites = 0
    factual = supported = hallucinated = 0
    for sent in split_sentences(text):
        cites = CITE_ANY.findall(sent)
        total_cites += len(cites)
        valid = [c for c in cites if registry.get(c)]
        valid_cites += len(valid)
        invalid = [c for c in cites if not registry.get(c)]
        body = CITE_ANY.sub("", sent)
        for ph in sorted(mask_phrases or [], key=len, reverse=True):
            if ph:
                body = re.sub(re.escape(ph), " ", body, flags=re.I)
        # strict spotting: list-context restrictions relaxed so plain-word skills ('Rust', 'Go') are caught
        mentions = {m.canonical_id: m.canonical_name
                    for m in sk.extract(body, "explanation", section_of=lambda _o: "skills") if m.canonical_id}
        numbers = {n.group(1) for n in NUMBER.finditer(body)}
        qualification = bool(QUALIFICATION_CLAIM.search(body)) and not MISSING_CUES.search(body)
        is_factual = bool(mentions or numbers or EXPERIENCE_CLAIM.search(body) or qualification)
        reasons = []
        if invalid:
            reasons.append(f"invalid citation(s): {', '.join(invalid)}")
        cited = [registry.get(c) for c in valid]
        cited_text = " ".join(i.text for i in cited) + " " + " ".join(n for i in cited for n in i.skill_names)
        claims_has = bool(HAS_CUES.search(body)) and not MISSING_CUES.search(body)
        asserted_absent = []
        for sid, name in mentions.items():
            in_cited = [i for i in cited if sid in i.skill_ids or sid in i.meta.get("related_skill_id", "")]
            if not in_cited:
                reasons.append(f"skill '{name}' not supported by cited evidence")
                # skill asserted as a candidate qualification but absent from all candidate evidence
                if claims_has and not any(sid in i.skill_ids for i in registry.items if i.kind == "candidate") \
                        and not any(i.meta.get("related_skill_id") == sid for i in registry.items):
                    asserted_absent.append(name)
            elif claims_has and not any(i.kind in ("candidate", "graph") for i in in_cited) \
                    and any(i.kind == "job" for i in in_cited) and not MISSING_CUES.search(body):
                reasons.append(f"'{name}' presented as a candidate skill but only job evidence is cited")
                if not any(sid in i.skill_ids for i in registry.items if i.kind == "candidate"):
                    asserted_absent.append(name)
        for n in numbers:
            if n not in cited_text and n not in allowed_numbers:
                reasons.append(f"number '{n}' not found in cited evidence")
        if qualification and not any(i.kind in ("candidate", "graph") for i in cited):
            reasons.append("qualification claim without candidate evidence")
            if not mentions:
                asserted_absent.append("(unnamed qualification)")
        computed_only = bool(numbers) and numbers <= allowed_numbers and not mentions  # e.g. coverage summary
        if is_factual and not valid and not computed_only:
            reasons.append("factual sentence without valid citation")
        ok = not reasons
        if is_factual:
            factual += 1
            supported += int(ok)
            hallucinated += int(bool(asserted_absent))
        results.append({"sentence": sent, "citations": cites, "valid_citations": valid, "factual": is_factual,
                        "supported": ok, "reasons": reasons,
                        "hallucinated_skills": asserted_absent})
    kept = [r["sentence"] for r in results if r["supported"]]
    return {
        "sentences": results,
        "verified_text": "\n".join(kept),
        "n_sentences": len(results), "n_factual": factual, "n_supported": supported,
        "rejected": [r for r in results if not r["supported"]],
        "citation_validity": round(valid_cites / total_cites, 4) if total_cites else None,
        "faithfulness": round(supported / factual, 4) if factual else None,
        "hallucination_rate": round(hallucinated / factual, 4) if factual else None,
    }
