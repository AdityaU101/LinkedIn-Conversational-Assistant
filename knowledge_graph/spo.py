"""Free-form subject–predicate–object extraction (RQ3 ablation baseline).

Runs on the SAME documents as the ontology-constrained extractor:
* classical: spaCy dependency parse (nsubj/dobj/pobj/attr around a verb)
* llm: free-form triples via the router (task 'spo'), grounded like other LLM output
Entities and predicates are kept as surface strings (lower-cased) — exactly
what makes free-form graphs fragment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from common.text import find_span, word_chunks


@dataclass
class Triple:
    doc_id: str
    subject: str
    predicate: str
    object: str
    evidence: str
    extractor: str


def _span_text(tok) -> str:
    keep = [t for t in tok.subtree if t.dep_ not in ("punct", "cc", "det") and not t.is_space]
    keep = keep[:6]
    return " ".join(t.text for t in keep).strip()


def classical_spo(doc_id: str, text: str, max_chars: int = 20000) -> list[Triple]:
    from parsing.classical_parser import _spacy

    nlp = _spacy()
    if nlp is None:
        raise RuntimeError("spaCy model en_core_web_sm is required for classical SPO extraction")
    out: list[Triple] = []
    doc = nlp(text[:max_chars])
    for sent in doc.sents:
        first = next((t for t in sent if not t.is_punct and not t.is_space), None)
        if first is not None and first.pos_ == "VERB" and first.dep_ in ("amod", "compound", "nmod") \
                and not any(c.dep_.startswith("nsubj") for c in first.children):
            # resume bullet 'Managed payroll for 200 employees': spaCy parses the verb as a modifier of
            # the noun root; read it as (<author>, manage, payroll) + prepositional objects of that noun
            head = first.head
            pred = (first.lemma_ or first.text).lower()
            obj = " ".join(t.text for t in sent if first.i < t.i <= head.i and not t.is_punct) or head.text
            out.append(Triple(doc_id, "<author>", pred, obj.lower(), sent.text.strip()[:300], "spacy_dependency"))
            for prep in (c for c in head.children if c.dep_ == "prep"):
                for g in (c for c in prep.children if c.dep_ == "pobj"):
                    out.append(Triple(doc_id, "<author>", f"{pred} {prep.text}".lower(), _span_text(g).lower(),
                                      sent.text.strip()[:300], "spacy_dependency"))
        for tok in sent:
            if tok.pos_ not in ("VERB", "AUX"):
                continue
            subs = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
            objs = [c for c in tok.children if c.dep_ in ("dobj", "attr", "oprd")]
            for prep in (c for c in tok.children if c.dep_ == "prep"):
                objs += [(prep, g) for g in prep.children if g.dep_ == "pobj"]
            if not subs:
                # resumes are mostly subject-less bullets ("Managed payroll ...") — implicit first person
                subs = ["<author>"] if sent.root == tok else []
            for s in subs:
                s_txt = s if isinstance(s, str) else _span_text(s)
                for o in objs:
                    if isinstance(o, tuple):
                        pred = f"{tok.lemma_ or tok.text} {o[0].text}".lower()
                        o_txt = _span_text(o[1])
                    else:
                        pred = (tok.lemma_ or tok.text).lower()
                        o_txt = _span_text(o)
                    if s_txt and o_txt:
                        out.append(Triple(doc_id, s_txt.lower(), pred, o_txt.lower(), sent.text.strip()[:300],
                                          "spacy_dependency"))
    return out


def llm_spo(doc_id: str, text: str, provider: Optional[str] = None, model: Optional[str] = None) -> list[Triple]:
    from llm.prompts.extraction import spo_messages
    from llm.router import get_router
    from llm.schemas.extraction import SPOExtraction

    router = get_router()
    out = []
    for _, _, chunk in word_chunks(text, 350, 0.10):
        res = router.complete_json("spo", spo_messages(chunk), schema=SPOExtraction, provider=provider, model=model)
        for t in res.parsed.triples:
            if find_span(chunk, t.evidence):  # grounded only
                out.append(Triple(doc_id, t.subject.lower().strip(), t.predicate.lower().strip(),
                                  t.object.lower().strip(), t.evidence, f"llm:{res.provider}/{res.model}"))
    return out
