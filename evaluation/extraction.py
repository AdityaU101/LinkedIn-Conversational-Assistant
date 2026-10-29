"""Experiment A — extraction precision/recall/F1 against human span annotations."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Optional

from annotation.agreement import span_f1
from annotation.status import root
from evaluation.registry import write_result

LABELS = ["Occupation", "Skill", "Experience", "Education", "Credential", "Project"]


def load_gold() -> dict[str, dict[str, set]]:
    """doc_id -> annotator -> {(label, start, end)}"""
    p = root() / "extraction" / "annotations.jsonl"
    gold: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                gold[d["doc_id"]][d["annotator_id"]].add((d["label"], int(d["start"]), int(d["end"])))
    return gold


def predicted_spans(cand) -> set:
    out = set()
    for label, items in (("Occupation", cand.occupations), ("Skill", cand.skills), ("Experience", cand.experiences),
                         ("Education", cand.education), ("Credential", cand.credentials), ("Project", cand.projects)):
        for it in items:
            for ev in it.evidence:
                out.add((label, ev.start_offset, ev.end_offset))
    return out


JOB_KIND_LABEL = {"skill": "Skill", "education": "Education", "credential": "Credential", "experience": "Experience"}


def _job(doc_id: str, text: str):
    from common.schemas import Job

    src, _, sid = doc_id.partition(":")
    return Job(job_id=doc_id, source_job_id=sid or doc_id, source=src or "eval", description=text)


def job_spans(job) -> set:
    out = set()
    for r in job.requirements:
        lab = JOB_KIND_LABEL.get(r.kind)
        for ev in r.evidence if lab else []:
            out.add((lab, ev.start_offset, ev.end_offset))
    if job.occupation:
        for ev in job.occupation.evidence:
            out.add(("Occupation", ev.start_offset, ev.end_offset))
    return out


def _overlap_match(pred: set, gold: set) -> tuple[int, int, int]:
    """Relaxed matching: same label and overlapping spans (one-to-one, greedy)."""
    used = set()
    tp = 0
    for (l, s, e) in sorted(pred):
        for g in gold:
            if g in used or g[0] != l:
                continue
            if s < g[2] and g[1] < e:
                used.add(g)
                tp += 1
                break
    return tp, len(pred) - tp, len(gold) - tp


def score(pred_by_doc: dict[str, set], gold_by_doc: dict[str, set]) -> dict:
    res = {}
    for mode in ("exact", "overlap"):
        per = {l: [0, 0, 0] for l in LABELS}
        for doc, gold in gold_by_doc.items():
            pred = pred_by_doc.get(doc, set())
            for l in LABELS:
                pl = {x for x in pred if x[0] == l}
                gl = {x for x in gold if x[0] == l}
                if mode == "exact":
                    tp = len(pl & gl)
                    fp, fn = len(pl) - tp, len(gl) - tp
                else:
                    tp, fp, fn = _overlap_match(pl, gl)
                per[l][0] += tp
                per[l][1] += fp
                per[l][2] += fn
        out = {}
        T = [sum(v[i] for v in per.values()) for i in range(3)]
        for l, (tp, fp, fn) in list(per.items()) + [("micro", T)]:
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            out[l] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                      "support": tp + fn}
        res[mode] = out
    return res


def run(max_docs: Optional[int] = None, n_silver: int = 200, n_silver_llm: int = 30) -> dict:
    from evaluation import silver

    silver_res = {"occupation_extraction": silver.run_occupations(n_silver, n_silver_llm)}
    gold = load_gold()
    if not gold:
        return write_result("A_extraction", "partially_evaluated", silver=silver_res,
                            reason="Span-level P/R/F1 needs human extraction annotations "
                                   "(Data/annotations/extraction/annotations.jsonl). Evaluated instead: occupation "
                                   "extraction against resume_corpus expert labels (silver).")
    tasks = {t["data"]["doc_id"]: t["data"] for t in json.loads((root() / "extraction" / "tasks.json").read_text())}
    # adjudicated gold = union of the first annotator per doc (agreement reported separately)
    gold_doc = {d: next(iter(a.values())) for d, a in gold.items() if d in tasks}
    iaa = [span_f1(*list(a.values())[:2])["f1"] for a in gold.values() if len(a) >= 2]
    from parsing.classical_parser import ClassicalJobParser, ClassicalResumeParser
    from parsing.llm_parser import LLMParser

    cjp = ClassicalJobParser()

    docs = list(gold_doc)[:max_docs]
    cp = ClassicalResumeParser()
    t0 = time.perf_counter()
    classical = {d: (job_spans(cjp.parse(_job(d, tasks[d]["text"]))) if tasks[d].get("doc_type") == "job"
                     else predicted_spans(cp.parse(tasks[d]["text"], d, d))) for d in docs}
    lat_c = (time.perf_counter() - t0) * 1000 / max(1, len(docs))
    metrics = {"classical": score(classical, gold_doc), "classical_latency_ms_per_doc": round(lat_c, 1)}
    lp = LLMParser()
    status = "partially_evaluated"
    if lp.available():
        t0 = time.perf_counter()
        llm = {d: (job_spans(lp.parse_job(_job(d, tasks[d]["text"]))[0]) if tasks[d].get("doc_type") == "job"
                   else predicted_spans(lp.parse_resume(tasks[d]["text"], d, d)[0])) for d in docs}
        metrics["llm"] = score(llm, gold_doc)
        metrics["llm_latency_ms_per_doc"] = round((time.perf_counter() - t0) * 1000 / max(1, len(docs)), 1)
        status = "evaluated"
    return write_result("A_extraction", status, metrics, silver=silver_res,
                        reason=None if status == "evaluated" else "LLM extractor unavailable (no local model running)",
                        details={"documents": len(docs), "double_annotated": len(iaa),
                                 "iaa_span_f1_mean": round(sum(iaa) / len(iaa), 4) if iaa else None})
