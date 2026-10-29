"""Teacher-label generation for QLoRA training (RQ2).

A larger "teacher" model (configured via --provider/--model) extracts
structured output for documents in the TRAIN/DEV splits; only grounded items
(verbatim evidence found in the text) are kept. Labels are marked
source="teacher" and can be corrected by humans (source="human") before
training. Test-split documents are reserved for evaluation against human gold.

  python training/teacher_labels.py --provider groq --model llama-3.1-70b-versatile --jobs 300
Resume documents are only sent to local providers unless privacy.allow_remote_pii is true.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402,F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=int, default=200)
    a = ap.parse_args()
    from llm.prompts.extraction import job_messages
    from llm.router import get_router
    from llm.schemas.extraction import JobExtraction
    from common.text import find_span
    from matching.store import get_store
    from training.qlora import data_dir

    router = get_router()
    store = get_store()
    out = {"train": [], "dev": [], "test": []}
    n = 0
    for i in range(len(store)):
        split = store.df.at[i, "split"] if "split" in store.df else "train"
        if split == "test":
            continue
        job = store.job(i)
        text = (job.description or "")[:6000]
        if not text:
            continue
        res = router.complete_json("research", job_messages(text, job.title), schema=JobExtraction,
                                   provider=a.provider, model=a.model, experiment="teacher_labels")
        tgt = res.parsed.model_dump()
        tgt["requirements"] = [r for r in tgt["requirements"] if find_span(text, r["evidence"])]
        tgt["responsibilities"] = [r for r in tgt["responsibilities"] if find_span(text, r["evidence"])]
        out[split].append({"doc_id": job.job_id, "doc_type": "job", "split": split, "text": text, "target": tgt,
                           "source": "teacher", "teacher": f"{a.provider}/{a.model}"})
        n += 1
        if n >= a.jobs:
            break
    d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    for split, rows in out.items():
        with open(d / f"{split}.jsonl", "a") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
