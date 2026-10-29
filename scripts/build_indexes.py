"""Build taxonomy, parse jobs, and build BM25 / TF-IDF / FAISS indexes.

  python scripts/build_indexes.py                 # uses Data/processed/jobs.parquet
  python scripts/build_indexes.py --no-dense      # skip embeddings
  python scripts/build_indexes.py --rebuild-taxonomy
  python scripts/build_indexes.py --add-new       # incremental: embed only jobs missing from FAISS
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import get_settings  # noqa: E402
from common.privacy import configure_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--rebuild-taxonomy", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="re-parse jobs ignoring the parse cache")
    ap.add_argument("--add-new", action="store_true", help="incremental FAISS update")
    a = ap.parse_args()
    configure_logging("WARNING")
    s = get_settings()
    t0 = time.time()
    report: dict = {"mode": s.mode}

    from normalization import taxonomy

    if a.rebuild_taxonomy or not (s.path("normalized") / "skills.parquet").exists():
        print("building taxonomy ...", flush=True)
        report["taxonomy"] = taxonomy.build()
    from matching.occupation_family import model_path, train_and_evaluate

    if not model_path().exists():
        print("training occupation-family LR on O*NET ...", flush=True)
        report["occupation_family"] = train_and_evaluate()

    from ingestion.pipeline import load_jobs
    from parsing.job_parser import parse_jobs

    jobs = load_jobs(indexable_only=True)
    print(f"parsing {len(jobs)} jobs (classical) ...", flush=True)
    tp = time.time()
    parsed = parse_jobs(jobs, use_cache=not a.no_cache, progress=lambda m: print(" ", m, flush=True))
    report["parse_seconds"] = round(time.time() - tp, 1)
    report["jobs"] = len(parsed)
    report["jobs_with_skill_requirements"] = sum(1 for j in parsed if j.requirement_skill_ids())
    report["mean_skill_requirements"] = round(sum(len(j.requirement_skill_ids()) for j in parsed) / max(1, len(parsed)), 2)
    report["jobs_with_occupation"] = sum(1 for j in parsed if j.occupation)

    from matching.store import JobStore, save_store
    import pandas as pd

    path = save_store(parsed)
    df = pd.read_parquet(path)
    splits = pd.read_parquet(s.path("processed") / "jobs.parquet", columns=["job_id", "split", "dedup_cluster"])
    df = df.merge(splits, on="job_id", how="left")  # leakage-safe cluster-level research splits
    df.to_parquet(path, index=False)
    store = JobStore(df)
    idx = s.path("indexes")
    idx.mkdir(parents=True, exist_ok=True)
    texts = store.texts()

    from matching.bm25 import BM25
    from matching.tfidf import TfidfRetriever

    print("building BM25 + TF-IDF ...", flush=True)
    BM25().fit(texts).save(idx / "bm25.pkl")
    TfidfRetriever().fit(texts).save(idx / "tfidf.pkl")

    if not a.no_dense and s.get("matching.use_dense", True):
        from common.embeddings import get_encoder
        from matching.dense import DenseIndex

        enc = get_encoder()
        if enc is None:
            report["dense"] = "skipped: embedding model unavailable"
        else:
            fpath = idx / "faiss_jobs.index"
            td = time.time()
            if a.add_new and fpath.exists():
                dense = DenseIndex.load(fpath)
                have = set(int(i) for i in __import__("faiss").vector_to_array(dense.index.id_map))
                new = [i for i in range(len(texts)) if i not in have]
                if new:
                    dense.add(enc.encode([texts[i] for i in new]), new)
                report["dense"] = {"added": len(new)}
            else:
                print(f"embedding {len(texts)} jobs with {enc.model_name} ...", flush=True)
                dense = DenseIndex.build(texts, list(range(len(texts))), enc,
                                         progress=lambda m: print(" ", m, flush=True))
                report["dense"] = {"vectors": int(dense.index.ntotal), "seconds": round(time.time() - td, 1)}
            dense.save(fpath)
            (idx / "faiss_jobs_ids.json").write_text(json.dumps(store.ids))
    report["seconds"] = round(time.time() - t0, 1)
    (idx / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
