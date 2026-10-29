"""Within- and cross-source deduplication of canonical jobs.

Signals (each recorded so the report can attribute every merge):
  1. identical ``job_id``
  2. identical normalised description hash
  3. identical normalised (title, company, location) key
  4. MinHash-LSH Jaccard ≥ threshold on word shingles of the description

Records are grouped into clusters with union-find. One representative per
cluster is kept for the retrieval index; research splits are assigned per
*cluster*, so near-duplicates can never leak across train/dev/test.
"""
from __future__ import annotations

import zlib
from collections import Counter, defaultdict
from typing import Optional

import pandas as pd

from common.text import normalize_for_hash, normalize_key
from ingestion.normalize import is_masked

# Higher = preferred as cluster representative (live > richest historical).
SOURCE_PRIORITY = {"adzuna": 5, "remoteok": 4, "linkedin_kaggle": 3, "datastax_linkedin": 2, "linkedin_insights": 0}


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


def shingles(text: str, k: int = 5) -> set[str]:
    words = normalize_for_hash(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def deduplicate(df: pd.DataFrame, use_minhash: bool = True, threshold: float = 0.85,
                num_perm: int = 64, shingle_size: int = 5) -> tuple[pd.DataFrame, dict]:
    """Return ``df`` with columns dedup_cluster, is_representative, dup_reason; plus a report."""
    df = df.reset_index(drop=True).copy()
    n = len(df)
    uf = UnionFind(n)
    reasons: dict[int, set[str]] = defaultdict(set)
    merges: Counter = Counter()
    cross: Counter = Counter()

    def link_groups(keys: pd.Series, reason: str) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, k in enumerate(keys):
            if isinstance(k, str) and k:  # NaN/None never form a group
                groups[k].append(i)
        for idxs in groups.values():
            for j in idxs[1:]:
                if uf.union(idxs[0], j):
                    merges[reason] += 1
                reasons[j].add(reason)
                reasons[idxs[0]].add(reason)
                a, b = sorted((df.at[idxs[0], "source"], df.at[j, "source"]))
                if a != b:
                    cross[f"{a}|{b}|{reason}"] += 1

    link_groups(df["job_id"], "same_job_id")
    has_desc = df["description"].fillna("").str.len() > 50
    link_groups(pd.Series([h if ok else None for h, ok in zip(df["content_hash"], has_desc)]), "description_hash")
    keys = [
        "|".join((normalize_key(t), normalize_key(c), normalize_key(l))) if normalize_key(t) and normalize_key(c) else None
        for t, c, l in zip(df["title"], df["company"], df["location"])
    ]
    link_groups(pd.Series(keys), "title_company_location")

    if use_minhash and has_desc.any():
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        sigs: dict[int, MinHash] = {}
        for i in df.index[has_desc]:
            sh = shingles(df.at[i, "description"], shingle_size)
            if not sh:
                continue
            m = MinHash(num_perm=num_perm, seed=13)
            m.update_batch([s.encode("utf-8") for s in sh])
            for j in lsh.query(m):
                if uf.find(i) != uf.find(j):
                    if m.jaccard(sigs[j]) >= threshold:
                        uf.union(i, j)
                        merges["minhash_near_duplicate"] += 1
                        reasons[i].add("minhash_near_duplicate")
                        reasons[j].add("minhash_near_duplicate")
                        a, b = sorted((df.at[i, "source"], df.at[j, "source"]))
                        if a != b:
                            cross[f"{a}|{b}|minhash_near_duplicate"] += 1
            lsh.insert(i, m)
            sigs[i] = m

    roots = [uf.find(i) for i in range(n)]
    df["dedup_cluster"] = [f"c{r:07d}" for r in roots]
    df["dup_reason"] = [";".join(sorted(reasons.get(i, []))) or None for i in range(n)]

    # Representative: a readable title first (never a source-redacted '**** ****' copy over its twin), then highest
    # source priority, then has description, then longest description.
    readable = ~df["title"].map(is_masked)
    score = (readable.astype(int) * 1e8 + df["source"].map(SOURCE_PRIORITY).fillna(1) * 1e7
             + has_desc.astype(int) * 1e6 + df["description"].fillna("").str.len().clip(upper=999_999))
    rep_idx = score.groupby(df["dedup_cluster"]).idxmax()
    df["is_representative"] = False
    df.loc[rep_idx.values, "is_representative"] = True

    sizes = df["dedup_cluster"].value_counts()
    report = {
        "records": n,
        "clusters": int(len(sizes)),
        "duplicate_records": int(n - len(sizes)),
        "duplicate_rate": round((n - len(sizes)) / n, 4) if n else 0.0,
        "merges_by_signal": dict(merges),
        "cross_source_links": dict(cross),
        "largest_clusters": sizes.head(5).astype(int).to_dict(),
        "per_source": {
            s: {"records": int((df["source"] == s).sum()),
                "representatives": int(((df["source"] == s) & df["is_representative"]).sum())}
            for s in sorted(df["source"].unique())
        },
        "params": {"minhash": use_minhash, "threshold": threshold, "num_perm": num_perm, "shingle_size": shingle_size},
    }
    return df, report


def assign_splits(clusters: pd.Series, ratios=(0.7, 0.15, 0.15), seed: int = 13) -> pd.Series:
    """Deterministic cluster-level split: all members of a cluster share a split."""
    def which(c: str) -> str:
        u = (zlib.crc32(f"{seed}:{c}".encode()) & 0xFFFFFFFF) / 0xFFFFFFFF
        return "train" if u < ratios[0] else ("dev" if u < ratios[0] + ratios[1] else "test")

    return clusters.map(which)


def check_no_leakage(df: pd.DataFrame, split_col: str = "split") -> Optional[str]:
    """Return a description of the first leaking cluster, or None if splits are clean."""
    per = df.groupby("dedup_cluster")[split_col].nunique()
    bad = per[per > 1]
    return None if bad.empty else f"cluster {bad.index[0]} spans {int(bad.iloc[0])} splits"
