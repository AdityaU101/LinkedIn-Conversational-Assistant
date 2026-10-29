"""End-to-end matching pipeline.

candidate -> filters -> {BM25, TF-IDF, Jaccard, Dense/FAISS} top-200 each
          -> RRF -> features -> LambdaMART (or documented fallback)
          -> cross-encoder rerank of top-20 -> MMR -> top-10
Every result keeps per-component scores and ranks for evaluation/debugging.
``method`` selects a single stage for comparisons (bm25, tfidf, jaccard,
dense, rrf, ltr, final).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from common.config import get_settings
from common.embeddings import get_encoder
from common.schemas import Candidate, Job
from matching import lambdamart
from matching.bm25 import BM25
from matching.cross_encoder import score_pairs
from matching.dense import DenseIndex
from matching.features import FEATURE_NAMES, CandidateContext, build_context, compute_features, fallback_score
from matching.jaccard import JaccardRetriever
from matching.mmr import minmax, mmr
from matching.rrf import rrf
from matching.skill_gap import SkillGap, compute_gap
from matching.store import JobFilters, JobStore, get_store
from matching.tfidf import TfidfRetriever
from normalization.taxonomy import load_taxonomy

log = logging.getLogger(__name__)
Method = Literal["bm25", "tfidf", "jaccard", "dense", "rrf", "ltr", "final"]
METHODS: list[str] = ["bm25", "tfidf", "jaccard", "dense", "rrf", "ltr", "final"]


def index_dir() -> Path:
    d = get_settings().path("indexes")
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class MatchResult:
    job_id: str
    rank: int
    score: float
    component_scores: dict[str, float] = field(default_factory=dict)
    component_ranks: dict[str, Optional[int]] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    gap: Optional[SkillGap] = None


@dataclass
class MatchResponse:
    candidate_id: str
    method: str
    results: list[MatchResult]
    stages: dict
    timings_ms: dict[str, float]
    ranker: str
    filtered_pool: int


class MatchingPipeline:
    def __init__(self, store: Optional[JobStore] = None, load: bool = True):
        self.store = store or get_store()
        self.tax = load_taxonomy()
        cfg = get_settings().section("matching")
        self.retrieve_k = int(cfg.get("retrieve_k", 200))
        self.rerank_k = int(cfg.get("rerank_k", 20))
        self.final_k = int(cfg.get("final_k", 10))
        self.rrf_k = int(cfg.get("rrf_k", 60))
        self.mmr_lambda = float(cfg.get("mmr_lambda", 0.7))
        self.ce_weight = float(cfg.get("cross_encoder_weight", 0.5))
        self.bm25: Optional[BM25] = None
        self.tfidf: Optional[TfidfRetriever] = None
        self.dense: Optional[DenseIndex] = None
        self.jaccard = JaccardRetriever(self.store.skill_sets)
        if load:
            self._load()

    def _load(self) -> None:
        d = index_dir()
        if (d / "bm25.pkl").exists():
            self.bm25 = BM25.load(d / "bm25.pkl")
        if (d / "tfidf.pkl").exists():
            self.tfidf = TfidfRetriever.load(d / "tfidf.pkl")
        if (d / "faiss_jobs.index").exists() and get_settings().get("matching.use_dense", True):
            try:
                self.dense = DenseIndex.load(d / "faiss_jobs.index")
            except Exception as e:
                log.warning("FAISS index failed to load: %s", e)
        if self.bm25 is None or self.tfidf is None:
            texts = self.store.texts()
            self.bm25 = self.bm25 or BM25().fit(texts)
            self.tfidf = self.tfidf or TfidfRetriever().fit(texts)

    @property
    def encoder(self):
        return get_encoder() if self.dense is not None else None

    def status(self) -> dict:
        return {"jobs": len(self.store), "bm25": self.bm25 is not None, "tfidf": self.tfidf is not None,
                "dense": self.dense is not None, "dense_vectors": int(self.dense.index.ntotal) if self.dense else 0,
                "lambdamart": lambdamart.status()["state"]}

    # ---------------------------------------------------------------- stages
    @staticmethod
    def _top(scores: np.ndarray, mask: np.ndarray, k: int, positive_only: bool = True) -> list[int]:
        s = np.where(mask, scores, -np.inf)
        if positive_only:
            s = np.where(s > 0, s, -np.inf)
        valid = int(np.isfinite(s).sum())
        if valid == 0:
            return []
        k = min(k, valid)
        idx = np.argpartition(-s, k - 1)[:k]
        return idx[np.argsort(-s[idx], kind="stable")].tolist()

    def component_scores(self, ctx: CandidateContext) -> dict[str, np.ndarray]:
        n = len(self.store)
        out = {"bm25": self.bm25.scores(ctx.query_text), "tfidf": self.tfidf.scores(ctx.query_text),
               "jaccard": self.jaccard.scores(ctx.skill_ids)}
        if self.dense is not None and ctx.qvec is not None:
            d = self.dense.all_scores(ctx.qvec)
            dense = np.full(n, -1.0, dtype=np.float32)
            dense[: len(d)] = d[:n]
            out["dense"] = dense
        return out

    def recommend(self, candidate: Candidate, filters: Optional[JobFilters] = None, method: Method = "final",
                  k: Optional[int] = None, with_gap: bool = True, use_cross_encoder: bool = True,
                  use_mmr: bool = True, drop_graph_features: bool = False) -> MatchResponse:
        t = {}
        t0 = time.perf_counter()
        k = k or self.final_k
        ctx = build_context(candidate, self.tax, self.encoder)
        t["context"] = (time.perf_counter() - t0) * 1000
        mask = self.store.mask(filters)
        pool = int(mask.sum())
        t1 = time.perf_counter()
        comp = self.component_scores(ctx)
        t["retrieval"] = (time.perf_counter() - t1) * 1000
        lists = {name: self._top(s, mask, self.retrieve_k, positive_only=(name != "dense")) for name, s in comp.items()}
        stages: dict = {name: len(v) for name, v in lists.items()}

        if method in ("bm25", "tfidf", "jaccard", "dense"):
            if method not in lists:
                raise ValueError(f"Retriever {method!r} is unavailable (index not built or model missing)")
            order = lists[method][:k]
            res = self._results(ctx, order, comp, {m: lists[m] for m in lists}, {}, {p: float(comp[method][p]) for p in order},
                                with_gap, None)
            return MatchResponse(candidate.candidate_id, method, res, stages, t, method, pool)

        fused, fused_scores, fused_ranks = rrf(lists, self.rrf_k)
        fused = fused[: self.retrieve_k]
        stages["rrf"] = len(fused)
        if method == "rrf" or not fused:
            order = fused[:k]
            res = self._results(ctx, order, comp, lists, fused_scores, {p: fused_scores[p] for p in order}, with_gap, None)
            return MatchResponse(candidate.candidate_id, method, res, stages, t, "rrf", pool)

        t2 = time.perf_counter()
        sub = {name: np.array([float(s[p]) for p in fused]) for name, s in comp.items()}
        sub["rrf"] = np.array([fused_scores[p] for p in fused])
        F = compute_features(ctx, self.store, fused, sub, self.tax)
        if drop_graph_features:
            for i, n in enumerate(FEATURE_NAMES):
                if n.startswith("graph_"):
                    F[:, i] = 0.0
        ltr = lambdamart.predict(F)
        if ltr is not None:
            stage_scores, ranker = ltr, "lambdamart"
        else:
            norm = {"rrf": minmax(sub["rrf"]), "dense": minmax(sub["dense"]) if "dense" in sub else np.zeros(len(fused))}
            stage_scores, ranker = fallback_score(F, norm), "fallback_weighted_features"
        t["features_ranker"] = (time.perf_counter() - t2) * 1000
        order_idx = np.argsort(-stage_scores, kind="stable")
        ranked = [fused[i] for i in order_idx]
        score_of = {fused[i]: float(stage_scores[i]) for i in range(len(fused))}
        feats = {fused[i]: dict(zip(FEATURE_NAMES, F[i].round(4).tolist())) for i in range(len(fused))}
        stages[ranker] = len(ranked)
        if method == "ltr":
            order = ranked[:k]
            res = self._results(ctx, order, comp, lists, fused_scores, score_of, with_gap, feats)
            return MatchResponse(candidate.candidate_id, method, res, stages, t, ranker, pool)

        top = ranked[: self.rerank_k]
        t3 = time.perf_counter()
        ce = score_pairs(ctx.query_text, [self.store.texts()[p] for p in top]) if use_cross_encoder else None
        base = minmax(np.array([score_of[p] for p in top]))
        if ce is not None:
            blended = (1 - self.ce_weight) * base + self.ce_weight * minmax(ce)
            stages["cross_encoder"] = len(top)
        else:
            blended = base
        t["cross_encoder"] = (time.perf_counter() - t3) * 1000
        t4 = time.perf_counter()
        if use_mmr and len(top) > 1:
            sim = self._similarity(top)
            pick = mmr(minmax(blended), sim, k, self.mmr_lambda)
            stages["mmr"] = len(pick)
        else:
            pick = list(np.argsort(-blended, kind="stable")[:k])
        t["mmr"] = (time.perf_counter() - t4) * 1000
        order = [top[i] for i in pick]
        final_scores = {top[i]: float(blended[i]) for i in range(len(top))}
        ce_scores = {top[i]: float(ce[i]) for i in range(len(top))} if ce is not None else {}
        res = self._results(ctx, order, comp, lists, fused_scores, final_scores, with_gap, feats,
                            extra={"ltr" if ranker == "lambdamart" else "fallback": score_of, "cross_encoder": ce_scores})
        t["total"] = (time.perf_counter() - t0) * 1000
        return MatchResponse(candidate.candidate_id, "final", res, stages, t,
                             ranker + ("+cross_encoder" if ce is not None else "") + ("+mmr" if use_mmr else ""), pool)

    def _similarity(self, positions: list[int]) -> np.ndarray:
        if self.dense is not None:
            V = self.dense.reconstruct(positions)
            return V @ V.T
        V = self.tfidf.vectors(positions)
        return (V @ V.T).toarray()

    def _results(self, ctx, order, comp, lists, fused_scores, final_scores, with_gap, feats, extra=None) -> list[MatchResult]:
        rank_maps = {name: {p: r + 1 for r, p in enumerate(lst)} for name, lst in lists.items()}
        out = []
        for r, p in enumerate(order, start=1):
            cs = {name: round(float(s[p]), 5) for name, s in comp.items()}
            if p in fused_scores:
                cs["rrf"] = round(fused_scores[p], 6)
            for name, sc in (extra or {}).items():
                if p in sc:
                    cs[name] = round(sc[p], 5)
            job = self.store.job(p)
            out.append(MatchResult(job_id=job.job_id, rank=r, score=round(float(final_scores.get(p, 0.0)), 5),
                                   component_scores=cs,
                                   component_ranks={name: m.get(p) for name, m in rank_maps.items()},
                                   features=(feats or {}).get(p, {}),
                                   gap=compute_gap(ctx.candidate, job, self.tax) if with_gap else None))
        return out

    # --------------------------------------------------- job -> candidates
    def recommend_candidates(self, job: Job, candidates: list[Candidate], k: int = 10) -> list[dict]:
        """Rank a pool of candidates for one job (BM25 + TF-IDF + Jaccard + dense, RRF, features)."""
        if not candidates:
            return []
        from matching.features import candidate_query_text

        texts = [candidate_query_text(c) for c in candidates]
        query = job.index_text()
        bm = BM25().fit(texts).scores(query)
        tf = TfidfRetriever().fit(texts).scores(query)
        job_sk = job.requirement_skill_ids()
        ctxs = [build_context(c, self.tax, None) for c in candidates]
        from matching.jaccard import jaccard

        jac = np.array([jaccard(c.skill_ids, frozenset(job_sk)) for c in ctxs])
        comps = {"bm25": bm, "tfidf": tf, "jaccard": jac}
        enc = get_encoder()
        if enc is not None:
            V = enc.encode(texts)
            comps["dense"] = V @ enc.encode([query])[0]
        lists = {n: list(np.argsort(-s, kind="stable")) for n, s in comps.items()}
        fused, fscores, franks = rrf(lists, self.rrf_k)
        req = job.requirement_skill_ids("required")
        out = []
        for i in fused:
            c = ctxs[i]
            cov = len(req & c.skill_ids) / len(req) if req else 0.0
            out.append({"candidate_id": candidates[i].candidate_id, "rrf": fscores[i], "required_coverage": cov,
                        "component_scores": {n: round(float(s[i]), 5) for n, s in comps.items()},
                        "component_ranks": franks[i]})
        rr = minmax(np.array([o["rrf"] for o in out]))
        for o, r in zip(out, rr):
            o["score"] = round(float(0.6 * r + 0.4 * o["required_coverage"]), 5)
        out.sort(key=lambda o: -o["score"])
        for r, o in enumerate(out[:k], 1):
            o["rank"] = r
            o["gap"] = compute_gap(next(c for c in candidates if c.candidate_id == o["candidate_id"]), job, self.tax)
        return out[:k]


@lru_cache(maxsize=1)
def get_pipeline() -> MatchingPipeline:
    return MatchingPipeline()


def reset_pipeline() -> None:
    get_pipeline.cache_clear()
    get_store.cache_clear()
