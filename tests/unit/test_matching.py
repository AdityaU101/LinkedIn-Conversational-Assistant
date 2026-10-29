"""BM25, TF-IDF, Jaccard, dense/FAISS, RRF, MMR, LambdaMART infrastructure, metrics."""
import math

import numpy as np
import pytest

from evaluation.ranking import evaluate_rankings, mrr, ndcg_at_k, precision_at_k
from matching.bm25 import BM25
from matching.jaccard import JaccardRetriever, jaccard
from matching.mmr import minmax, mmr
from matching.rrf import rrf
from matching.tfidf import TfidfRetriever

DOCS = ["python machine learning engineer", "java backend engineer", "registered nurse hospital",
        "python data analyst sql", "chef kitchen cook"]


def test_bm25_matches_hand_computation():
    bm = BM25(k1=1.5, b=0.75).fit(DOCS)
    s = bm.scores("python")
    n, df = 5, 2
    idf = math.log1p((n - df + 0.5) / (df + 0.5))
    dl, avg = 4, sum(len(d.split()) for d in DOCS) / n
    expected = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * dl / avg))
    assert s[0] == pytest.approx(expected, rel=1e-5)
    assert s[2] == 0 and np.argmax(s) in (0, 3)


def test_bm25_save_load(tmp_path):
    bm = BM25().fit(DOCS)
    bm.save(tmp_path / "b.pkl")
    assert np.allclose(BM25.load(tmp_path / "b.pkl").scores("nurse"), bm.scores("nurse"))


def test_tfidf_cosine():
    t = TfidfRetriever().fit(DOCS)
    s = t.scores("registered nurse")
    assert np.argmax(s) == 2 and 0 < s.max() <= 1.0001


def test_jaccard():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), {"a"}) == 0
    jr = JaccardRetriever([frozenset({"a", "b"}), frozenset({"c"}), frozenset()])
    assert list(jr.scores({"a"})) == pytest.approx([0.5, 0, 0])


def test_rrf_keeps_component_ranks():
    order, scores, ranks = rrf({"bm25": [1, 2, 3], "dense": [3, 1]}, k=60)
    assert order[0] == 1 and ranks[3] == {"bm25": 3, "dense": 1}
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)


def test_mmr_diversifies():
    rel = np.array([1.0, 0.99, 0.5])
    sim = np.array([[1, 0.99, 0.1], [0.99, 1, 0.1], [0.1, 0.1, 1]])
    assert mmr(rel, sim, 2, lam=0.5) == [0, 2]
    assert mmr(rel, sim, 2, lam=1.0) == [0, 1]
    assert list(minmax(np.array([2, 4]))) == [0, 1]


def test_dense_index_build_add_remove(tmp_path):
    from matching.dense import DenseIndex

    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(10, 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = DenseIndex(8, "test")
    idx.index.add_with_ids(vecs, np.arange(10))
    D, I = idx.search(vecs[3], 1)
    assert I[0] == 3 and D[0] == pytest.approx(1, abs=1e-5)
    idx.add(vecs[:1] * -1, [3])          # incremental update replaces id 3
    assert idx.search(vecs[3], 1)[1][0] != 3
    idx.remove([0])
    assert idx.index.ntotal == 9
    idx.save(tmp_path / "f.index")
    assert DenseIndex.load(tmp_path / "f.index").index.ntotal == 9


def test_ranking_metrics():
    assert ndcg_at_k([2, 1, 0], 3) == pytest.approx(1.0)
    assert ndcg_at_k([0, 1, 2], 3) < 1
    assert mrr([0, 0, 1]) == pytest.approx(1 / 3)
    assert precision_at_k([2, 0, 1, 0, 0], 5) == pytest.approx(0.4)
    res = evaluate_rankings({"q": ["a", "x", "b"]}, {"q": {"a": 2, "b": 0}})
    assert res["queries"] == 1 and res["ndcg@10"] == pytest.approx(1.0)


def test_lambdamart_awaits_annotations_and_trains_on_given_labels(tmp_path, monkeypatch):
    from matching import features, lambdamart

    monkeypatch.setattr(lambdamart, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(lambdamart, "judgments_path", lambda: tmp_path / "none.csv")
    st = lambdamart.status()
    assert st["state"] == "awaiting_annotations" and st["judgments"] == 0
    # unit test of the training code path only (test fixture data, never reported as results)
    rng = np.random.default_rng(1)
    X = rng.random((120, len(features.FEATURE_NAMES)))
    y = (X[:, 5] * 2.99).astype(int)
    groups = [f"c{i // 12}" for i in range(120)]
    out = lambdamart.train(X, y, groups, n_splits=3)
    assert out["pairs"] == 120 and out["cv_ndcg@10_mean"] is not None
    assert lambdamart.predict(X[:5]).shape == (5,)
