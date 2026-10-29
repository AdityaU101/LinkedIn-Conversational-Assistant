"""Free-form SPO extraction, real cross-encoder scoring, agreement statistics."""
import pytest

from annotation.agreement import cohen_kappa, krippendorff_alpha, span_f1


def test_classical_spo_has_verb_predicates():
    from knowledge_graph.spo import classical_spo

    t = classical_spo("d1", "The analyst built dashboards with Tableau. Managed payroll for 200 employees.")
    preds = {x.predicate for x in t}
    assert preds and all(p.strip() and not p.startswith(" ") for p in preds)
    assert any(p.startswith("build") for p in preds) and any(x.subject == "<author>" for x in t)


def test_cross_encoder_prefers_relevant_passage(monkeypatch):
    import matching.cross_encoder as ce

    monkeypatch.delenv("DISABLE_CROSS_ENCODER", raising=False)
    monkeypatch.setattr(ce, "_failed", False)
    s = ce.score_pairs("python machine learning engineer", ["Senior ML engineer, Python, PyTorch, model training",
                                                           "Line cook for a busy restaurant kitchen"])
    if s is None:
        pytest.skip("cross-encoder model not available offline")
    assert s[0] > s[1]


def test_agreement_statistics():
    assert cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)
    k = cohen_kappa([0, 1, 2, 2, 1], [0, 2, 2, 1, 1], weights="quadratic")
    assert 0 < k < 1
    assert cohen_kappa([1, 0], [0, 1]) < 0
    # Krippendorff (1980) nominal example: perfect agreement -> 1
    assert krippendorff_alpha({"a": [1, 1], "b": [2, 2], "c": [3, 3]}) == pytest.approx(1.0)
    assert krippendorff_alpha({"a": [1, 2], "b": [2, 1]}) < 0.5
    f = span_f1({("d", "Skill", 0, 5)}, {("d", "Skill", 0, 5), ("d", "Skill", 9, 12)})
    assert f["precision"] == 0.5 and f["recall"] == 1.0
