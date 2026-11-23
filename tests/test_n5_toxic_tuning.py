import numpy as np
from sklearn.metrics import f1_score

from src.models.n5_fasttext import select_toxic_cutoff


def _expected_cutoff(scores, labels, default_cutoff):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    cutoffs = {float(value) for value in scores}
    cutoffs.add(float(default_cutoff))
    cutoffs.add(float(np.nextafter(scores.max(), np.inf)))
    f1_by_cutoff = {
        cutoff: f1_score(labels, scores >= cutoff, pos_label=1, zero_division=0)
        for cutoff in cutoffs
    }
    best_f1 = max(f1_by_cutoff.values())
    return max(cutoff for cutoff, score in f1_by_cutoff.items() if np.isclose(score, best_f1, rtol=0, atol=1e-12))


def test_select_toxic_cutoff_maximizes_validation_f1():
    labels = np.array([0, 0, 1, 1, 1, 0])
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.6, 0.4])
    cutoff = select_toxic_cutoff(scores, labels, default_cutoff=0.5)
    expected = _expected_cutoff(scores, labels, default_cutoff=0.5)
    assert cutoff == expected
    assert f1_score(labels, scores >= cutoff, pos_label=1, zero_division=0) == max(
        f1_score(labels, scores >= candidate, pos_label=1, zero_division=0)
        for candidate in {*scores, 0.5, np.nextafter(scores.max(), np.inf)}
    )


def test_select_toxic_cutoff_breaks_f1_ties_with_higher_cutoff():
    labels = np.array([1, 0, 0, 0, 1, 0])
    scores = np.array([0.4, 0.4, 0.4, 0.4, 0.6, 0.6])
    assert select_toxic_cutoff(scores, labels, default_cutoff=0.5) == 0.6


def test_select_toxic_cutoff_can_represent_all_negative_predictions():
    labels = np.zeros(3, dtype=int)
    scores = np.array([0.1, 0.9, 0.8])
    assert select_toxic_cutoff(scores, labels, default_cutoff=0.5) == np.nextafter(scores.max(), np.inf)