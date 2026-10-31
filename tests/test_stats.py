"""The fast weighted metrics must match sklearn exactly, including tied scores."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from uniathleteiq import stats


@pytest.mark.parametrize("tied", [False, True])
def test_point_metrics_match_sklearn(tied):
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(int)
    p = rng.random(5000) + y * 0.3
    if tied:  # coarse rounding creates many ties, like the binary "ACWR > 1.5" rule
        p = np.round(p, 1)
    m = stats.point_metrics(y, p)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p), abs=1e-12)


def test_weights_equal_row_duplication():
    """Weight 2 on a row must give the same metric as physically duplicating it."""
    rng = np.random.default_rng(2)
    y = (rng.random(300) < 0.2).astype(int)
    p = np.round(rng.random(300), 2)
    w = rng.integers(0, 3, size=300)
    yd, pd_ = np.repeat(y, w), np.repeat(p, w)
    got = stats._SortedScores(y, p).metrics(w)
    assert got["roc_auc"] == pytest.approx(roc_auc_score(yd, pd_), abs=1e-12)
    assert got["pr_auc"] == pytest.approx(average_precision_score(yd, pd_), abs=1e-12)


def test_bootstrap_resamples_whole_athletes():
    groups = np.repeat(np.arange(10), 5)  # 10 athletes x 5 rows each
    for w in stats.athlete_bootstrap_weights(groups, n_boot=20, seed=3):
        # every row of an athlete carries the same weight, and 10 athletes are drawn in total
        assert all(len(set(w[groups == a])) == 1 for a in range(10))
        assert w[::5].sum() == 10


def test_holm_adjust_known_values():
    np.testing.assert_allclose(stats.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
