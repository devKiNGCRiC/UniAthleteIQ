"""Guards against the leakage problems found in v1. Needs the CSVs in datasets/."""
import numpy as np
import pandas as pd
import pytest

from uniathleteiq import data, evaluation as ev, paths

needs_data = pytest.mark.skipif(not paths.RUNNERS_DAY_CSV.exists(), reason="datasets/ CSVs not present")


@needs_data
def test_holdout_athletes_are_disjoint():
    r = data.load_runners("engineered")
    tr, te = ev.holdout_split(r)
    assert not set(r.groups.iloc[tr]) & set(r.groups.iloc[te])


@needs_data
def test_grouped_cv_never_shares_athletes():
    from sklearn.model_selection import StratifiedGroupKFold
    r = data.load_runners("engineered")
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(r.X, r.y, r.groups):
        assert not set(r.groups.iloc[tr]) & set(r.groups.iloc[te])


@needs_data
def test_football_uses_no_current_season_or_career_total_columns():
    fb = data.load_football()
    assert not set(data.FOOTBALL_LEAKY) & set(fb.X.columns)


@needs_data
def test_runner_feature_names_unique_and_finite():
    for fs in data.RUNNER_FEATURE_SETS:
        X = data.load_runners(fs).X
        assert X.columns.is_unique
        assert not np.isinf(X.to_numpy(dtype=float)).any()


@needs_data
def test_personal_baseline_uses_only_past_days():
    """Changing an athlete's future must not change today's personal-baseline feature."""
    r = data.load_runners("engineered")
    feats, athlete, date = r.X, r.meta["Athlete ID"], r.meta["Date"]
    a = athlete.iloc[0]
    rows = athlete[athlete == a].index
    cut = rows[len(rows) // 2]

    before = data.personal_baseline_features(feats, athlete, date)
    future = rows[rows > cut]
    tampered = feats.copy()
    tampered.loc[future, data.PERSONAL_BASE] = tampered.loc[future, data.PERSONAL_BASE] * 10 + 99
    after = data.personal_baseline_features(tampered, athlete, date)

    past = rows[rows <= cut]
    pd.testing.assert_frame_equal(before.loc[past], after.loc[past])


def test_threshold_for_recall_hits_target():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    p = np.array([.1, .2, .3, .4, .5, .6, .7, .8])
    t = ev.threshold_for_recall(y, p, 0.75)
    assert ((p >= t) & (y == 1)).sum() / y.sum() >= 0.75
