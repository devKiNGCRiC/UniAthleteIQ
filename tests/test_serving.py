"""Training/serving consistency: features computed from daily logs (as the API does)
must equal the features the model was trained on."""
import numpy as np
import pandas as pd
import pytest

from uniathleteiq import data, paths, serving

needs_data = pytest.mark.skipif(not paths.RUNNERS_DAY_CSV.exists(), reason="datasets/ CSVs not present")


@pytest.fixture(scope="module")
def runners():
    return pd.read_csv(paths.RUNNERS_DAY_CSV), data.load_runners("engineered+personal")


@needs_data
@pytest.mark.parametrize("athlete", [0, 5, 20, 41, 60])
def test_engineered_features_match_training(runners, athlete):
    day_df, ds = runners
    logs = serving.reconstruct_daily_logs(day_df, athlete)
    feats, info = serving.features_from_logs(logs, feature_names=ds.X.columns)

    mask = (ds.meta["Athlete ID"] == athlete).to_numpy()
    train = ds.X[mask].set_index(ds.meta.loc[mask, "Date"].astype(int))
    # Only days whose full 21-day history could be rebuilt (the dataset has gaps)
    ok = info.index[info["days_logged_last_21"] == serving.WINDOW_DAYS].intersection(train.index)
    assert len(ok) > 20  # some athletes have short records

    eng_cols = [c for c in ds.X.columns if not c.endswith("_vs_personal")]
    pd.testing.assert_frame_equal(feats.loc[ok, eng_cols], train.loc[ok, eng_cols],
                                  check_dtype=False, check_names=False, atol=1e-6)


@needs_data
def test_personal_features_match_for_gap_free_athlete(runners):
    """Personal baselines average over the athlete's history; identical when there are no gaps."""
    day_df, ds = runners
    dates = day_df.groupby("Athlete ID")["Date"]
    gap_free = [a for a, s in dates if (np.diff(np.sort(s)) == 1).all()]
    assert gap_free, "expected at least one athlete without date gaps"
    athlete = gap_free[0]

    # The research weekly file includes training from BEFORE the athlete's first daily row,
    # which cannot be rebuilt from the daily windows. So the comparison starts at the first
    # day with a fully rebuilt 21-day history, and both pipelines start their personal
    # baselines from that same day.
    logs = serving.reconstruct_daily_logs(day_df, athlete)
    serve_feats, info = serving.features_from_logs(logs, feature_names=ds.X.columns)
    mask = (ds.meta["Athlete ID"] == athlete).to_numpy()
    train = ds.X[mask].set_index(ds.meta.loc[mask, "Date"].astype(int))
    start = info.index[info["days_logged_last_21"] == serving.WINDOW_DAYS].min()
    rows = train.index[train.index >= start]
    assert len(rows) > data.PERSONAL_MIN_HISTORY + 30  # enough rows for personal z-scores to kick in

    eng_cols = [c for c in ds.X.columns if not c.endswith("_vs_personal")]

    def personal(eng):
        return data.personal_baseline_features(eng.loc[rows, eng_cols], pd.Series(athlete, index=rows),
                                               pd.Series(rows, index=rows))

    from_serving, from_training = personal(serve_feats), personal(train)
    assert (from_training.abs().sum(axis=1) > 0).sum() > 30  # the check is not vacuous (all zeros)
    pd.testing.assert_frame_equal(from_serving, from_training, check_dtype=False, check_names=False, atol=1e-6)
